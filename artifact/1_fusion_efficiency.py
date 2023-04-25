import argparse
import os
import sys
import time
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing.pool import ThreadPool

from thop import profile
from thop import clever_format
from smi import smi_getter
from resnet import resnet18
from vision_utils import get_datasets

from hydro.fx import symbolic_trace, fuse_model
from hydro.scale import (
    scale_model,
    scale_fused_model,
    hydro_optimizer,
    set_base_shapes,
    make_base_shapes,
)


parser = argparse.ArgumentParser(description="Run a Cifar experiment with Hydro")
parser.add_argument("--model", type=str, default="resnet18", help="Model to use")
parser.add_argument("--batch-size", type=int, default=256, help="batch size")
parser.add_argument("--fusion-num", type=int, default=1, help="fusion number")
parser.add_argument("--scale", type=int, default=1, help="Scale of model to shrink")
parser.add_argument("--epochs", type=int, default=4, help="upper epoch limit")
parser.add_argument("--dataset", type=str, default="cifar10", help="Choose dataset type")
parser.add_argument("--log_dir", type=str, default="./results", help="path to save logs")
parser.add_argument("--tag", type=str, default="hd_sgd", help="tags")
parser.add_argument("--amp", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument("--save_file", type=str, help="file name to save results")
parser.add_argument("--gpu", type=int, default=1, help="the ID of used GPU")
parser.add_argument("--co", type=int, help="MPS colocate number")
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
gpu_id = []
gpu_id.append(args.gpu)


def print_info():
    hostname = os.uname()[1]
    gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"hostname: {hostname}, gpu_id: {gpu_id}")


def train_epoch(dataloader, model, loss_fn, optimizer, fusion_num):
    if args.amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    model.train()
    num_batches = len(dataloader)

    # NOTE
    train_loss, correct, total = 0, 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # NOTE
        X = X.unsqueeze(1).expand(-1, fusion_num, -1, -1, -1).contiguous()
        y = y.repeat(fusion_num)

        optimizer.zero_grad()
        if args.amp:
            with torch.cuda.amp.autocast():
                pred = model(X)
                losses = loss_fn(reduction="none")(pred.contiguous().view(y.size(0), -1), y).view(fusion_num, -1).mean(dim=1)
                loss = losses.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X)
            losses = loss_fn(reduction="none")(pred.contiguous().view(y.size(0), -1), y).view(fusion_num, -1).mean(dim=1)
            loss = losses.mean()
            loss.backward()
            optimizer.step()

        # NOTE
        with torch.no_grad():
            train_loss += losses
            total += y.size(0) / fusion_num
            pred = pred.argmax(dim=2, keepdim=True)
            correct += pred.eq(y.view_as(pred)).view(fusion_num, -1).sum(dim=1).float()
    train_loss /= num_batches
    correct /= total
    train_record = {"train_loss": train_loss.tolist(), "train_acc": correct.tolist()}
    return train_record


def validate_epoch(dataloader, model, loss_fn, fusion_num):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # NOTE
            X = X.unsqueeze(1).expand(-1, fusion_num, -1, -1, -1).contiguous()
            y = y.repeat(fusion_num)

            pred = model(X)

            # NOTE
            test_loss += loss_fn(reduction="none")(pred.contiguous().view(y.size(0), -1), y).view(fusion_num, -1).mean(dim=1)
            total += y.size(0) / fusion_num
            pred = pred.argmax(dim=2, keepdim=True)
            correct += pred.eq(y.view_as(pred)).view(fusion_num, -1).sum(dim=1).float()
        test_loss /= num_batches
        correct /= total
        test_record = {"test_loss": test_loss.tolist(), "test_acc": correct.tolist()}
        return test_record


def convert_result_string(result):
    result_string = ""
    for k, v in result.items():
        rounded_v = [round(i, 4) for i in v]
        result_string += f"{k}: {rounded_v} | "
    return result_string


# print_info()
pool = ThreadPool(processes=1)


dataset = args.dataset
model = resnet18(dataset)
train_set, val_set = get_datasets(dataset)
train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)

"""
Hydro
"""
fusion_num = args.fusion_num
scaling_ratio = args.scale
lr = [0.01] * fusion_num

model = symbolic_trace(model)
base_model = fuse_model(model, 1)
base_scaled_model = scale_fused_model(base_model, scaling_ratio)
base_shape = make_base_shapes(base_model, base_scaled_model)
scaled_model = scale_model(model, scaling_ratio)
fused_model = fuse_model(scaled_model, fusion_num)
set_base_shapes(fused_model, base_shape)
model = fused_model

# Calculate number of parameters and MACs
features, targets = next(iter(train_loader))
dsize = list(features.shape)
dsize[0] = 1
dinput = torch.randn(torch.Size(dsize))
total_ops, total_params = profile(scaled_model, inputs=(dinput,), verbose=False)
total_ops *= fusion_num
total_params *= fusion_num
macs, params = clever_format([total_ops, total_params], "%.3f")
print(f"MACs: {macs}, Paras: {params}")

device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.CrossEntropyLoss
optimizer = hydro_optimizer(optim.SGD, model.parameters(), fusion_num, scaling_ratio, lr=lr)
model = model.to(device)

smi_list = []
time_list = []
for epoch in range(1, args.epochs + 1):
    if epoch == 1:
        pool.apply_async(
            smi_getter,
            args=(
                sys.argv[1:],
                smi_list,
                gpu_id,
            ),
        )

    epoch_start_time = time.time()
    train_result = train_epoch(train_loader, model, criterion, optimizer, fusion_num)
    test_result = validate_epoch(val_loader, model, criterion, fusion_num)

    t1 = time.time() - epoch_start_time
    time_list.append(t1)

    if epoch == 1:
        pool.terminate()

    print("-" * 89)
    print(f"|epoch {epoch:3} | time: {t1:5.2f}s | {convert_result_string(train_result)} {convert_result_string(test_result)}")

# Process GPU info
smi_df = pd.DataFrame(smi_list)
n = 3
smi_df.drop(smi_df.head(n).index, inplace=True)
smi_df.drop(smi_df.tail(n).index, inplace=True)
smi_df["gpuMem"] = smi_df["gpuMem"].apply(lambda x: x[:-4]).astype("int64")

if args.save_file is not None:
    if args.co is not None:
        dict = {
            "dataset": args.dataset,
            "model": args.model,
            "bs": args.batch_size,
            "amp": args.amp,
            "fusion": args.fusion_num,
            "scale": args.scale,
            "speed(s)": time_list[2],
            "parameters": params,
            "MACs": macs,
            "MPS": args.co,
        }
    else:
        dict = {
            "dataset": args.dataset,
            "model": args.model,
            "bs": args.batch_size,
            "amp": args.amp,
            "fusion": args.fusion_num,
            "scale": args.scale,
            "speed(s)": time_list[2],
            "parameters": params,
            "MACs": macs,
        }
    df = pd.DataFrame([dict])
    df["gpu_util"] = round(pd.to_numeric(smi_df["gpuUtil"]).mean(), 3)
    df["gmem_util"] = round(pd.to_numeric(smi_df["gpuMemUtil"]).mean(), 3)
    df["gmem"] = round(smi_df["gpuMem"].max(), 3)

    file_csv = "./results/" + args.save_file
    if os.path.exists(file_csv):
        df.to_csv(file_csv, mode="a", header=False)
    else:
        df.to_csv(file_csv, mode="a")

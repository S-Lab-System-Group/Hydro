import argparse
from resnet import resnet18
from vision_utils import get_datasets, fix_seed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.air import session
from ray.air.config import FailureConfig, RunConfig, ScalingConfig, CheckpointConfig

from hydro import HydroTuner, HydroTrainer
from hydro.tune.tune_config import TuneConfig
import hydro.train as ht


SEARCH_SPACE = {
    "lr": tune.qloguniform(1e-4, 1, 1e-4),
    "momentum": tune.quniform(0.5, 0.999, 0.001),
    "batch_size": tune.choice([128, 256, 512]),
    "gamma": tune.quniform(0.01, 0.9, 0.01),
}


def train_epoch(dataloader, model, loss_fn, optimizer, fusion_num):
    model.train()
    for _, (X, y) in enumerate(dataloader):
        if fusion_num > 0:
            X = X.unsqueeze(1).expand(-1, fusion_num, -1, -1, -1).contiguous()
            y = y.repeat(fusion_num)

        pred = model(X)

        if fusion_num > 0:
            losses = (
                loss_fn.__class__(reduction="none")(pred.contiguous().view(y.size(0), -1), y).view(fusion_num, -1).mean(dim=1)
            )
            loss = losses.mean()
        else:
            loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # ht.backward(loss)  # For AMP support
        optimizer.step()


def validate_epoch(dataloader, model, loss_fn, fusion_num):
    size = len(dataloader.dataset) // session.get_world_size()
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            if fusion_num > 0:
                X = X.unsqueeze(1).expand(-1, fusion_num, -1, -1, -1).contiguous()
                y = y.repeat(fusion_num)

            pred = model(X)

            if fusion_num > 0:
                test_loss += (
                    loss_fn.__class__(reduction="none")(pred.contiguous().view(y.size(0), -1), y)
                    .view(fusion_num, -1)
                    .mean(dim=1)
                )
                pred = pred.argmax(dim=2, keepdim=True)
                correct += pred.eq(y.view_as(pred)).view(fusion_num, -1).sum(dim=1).float()
            else:
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if fusion_num > 0:
        return {"loss": test_loss.cpu().detach().tolist(), "val_acc": correct.cpu().detach().tolist()}
    else:
        return {"loss": test_loss, "val_acc": correct}


def train_func(config):
    ht.accelerate(config, amp=False)  # For AMP support
    ht.enable_reproducibility(seed=config["seed"])
    fusion_num = config.get("FUSION_N", 0)

    dataset = config.get("dataset")
    model = resnet18(dataset)
    model = ht.prepare_model(model)

    optimizer = ht.prepare_optimizer(
        torch.optim.SGD,
        model.parameters(),
        lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 0.001),
    )

    lr_scheduler = ht.prepare_scheduler(
        torch.optim.lr_scheduler.StepLR, optimizer, step_size=20, gamma=config.get("gamma", 0.2)
    )

    worker_batch_size = config["batch_size"] // session.get_world_size()
    train_set, val_set = get_datasets(dataset)
    train_loader = DataLoader(train_set, batch_size=worker_batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=worker_batch_size, num_workers=4, pin_memory=True)
    train_loader = ht.prepare_data_loader(train_loader)
    val_loader = ht.prepare_data_loader(val_loader)

    criterion = nn.CrossEntropyLoss()

    for _ in range(10000):
        train_epoch(train_loader, model, criterion, optimizer, fusion_num)
        result = validate_epoch(val_loader, model, criterion, fusion_num)
        lr_scheduler.step()
        session.report(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument("--max-epoch", default=50, type=int, help="Max Epochs")
    parser.add_argument("--max-sample", default=50, type=int, help="Max Samples")
    parser.add_argument("--seed", default=10, type=int, help="Fix Random Seed for Reproducing")

    args, _ = parser.parse_known_args()

    fix_seed(args.seed)
    ray.init(address=None)
    config = SEARCH_SPACE | {
        "dataset": args.dataset,
        "seed": args.seed,
    }

    trainer = HydroTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
            resources_per_worker={"CPU": 4, "GPU": 1},
        ),
    )

    tuner = HydroTuner(
        trainer,
        param_space={"train_loop_config": config},
        tune_config=TuneConfig(
            num_samples=args.max_sample,
            metric="val_acc",
            mode="max",
            scheduler="fifo",
            scaling_num=8,  # Hydro args
            fusion_limit=10,  # Hydro args
        ),
        run_config=RunConfig(
            log_to_file=True,
            stop={"training_iteration": args.max_epoch},
            checkpoint_config=CheckpointConfig(num_to_keep=1),
            failure_config=FailureConfig(fail_fast=True, max_failures=0),
        ),
    )

    results = tuner.fit()
    df = results.get_dataframe()
    df.to_csv(f"./results/hydro.csv")

    from ast import literal_eval
    import pandas as pd

    df = pd.read_csv(f"./results/hydro.csv")
    df.dropna(axis=1, inplace=True)

    cols = [
        "loss",
        "val_acc",
        "config/train_loop_config/gamma",
        "config/train_loop_config/lr",
        "config/train_loop_config/momentum",
    ]
    for col in cols:
        df[col] = df[col].apply(literal_eval)

    new_df = df.explode(cols, ignore_index=True)
    new_df.to_csv(f"./results/hydro_parsed.csv")

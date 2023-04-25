import os
import argparse
from resnet import resnet18
from vision_utils import get_datasets, fix_seed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ray
from ray.air import session
from ray.air.config import ScalingConfig

from hydro import HydroTrainer
import hydro.train as ht


POINTS_TO_EVALUATE = [
    {"lr": 0.01, "momentum": 0.9, "batch_size": 512},
    {"lr": 0.002, "momentum": 0.9, "batch_size": 512},
    {"lr": 0.0006, "momentum": 0.9, "batch_size": 512},
    {"lr": 0.0004, "momentum": 0.95, "batch_size": 512},
    {"lr": 0.001, "momentum": 0.7, "batch_size": 256},
    {"lr": 0.005, "momentum": 0.6, "batch_size": 128},
    {"lr": 0.2, "momentum": 0.99, "batch_size": 128},
    {"lr": 0.3, "momentum": 0.6, "batch_size": 128},
]


def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    for _, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset) // session.get_world_size()
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return {"loss": test_loss, "val_acc": correct}


def train_func(config):
    ht.accelerate(config, amp=False)  # For AMP support
    ht.enable_reproducibility(seed=config["seed"])

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
        torch.optim.lr_scheduler.StepLR, optimizer, step_size=20, gamma=config.get("gamma", 0.3)
    )

    worker_batch_size = config["batch_size"] // session.get_world_size()
    train_set, val_set = get_datasets(dataset)
    train_loader = DataLoader(train_set, batch_size=worker_batch_size, num_workers=8, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=worker_batch_size, num_workers=8, pin_memory=True)
    train_loader = ht.prepare_data_loader(train_loader)
    val_loader = ht.prepare_data_loader(val_loader)

    criterion = nn.CrossEntropyLoss()

    for _ in range(100):
        train_epoch(train_loader, model, criterion, optimizer)
        result = validate_epoch(val_loader, model, criterion)
        lr_scheduler.step()
        session.report(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default=1, type=int, help="Scale Number")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument("--seed", default=1, type=int, help="Fix Random Seed for Reproducing")

    args, _ = parser.parse_known_args()

    fix_seed(args.seed)
    ray.init(address=None, num_gpus=1)

    for point in POINTS_TO_EVALUATE:
        config = point | {
            "dataset": args.dataset,
            "seed": args.seed,
            "SCALING_N": args.scale,
        }
        print(config)

        trainer = HydroTrainer(
            train_func,
            train_loop_config=config,
            scaling_config=ScalingConfig(
                num_workers=1,
                use_gpu=True,
                resources_per_worker={"CPU": 4, "GPU": 1},
            ),
        )

        result = trainer.fit()
        result = result.metrics

        logfile = f"./results/scaling_fidelity_seed{args.seed}.csv"
        if not os.path.exists(logfile):
            with open(logfile, "w") as f:
                f.write(f"batch_size,lr,momentum,epoch,scale,loss,val_acc\n")

        with open(logfile, "a") as f:
            record = f"{config['batch_size']},{config['lr']},{config['momentum']},100,{args.scale},{result['loss']},{result['val_acc']}\n"
            f.write(record)

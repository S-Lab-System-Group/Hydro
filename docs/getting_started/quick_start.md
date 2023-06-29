# **Getting started**

In this walkthrough, we will present how to tune an image classifier (e.g., ResNet-18 model) on CIFAR-10 dataset using Hydro.


!!! tip "Tip"

    Please refer to [Ray Docs](https://docs.ray.io/en/latest/) for more information if you are not familiar with Ray Tune.
    


## **Installation**
To run this example, we need to install Hydro package beforehand. Further installation instructions can be found in [here](installation.md).


=== "pip"
    ``` sh
    pip install hydro-tune
    ```

## **Import Libraries**
Let's begin by importing the necessary modules:

``` python linenums="1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import ray
from ray import tune
from ray.air.config import RunConfig, ScalingConfig

from hydro import HydroTuner, HydroTrainer, session
from hydro.tune.tune_config import TuneConfig
import hydro.train as ht

from filelock import FileLock
from pathlib import Path
```

## **Setup the Search Space**
We need to define search space with Ray Tune API. Here is an example:
``` python linenums="1"
SEARCH_SPACE = {
    "lr": tune.qloguniform(1e-4, 1, 1e-4),
    "momentum": tune.quniform(0.5, 0.999, 0.001),
    "batch_size": tune.choice([128, 256, 512]),
}
```
The `tune.qloguniform(lower, upper, q)` function samples in different orders of magnitude and quantizes the value to an integer increment of `q`. For more search space functions, please refer to [Ray Tune Search Space API](https://docs.ray.io/en/latest/tune/api/search_space.html#tune-search-space).

## **Load Dataset**
We first load the CIFAR10 dataset and use a `FileLock` to prevent multiple processes from downloading the same data.

``` python linenums="1"
def get_dataset():
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    with FileLock(Path("~/data/data.lock").expanduser()):
        train_dataset = datasets.CIFAR10(
            root="~/data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
            ),
        )
        val_dataset = datasets.CIFAR10(
            root="~/data", train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), normalize])
        )
    return train_dataset, val_dataset
```


## **Train & Validation Function**
To support inter-trial fusion feature, we add `fusion_num` as an argument to the train and validation functions. Besides, we need to incorporate some code to resize specific tensors as highlighted below.


``` python linenums="1" hl_lines="4-6 10-14 30-32 36-43 49-50"
def train_epoch(dataloader, model, optimizer, fusion_num):
    model.train()
    for _, (X, y) in enumerate(dataloader):
        if fusion_num > 0:
            X = X.unsqueeze(1).expand(-1, fusion_num, -1, -1, -1).contiguous()
            y = y.repeat(fusion_num)

        pred = model(X)

        if fusion_num > 0:
            losses = (
                nn.CrossEntropyLoss(reduction="none")(pred.contiguous().view(y.size(0), -1), y).view(fusion_num, -1).mean(dim=1)
            )
            loss = losses.mean()
        else:
            loss = nn.CrossEntropyLoss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_epoch(dataloader, model, fusion_num):
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
                    nn.CrossEntropyLoss(reduction="none")(pred.contiguous().view(y.size(0), -1), y)
                    .view(fusion_num, -1)
                    .mean(dim=1)
                )
                pred = pred.argmax(dim=2, keepdim=True)
                correct += pred.eq(y.view_as(pred)).view(fusion_num, -1).sum(dim=1).float()
            else:
                test_loss += nn.CrossEntropyLoss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if fusion_num > 0:
        return {"loss": test_loss.cpu().detach().tolist(), "val_acc": correct.cpu().detach().tolist()}
    else:
        return {"loss": test_loss, "val_acc": correct}
```

## **Wrap Model, Optimizer and DataLoader**
We need to wrap model, optimizer and dataLoader with `hydro.train` api.

``` python linenums="1" hl_lines="6 8 20 21"
def train_func(config):
    ht.accelerate(config)
    fusion_num = config.get("FUSION_N", 0)

    model = torchvision.models.__dict__["resnet18"]()
    model = ht.prepare_model(model)

    optimizer = ht.prepare_optimizer(
        torch.optim.SGD,
        model.parameters(),
        lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 0.001),
    )

    worker_batch_size = config["batch_size"] // session.get_world_size()
    train_set, val_set = get_dataset()
    train_loader = DataLoader(train_set, batch_size=worker_batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=worker_batch_size, pin_memory=True)
    train_loader = ht.prepare_data_loader(train_loader)
    val_loader = ht.prepare_data_loader(val_loader)

    for _ in range(10000): # Not determine the actual epoch number
        train_epoch(train_loader, model, optimizer, fusion_num)
        result = validate_epoch(val_loader, model, fusion_num)
        session.report(result)
```

## **Configure Tuner**

`HydroTuner` is the key interface of configuring hyperparameter tuning job. Users can specify maximum number of trials `num_samples`, maximum epochs `stop`, model scaling ratio `scaling_num` and inter-trial fusion limit `fusion_limit`.

``` python linenums="1"
if __name__ == "__main__":
    ray.init(address=None)

    trainer = HydroTrainer(
        train_func,
        train_loop_config=SEARCH_SPACE,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
            resources_per_worker={"CPU": 2, "GPU": 1},
        ),
    )

    tuner = HydroTuner(
        trainer,
        param_space={"train_loop_config": SEARCH_SPACE},
        tune_config=TuneConfig(
            num_samples=50,
            metric="val_acc",
            mode="max",
            scaling_num=8,  # Hydro args
            fusion_limit=10,  # Hydro args
        ),
        run_config=RunConfig(
            stop={"training_iteration": 50},
        ),
    )

    results = tuner.fit()
```

## **Example Output**

After tuning the models, we will find the best performing one and load the trained network from the checkpoint file. We then obtain the test set accuracy and report everything by printing.

If you run the code, an example output could look like this:

```sh
== Status ==
Current time: 2023-04-25 08:20:42 (running for 00:23:42.54)
Memory usage on this node: 25.5/251.5 GiB 
Using FIFO scheduling algorithm.
Resources requested: 0/64 CPUs, 0/4 GPUs, 0.0/157.96 GiB heap, 0.0/71.69 GiB objects (0.0/1.0 accelerator_type:G)
Current best trial: b38f6_T0001(target trial) with val_acc=0.9162 and parameters={'lr': 0.1102, 'momentum': 0.584, 'batch_size': 128, 'gamma': 0.14, 'dataset': 'cifar10', 'seed': 10, 'FUSION_N': 0, 'SCALING_N': 0}
Result logdir: ~/ray_results
Number of trials: 8/50 (8 TERMINATED)
+--------------------------+------------+----------------------+----------+------+----------------------+----------------------+----------------------+--------+------------------+--------------+---------------------+-----------------------+
| Trial name               | status     | loc                  | hydro    |   bs | gamma                | lr                   | momentum             |   iter |   total time (s) |   _timestamp |   _time_this_iter_s |   _training_iteration |
|--------------------------+------------+----------------------+----------+------+----------------------+----------------------+----------------------+--------+------------------+--------------+---------------------+-----------------------|
| HydroTrainer_b38f6_T0001 | TERMINATED | 10.100.79.96:3657182 | Target   |  128 | 0.14                 | 0.1102               | 0.584                |     50 |          384.496 |   1682382041 |             7.71278 |                    50 |
| HydroTrainer_b38f6_T0000 | TERMINATED | 10.100.79.96:3479472 | Target   |  512 | 0.05                 | 0.1827               | 0.846                |     50 |          279.453 |   1682381326 |             5.47435 |                    50 |
| HydroTrainer_b38f6_F0000 | TERMINATED | 10.100.79.96:3223763 | F=9, S=8 |  256 | [0.74, 0.38, 0._df80 | [0.0204, 0.4689_6d40 | [0.584, 0.857, _dec0 |     50 |          427.166 |   1682381050 |             8.64703 |                    50 |
| HydroTrainer_b38f6_F0001 | TERMINATED | 10.100.79.96:3223967 | F=8, S=8 |  256 | [0.32, 0.31, 0._fb80 | [0.013600000000_23c0 | [0.507, 0.69400_6500 |     50 |          415.149 |   1682381041 |             9.09435 |                    50 |
| HydroTrainer_b38f6_F0002 | TERMINATED | 10.100.79.96:3223968 | F=9, S=8 |  512 | [0.46, 0.28, 0._2b00 | [0.004500000000_f080 | [0.615, 0.659, _2180 |     50 |          382.011 |   1682381008 |             7.61978 |                    50 |
| HydroTrainer_b38f6_F0003 | TERMINATED | 10.100.79.96:3223969 | F=8, S=8 |  512 | [0.04, 0.4, 0.0_eb40 | [0.0011, 0.1303_eec0 | [0.65, 0.791, 0_fe00 |     50 |          357.47  |   1682380984 |             6.90358 |                    50 |
| HydroTrainer_b38f6_F0004 | TERMINATED | 10.100.79.96:3451196 | F=8, S=8 |  128 | [0.14, 0.54, 0._8280 | [0.1102, 0.2675_a200 | [0.584, 0.675, _8400 |     50 |          773.026 |   1682381761 |            15.4453  |                    50 |
| HydroTrainer_b38f6_F0005 | TERMINATED | 10.100.79.96:3464377 | F=8, S=8 |  128 | [0.42, 0.54, 0._a140 | [0.307100000000_e880 | [0.981, 0.81800_8f40 |     50 |          737.031 |   1682381749 |            14.6641  |                    50 |
+--------------------------+------------+----------------------+----------+------+----------------------+----------------------+----------------------+--------+------------------+--------------+---------------------+-----------------------+
```

## **See More PyTorch Examples**

### **`vision`: Image Classification Example**

- [run_hydro.py](https://github.com/S-Lab-System-Group/Hydro/tree/master/examples/vision/run_hydro.py)

    Tuning ResNet-18 on CIFAR-10 dataset using Hydro.

- [run_ray.py](https://github.com/S-Lab-System-Group/Hydro/tree/master/examples/vision/run_ray.py)

    The original Ray Tune script for reference.

### **`language`: Language Modeling Example**

- [run_hydro_lm.py](https://github.com/S-Lab-System-Group/Hydro/tree/master/examples/language/run_hydro_lm.py)

    Tuning HuggingFace GPT-2 on WikiText dataset using Hydro. To be compatible with most machines, we set `n_layer=2` by default.

- [run_ray_lm.py](https://github.com/S-Lab-System-Group/Hydro/tree/master/examples/language/run_ray_lm.py)

    The original Ray Tune script for reference.
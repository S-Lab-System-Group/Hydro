import random
import numpy as np

import torch
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pathlib import Path
from filelock import FileLock


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # # If you want to use deterministic algorithms with CUDA, then you need to set
    # # the CUBLAS_WORKSPACE_CONFIG environment variable; otherwise, Torch errors.
    # # See https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility.
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_datasets(dataset):
    """Data loader for Cifar10/100 & Imagenet"""
    if dataset == "imagenet":
        traindir = Path("/home/data/imagenet") / "train"
        valdir = Path("/home/data/imagenet") / "val"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        with FileLock(Path("~/data/data.lock").expanduser()):
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
                ),
            )
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]),
            )
    elif dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        with FileLock(Path("~/data/data.lock").expanduser()):
            train_dataset = datasets.CIFAR100(
                root="~/data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
                ),
            )
            val_dataset = datasets.CIFAR100(
                root="~/data", train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), normalize])
            )
    elif dataset == "cifar10":
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
    else:
        raise ValueError("Incorrect dataset name.")
    return train_dataset, val_dataset


def get_dataloaders(dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=False):
    train_set, val_set = get_datasets(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    return train_loader, train_loader

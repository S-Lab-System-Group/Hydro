import os
import math
import time
import random
import numpy as np
import logging
from enum import Enum

import torch
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist

from pathlib import Path
from filelock import FileLock

"""Default
DATAPATH = '~/data'
DATASET = 'cifar10'  # 'cifar100', 'cifar10', 'imagenet'
"""


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


def logger_init(file):
    logger = logging.getLogger()
    handler_file = logging.FileHandler(f"{file}.log", "w")
    handler_stream = logging.StreamHandler()  # sys.stdout

    logger.setLevel(logging.INFO)
    handler_file.setLevel(logging.INFO)
    handler_stream.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(processName)s | %(message)s", datefmt="%Y %b %d %H:%M:%S")
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)

    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)

    return logger


def get_datasets(dataset):
    """Data loader for Cifar10/100 & Imagenet"""
    if dataset == "imagenet":
        # traindir = os.path.join("/home/data/imagenet", "train")
        # valdir = os.path.join("/home/data/imagenet", "val")
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
        lock_file = Path("./data/data.lock")

        if not lock_file.exists():
            if not Path("./data").exists():
                Path("./data").mkdir()
            with open(lock_file, "w") as f:
                f.write("")

        with FileLock(lock_file):
            train_dataset = datasets.CIFAR100(
                root="./data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
                ),
            )
            val_dataset = datasets.CIFAR100(
                root="./data", train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), normalize])
            )
    elif dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        lock_file = Path("./data/data.lock")

        if not lock_file.exists():
            if not Path("./data").exists():
                Path("./data").mkdir()
            with open(lock_file, "w") as f:
                f.write("")

        with FileLock(lock_file):
            train_dataset = datasets.CIFAR10(
                root="./data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
                ),
            )
            val_dataset = datasets.CIFAR10(
                root="./data", train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), normalize])
            )
    else:
        raise ValueError("Incorrect dataset name.")
    return train_dataset, val_dataset


def get_dataloaders(dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=False):
    train_set, val_set = get_datasets(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    return train_loader, train_loader


# def train(train_loader, model, criterion, optimizer, max_steps=None):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     timers = {k: TimerStat() for k in ["data2gpu", "forward", "gradient", "step"]}
#     # switch to train mode
#     model.train()

#     end = time.time()
#     train_start = time.time()
#     for i, (images, target) in enumerate(train_loader):
#         if max_steps and i > max_steps:
#             break

#         # measure data loading time
#         data_time.update(time.time() - end)

#         with timers["data2gpu"]:
#             images = images.cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)

#         with timers["forward"]:
#             # compute output
#             output = model(images)
#             loss = criterion(output, target)

#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1[0], images.size(0))
#             top5.update(acc5[0], images.size(0))

#         with timers["gradient"]:
#             # compute gradient and do SGD step
#             optimizer.zero_grad()
#             loss.backward()

#         with timers["step"]:
#             optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#     stats = {
#         "train_accuracy": top1.avg.cpu(),
#         "batch_time": batch_time.avg,
#         "train_loss": losses.avg,
#         "data_time": data_time.avg,
#         "train_time": time.time() - train_start,
#     }
#     stats.update({k: t.mean for k, t in timers.items()})
#     return stats


# def validate(val_loader, model, criterion, max_steps=None):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()
#     val_start = time.time()
#     with torch.no_grad():
#         end = time.time()
#         for i, (images, target) in enumerate(val_loader):
#             if max_steps and i > max_steps:
#                 break

#             images = images.cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)

#             # compute output
#             output = model(images)
#             loss = criterion(output, target)

#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1[0], images.size(0))
#             top5.update(acc5[0], images.size(0))

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#     stats = {
#         "val_accuracy": top1.avg.cpu(),
#         "batch_time": batch_time.avg,
#         "val_loss": losses.avg,
#         "val_time": time.time() - val_start,
#     }
#     return stats


class TimerStat:
    """A running stat for conveniently logging the duration of a code block.
    Example:
        wait_timer = TimerStat()
        with wait_timer:
            ray.wait(...)
    Note that this class is *not* thread-safe.
    """

    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, exc_type, exc_value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    def push_units_processed(self, n):
        self._units_processed.append(n)
        if len(self._units_processed) > self._window_size:
            self._units_processed.pop(0)

    def has_units_processed(self):
        return len(self._units_processed) > 0

    @property
    def mean(self):
        if not self._samples:
            return 0.0
        return float(np.mean(self._samples))

    @property
    def mean_units_processed(self):
        if not self._units_processed:
            return 0.0
        return float(np.mean(self._units_processed))

    @property
    def mean_throughput(self):
        time_total = float(sum(self._samples))
        if not time_total:
            return 0.0
        return float(sum(self._units_processed)) / time_total


class Adjuster:
    """A ramp-up period for increasing learning rate"""

    def __init__(self, initial, target=None, steps=None, mode=None):
        self.mode = mode
        self._initial = initial
        self._lr = initial
        self._target = initial
        if target and self.mode is not None:
            self._target = max(target, initial)
        self.steps = steps or 10
        print(f"Creating an adjuster from {initial} to {target}: mode {mode}")

    def adjust(self):
        if self._lr < self._target:
            diff = (self._target - self._initial) / self.steps
            self._lr += min(diff, 0.1 * self._initial)
        return self._lr

    @property
    def current_lr(self):
        return self._lr


def adjust_learning_rate(initial_lr, optimizer, epoch, decay=True):
    optim_factor = 0
    if decay:
        if epoch > 160:
            optim_factor = 3
        elif epoch > 120:
            optim_factor = 2
        elif epoch > 60:
            optim_factor = 1

    lr = initial_lr * math.pow(0.2, optim_factor)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

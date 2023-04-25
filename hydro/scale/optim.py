from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple, Iterable
import logging
from collections import defaultdict
from numbers import Number

import torch
import torch.nn as nn
import torch.fx as fx
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR

from hydro.optim import fuse_optimizer
from hydro.optim import OPTIMIZERS_MAP, LR_SCHEDULER_MAP


logger = logging.getLogger(__name__)

_SUPPORTED_OPTIM_LIST = [SGD, Adam, AdamW]
_SUPPORTED_LRSCH_LIST = [StepLR]


def _process_param_groups(params, **kwargs):
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{"params": param_groups}]
    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.0)
    return param_groups


def process_adam_like(params, scaling_ratio: Union[int, float] = 2, decoupled_wd=False, **kwargs):
    new_param_groups = []
    for param_group in _process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != "params"}
            new_g["params"] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        matrix_like_p = defaultdict(new_group)  # key is width_mult
        vector_like_p = new_group()
        for p in param_group["params"]:

            if p.infshape.ninf() == 2:
                matrix_like_p[scaling_ratio]["params"].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError("more than 2 dimensions")
            else:
                vector_like_p["params"].append(p)
        for scaling_ratio, group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            if isinstance(group["lr"], Number):
                group["lr"] *= scaling_ratio
            else:
                group["lr"] = [lr * scaling_ratio for lr in group["lr"]]
            if not decoupled_wd:
                if isinstance(group["weight_decay"], Number):
                    group["weight_decay"] /= scaling_ratio
                else:
                    group["weight_decay"] = [wd / scaling_ratio for wd in group["weight_decay"]]
        new_param_groups.extend(list(matrix_like_p.values()) + [vector_like_p])
    return new_param_groups


def process_sgd_like(params, scaling_ratio: Union[int, float] = 2, decoupled_wd=False, **kwargs):
    new_param_groups = []
    for param_group in _process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != "params"}
            new_g["params"] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        matrix_like_p = defaultdict(new_group)  # key is width_mult
        vector_like_p = defaultdict(new_group)  # key is fan_in/out ratio
        fixed_p = new_group()
        for p in param_group["params"]:

            if p.infshape.ninf() == 1:
                vector_like_p[scaling_ratio]["params"].append(p)
            elif p.infshape.ninf() == 2:
                matrix_like_p[scaling_ratio]["params"].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError("more than 2 dimensions")
            else:
                fixed_p["params"].append(p)
        for scaling_ratio, group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            if isinstance(group["lr"], Number):
                group["lr"] *= scaling_ratio
            else:
                group["lr"] = [lr * scaling_ratio for lr in group["lr"]]
            if not decoupled_wd:
                if isinstance(group["weight_decay"], Number):
                    group["weight_decay"] /= scaling_ratio
                else:
                    group["weight_decay"] = [wd / scaling_ratio for wd in group["weight_decay"]]
        for scaling_ratio, group in vector_like_p.items():
            # Scale learning rate and weight decay accordingly
            if isinstance(group["lr"], Number):
                group["lr"] /= scaling_ratio
            else:
                group["lr"] = [lr / scaling_ratio for lr in group["lr"]]
            if not decoupled_wd:
                if isinstance(group["weight_decay"], Number):
                    group["weight_decay"] *= scaling_ratio
                else:
                    group["weight_decay"] = [wd * scaling_ratio for wd in group["weight_decay"]]

        new_param_groups.extend(list(matrix_like_p.values()) + list(vector_like_p.values()) + [fixed_p])
    return new_param_groups


def hydro_optimizer(
    optim: torch.optim.Optimizer,
    params: Any,
    fusion_num: int = -1,
    scaling_num: Union[int, float] = -1,
    initial_lr: Optional[float] = None,
    decoupled_wd=False,
    **kwargs,
):
    assert optim in _SUPPORTED_OPTIM_LIST, f"{optim} is not supported"

    if initial_lr is not None:
        params = [{"params": params, "initial_lr": initial_lr}]

    if fusion_num > 0 and scaling_num > 0:
        optim = OPTIMIZERS_MAP[optim]
        kwargs = kwargs | {"scaling_num": scaling_num}
        return optim(params, **kwargs)

    if scaling_num > 0:
        if optim == SGD:
            new_param = process_sgd_like(params, scaling_num, decoupled_wd, **kwargs)
        elif optim == Adam or optim == AdamW:
            new_param = process_adam_like(params, scaling_num, decoupled_wd, **kwargs)
        params = new_param

    if fusion_num > 0:
        optim = OPTIMIZERS_MAP[optim]
    return optim(params, **kwargs)


def hydro_lr_scheduler(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    optimizer: torch.optim.Optimizer,
    fusion_num: int = -1,
    last_epoch: int = -1,
    **kwargs,
):
    assert scheduler in _SUPPORTED_LRSCH_LIST, f"{scheduler} is not supported"

    if fusion_num > 0:
        scheduler = LR_SCHEDULER_MAP[scheduler]
        return scheduler(optimizer, B=fusion_num, last_epoch=last_epoch, **kwargs)
    return scheduler(optimizer, last_epoch=last_epoch, **kwargs)

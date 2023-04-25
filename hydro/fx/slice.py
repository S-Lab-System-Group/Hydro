from typing import Type, List, Dict, Any, Tuple, Iterable
import copy

import torch
import torch.fx as fx
import torch.nn as nn

import hydro.fuse_ops as fnn
from hydro.fx.utils import get_parent_name, parse_params_repr, matches_module_pattern


_SUPPORTED_SLICED_FUSION_MODULE_LIST = [
    fnn.Conv1d,
    fnn.Conv2d,
    fnn.ConvTranspose2d,
    fnn.Linear,
    fnn.BatchNorm1d,
    fnn.BatchNorm2d,
    fnn.LayerNorm,
    fnn.Embedding,
    fnn.TransformerEncoderLayer,
    fnn.AdaptiveAvgPool2d,
    fnn.MaxPool2d,
]


def _slice_fusion_module(module: nn.Module, keep_list: List):
    """
    Input:
        module: nn.Module to be scaled.
        keep_list: indexes of trials to retain.
    """
    repr = module.extra_repr()
    args, kwargs = parse_params_repr(repr)
    module.keep_partial_parameters(keep_list)
    return module


def slice_model(model: fx.GraphModule, keep_list: List, inplace=False) -> torch.nn.Module:
    """
    Input:
        model: traced Pytorch model.
        keep_list: indexes of trials to retain.
        inplace: whether to modify the original model.
    """
    if not inplace:
        model = copy.deepcopy(model)

    assert isinstance(model, fx.GraphModule), "Model need to be traced first."
    assert len(keep_list) > 0, "keep_list should not be empty."

    new_B = len(keep_list)
    modules = dict(model.named_modules())

    for _, node in enumerate(model.graph.nodes):
        if matches_module_pattern(_SUPPORTED_SLICED_FUSION_MODULE_LIST, node, modules):
            ori_module = modules[node.target]
            sliced_module = _slice_fusion_module(ori_module, keep_list)

            parent_name, name = get_parent_name(node.target)
            modules[node.target] = sliced_module
            setattr(modules[parent_name], name, sliced_module)

    model.delete_all_unused_submodules()
    model.recompile()
    return model

from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple, Iterable

import copy
import logging
import torch
import torch.nn as nn
import torch.fx as fx

from hydro.fx.utils import get_parent_name, parse_params_repr, matches_module_pattern
import hydro.fuse_ops as fnn

# nn.modules.conv._ConvNd
_SUPPORTED_SCALABLE_MODULE_LIST = [
    nn.Conv1d,
    nn.Conv2d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.Linear,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.LayerNorm,
    nn.Embedding,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
    nn.TransformerEncoder,
]

_SUPPORTED_SCALABLE_FUSION_MODULE_LIST = [
    fnn.Conv1d,
    fnn.Conv2d,
    fnn.ConvTranspose2d,
    fnn.Linear,
    fnn.BatchNorm1d,
    fnn.BatchNorm2d,
    fnn.LayerNorm,
    fnn.Embedding,
    fnn.TransformerEncoderLayer,
]


# def replace_output_layer(node, modules):
#     assert isinstance(modules[node.target], nn.Linear), "Currently, only support `nn.Linear` as the output layer."
#     repr = modules[node.target].extra_repr()
#     args, kwargs = parse_params_repr(repr)
#     parent_name, name = get_parent_name(node.target)
#     modules[node.target] = functools.partial(MuReadout, readout_zero_init=True)(*args, **kwargs)
#     setattr(modules[parent_name], name, functools.partial(MuReadout, readout_zero_init=True)(*args, **kwargs))

logger = logging.getLogger(__name__)


class Multiply(nn.Module):
    r"""A placeholder operator for weight scaling.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    """

    def __init__(self, multiplier: Union[int, float] = 1) -> None:
        super(Multiply, self).__init__()
        self.multiplier = multiplier

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.multiplier * input

    def extra_repr(self) -> str:
        return "multiplier={}".format(self.multiplier)


def _valid_check(scaled: Union[Dict, List], module: nn.Module, min_limit: int = 2):
    """
    Check if the scaling ratio is valid. Avoid too small model.

    Input:
        scaled: changed arguments to be applied.
        min_limit: the minimum module width.
    """
    if isinstance(scaled, dict):
        for key, value in scaled.items():
            if value is int:
                if value < min_limit:
                    raise ValueError(f"Module width is too small. Original module {module.__repr__} -> Scaled: {scaled}.")
    elif isinstance(scaled, list):
        for value in scaled:
            if value < min_limit:
                raise ValueError(f"Module width is too small. Original module {module.__repr__} -> Scaled: {scaled}.")
    else:
        raise NotImplementedError("Currently, only support `dict` or `list`.")


def _scale_input_module(module: nn.Module, scaling_ratio: Union[int, float] = 2):
    """
    Input:
        module: nn.Module to be scaled.
        scaling_ratio: the ratio to shrink the target model.
    """
    repr = module.extra_repr()
    args, kwargs = parse_params_repr(repr)
    if isinstance(module, nn.Linear):
        scaled_kargs = kwargs.copy()
        scaled_kargs["out_features"] = int(scaled_kargs["out_features"] // scaling_ratio)
        _valid_check(scaled_kargs, module)
        return nn.Linear(*args, **scaled_kargs)
    elif isinstance(module, nn.Conv2d):
        assert len(args) == 2
        scaled_args = [args[0], int(args[1] // scaling_ratio)]
        _valid_check(scaled_args, module)
        return nn.Conv2d(*scaled_args, **kwargs)
    elif isinstance(module, nn.Embedding):
        assert len(args) == 2, "Currently, only support `nn.Embedding` in Transformer."
        scaled_args = [args[0], int(args[1] // scaling_ratio)]
        _valid_check(scaled_args, module)
        return nn.Embedding(*scaled_args, **kwargs)


def _scale_output_module(module: nn.Module, scaling_ratio: Union[int, float] = 2):
    """
    Input:
        module: nn.Module to be scaled.
        scaling_ratio: the ratio to shrink the target model
    """
    repr = module.extra_repr()
    args, kwargs = parse_params_repr(repr)
    if isinstance(module, nn.Linear):
        scaled_kargs = kwargs.copy()
        scaled_kargs["in_features"] = int(scaled_kargs["in_features"] // scaling_ratio)
        _valid_check(scaled_kargs, module)
        # return nn.Sequential(Multiply(scaling_ratio), nn.Linear(*args, **scaled_kargs))
        return nn.Linear(*args, **scaled_kargs)
    else:
        raise NotImplementedError("Currently, only support `nn.Linear` as the output layer.")


def _scale_module(module: nn.Module, scaling_ratio: Union[int, float] = 2):
    """
    Input:
        module: nn.Module to be scaled.
        scaling_ratio: the ratio to shrink the target model
    """
    repr = module.extra_repr()
    args, kwargs = parse_params_repr(repr)
    if isinstance(module, nn.Linear):
        scaled_kargs = {k: int(v // scaling_ratio) for k, v in kwargs.items() if k in ["in_features", "out_features"]}
        scaled_kargs.update({k: v for k, v in kwargs.items() if k not in ["in_features", "out_features"]})
        _valid_check(scaled_kargs, module)
        return nn.Linear(*args, **scaled_kargs)
    elif isinstance(module, nn.Conv2d):
        scaled_args = [int(x // scaling_ratio) for x in args]
        if "groups" in kwargs:
            scaled_kargs = kwargs | {"groups": int(kwargs["groups"] // scaling_ratio)}
        else:
            scaled_kargs = kwargs
        _valid_check(scaled_args, module)
        return nn.Conv2d(*scaled_args, **scaled_kargs)
    elif isinstance(module, nn.BatchNorm2d):
        scaled_args = [int(x // scaling_ratio) for x in args]
        _valid_check(scaled_args, module)
        return nn.BatchNorm2d(*scaled_args, **kwargs)
    elif isinstance(module, nn.LayerNorm):
        assert len(args[0]) == 1, "Currently, only support `nn.LayerNorm` in Transformer."
        scaled_args = (int(args[0][0] // scaling_ratio),)
        # _valid_check(scaled_args, module)
        return nn.LayerNorm(*scaled_args, **kwargs)
    elif isinstance(module, nn.Embedding):
        assert len(args) == 2, "Currently, only support `nn.Embedding` in Transformer."
        scaled_args = [args[0], int(args[1] // scaling_ratio)]
        # _valid_check(scaled_args, module)
        return nn.Embedding(*scaled_args, **kwargs)
    elif isinstance(module, nn.TransformerEncoder):
        args, kwargs = parse_params_repr(module.extra_repr())
        assert len(kwargs) == 0

        layer_args, layer_kwargs = parse_params_repr(module.layers[0].extra_repr())
        scaled_layer_args = [int(layer_args[0] // scaling_ratio), layer_args[1]]
        scaled_layer_kwargs = layer_kwargs | {"dim_feedforward": int(layer_kwargs["dim_feedforward"] // scaling_ratio)}
        scaled_layer = nn.TransformerEncoderLayer(*scaled_layer_args, **scaled_layer_kwargs)
        return nn.TransformerEncoder(scaled_layer, args[0])


def _scale_input_fusion_module(module: nn.Module, scaling_ratio: Union[int, float] = 2):
    """
    Input:
        module: nn.Module to be scaled.
        scaling_ratio: the ratio to shrink the target model.
    """
    repr = module.extra_repr()
    args, kwargs = parse_params_repr(repr)
    if isinstance(module, fnn.Linear):
        scaled_kargs = kwargs.copy()
        scaled_kargs["out_features"] = int(scaled_kargs["out_features"] // scaling_ratio)
        _valid_check(scaled_kargs, module)
        return fnn.Linear(*args, **scaled_kargs)
    elif isinstance(module, fnn.Conv2d):
        assert len(args) == 2
        scaled_args = [args[0], int(args[1] // scaling_ratio)]
        _valid_check(scaled_args, module)
        return fnn.Conv2d(*scaled_args, **kwargs)
    elif isinstance(module, fnn.Embedding):
        assert len(args) == 2, "Currently, only support `nn.Embedding` in Transformer."
        scaled_args = [args[0], int(args[1] // scaling_ratio)]
        _valid_check(scaled_args, module)
        return fnn.Embedding(*scaled_args, **kwargs)


def _scale_output_fusion_module(module: nn.Module, scaling_ratio: Union[int, float] = 2):
    """
    Input:
        module: nn.Module to be scaled.
        scaling_ratio: the ratio to shrink the target model
    """
    repr = module.extra_repr()
    args, kwargs = parse_params_repr(repr)
    if isinstance(module, fnn.Linear):
        scaled_kargs = kwargs.copy()
        scaled_kargs["in_features"] = int(scaled_kargs["in_features"] // scaling_ratio)
        _valid_check(scaled_kargs, module)
        # return nn.Sequential(Multiply(scaling_ratio), nn.Linear(*args, **scaled_kargs))
        return fnn.Linear(*args, **scaled_kargs)
    else:
        raise NotImplementedError("Currently, only support `nn.Linear` as the output layer.")


def _scale_fusion_module(module: nn.Module, scaling_ratio: Union[int, float] = 2):
    """
    Input:
        module: nn.Module to be scaled.
        scaling_ratio: the ratio to shrink the target model
    """
    repr = module.extra_repr()
    args, kwargs = parse_params_repr(repr)
    if isinstance(module, fnn.Linear):
        scaled_kargs = {k: int(v // scaling_ratio) for k, v in kwargs.items() if k in ["in_features", "out_features"]}
        scaled_kargs.update({k: v for k, v in kwargs.items() if k not in ["in_features", "out_features"]})
        _valid_check(scaled_kargs, module)
        return fnn.Linear(*args, **scaled_kargs)
    elif isinstance(module, fnn.Conv2d):
        scaled_args = [int(x // scaling_ratio) for x in args]
        if "groups" in kwargs:
            scaled_kargs = kwargs | {"groups": int(kwargs["groups"] // scaling_ratio)}
        else:
            scaled_kargs = kwargs
        _valid_check(scaled_args, module)
        return fnn.Conv2d(*scaled_args, **scaled_kargs)
    elif isinstance(module, fnn.BatchNorm2d):
        scaled_args = [int(x // scaling_ratio) for x in args]
        _valid_check(scaled_args, module)
        return fnn.BatchNorm2d(*scaled_args, **kwargs)
    elif isinstance(module, fnn.LayerNorm):
        assert len(args[0]) == 1, "Currently, only support `nn.LayerNorm` in Transformer."
        scaled_args = (int(args[0][0] // scaling_ratio),)
        # _valid_check(scaled_args, module)
        return fnn.LayerNorm(*scaled_args, **kwargs)
    elif isinstance(module, fnn.Embedding):
        assert len(args) == 3, "Currently, only support `nn.Embedding` in Transformer."
        scaled_args = [args[0], int(args[1] // scaling_ratio)]
        # _valid_check(scaled_args, module)
        return fnn.Embedding(*scaled_args, **kwargs)


############################################################################################################


def scale_model(model: fx.GraphModule, scaling_ratio: Union[int, float] = 2, inplace=False):
    """
    Input:
        model: traced Pytorch model.
        scaling_ratio: the ratio to shrink the target model
    """
    assert scaling_ratio > 0, f"Invalid value: `scaling_ratio`={scaling_ratio}."
    assert isinstance(model, fx.GraphModule), "Model need to be traced first."
    if not inplace:
        model = copy.deepcopy(model)
    graph_len = len(model.graph.nodes)
    modules = dict(model.named_modules())
    for idx, node in enumerate(model.graph.nodes):
        if matches_module_pattern(_SUPPORTED_SCALABLE_MODULE_LIST, node, modules):
            ori_module = modules[node.target]

            if idx == 1:  # Dealing with input layer
                scaled_module = _scale_input_module(ori_module, scaling_ratio)
            elif idx == graph_len - 2:  # Dealing with output layer
                scaled_module = _scale_output_module(ori_module, scaling_ratio)

                model.add_module("readout", Multiply(scaling_ratio))
                with model.graph.inserting_before(node):
                    new_node = model.graph.call_module("readout", (node.prev,))

                parent_name, name = get_parent_name(node.target)
                modules[node.target] = scaled_module
                setattr(modules[parent_name], name, scaled_module)
                node.update_arg(0, node.prev)
                continue
            else:
                scaled_module = _scale_module(ori_module, scaling_ratio)

            parent_name, name = get_parent_name(node.target)
            modules[node.target] = scaled_module
            setattr(modules[parent_name], name, scaled_module)
    model.delete_all_unused_submodules()
    model.recompile()
    return model


def scale_fused_model(model: fx.GraphModule, scaling_ratio: Union[int, float] = 2, inplace=False):
    """
    Input:
        model: traced Pytorch model.
        scaling_ratio: the ratio to shrink the target model
        inplace: whether to modify the original model.
    """
    assert scaling_ratio > 0, f"Invalid value: `scaling_ratio`={scaling_ratio}."
    assert isinstance(model, fx.GraphModule), "Model need to be traced first."
    if not inplace:
        model = copy.deepcopy(model)
    graph_len = len(model.graph.nodes)
    modules = dict(model.named_modules())
    for idx, node in enumerate(model.graph.nodes):
        if matches_module_pattern(_SUPPORTED_SCALABLE_FUSION_MODULE_LIST, node, modules):
            ori_module = modules[node.target]

            if idx == 1:  # Dealing with input layer
                scaled_module = _scale_input_fusion_module(ori_module, scaling_ratio)
            elif idx == graph_len - 2:  # Dealing with output layer
                scaled_module = _scale_output_fusion_module(ori_module, scaling_ratio)

                model.add_module("readout", Multiply(scaling_ratio))
                with model.graph.inserting_before(node):
                    new_node = model.graph.call_module("readout", (node.prev,))

                parent_name, name = get_parent_name(node.target)
                modules[node.target] = scaled_module
                setattr(modules[parent_name], name, scaled_module)
                node.update_arg(0, node.prev)
                continue
            else:
                scaled_module = _scale_fusion_module(ori_module, scaling_ratio)

            parent_name, name = get_parent_name(node.target)
            modules[node.target] = scaled_module
            setattr(modules[parent_name], name, scaled_module)
    model.delete_all_unused_submodules()
    model.recompile()
    return model

from typing import Type, List, Dict, Any, Tuple, Iterable
import copy
import functools

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

import hydro.fuse_ops
from hydro.fuse_ops import OPS_MAP, FUNCTION_MAP
from hydro.fx.utils import get_parent_name, parse_params_repr, matches_module_pattern


_SUPPORTED_FUSIBLE_MODULE_LIST = [
    nn.Conv1d,
    nn.Conv2d,
    nn.ConvTranspose2d,
    nn.Linear,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Dropout2d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.LayerNorm,
    nn.Embedding,
    nn.Flatten,
    nn.TransformerEncoder,
    # nn.TransformerEncoderLayer,
]

_SUPPORTED_FUSIBLE_FUNC_LIST = [
    F.adaptive_avg_pool2d,
    torch.flatten,
]


# def convert_ops(B, *torch_op_classes):
#     return (convert_op(op_class, B=B) for op_class in torch_op_classes)


def matches_function_pattern(pattern: Iterable[Any], node: fx.Node):
    if len(node.args) == 0:
        return False
    if node.op != "call_function":
        return False
    for ops in pattern:
        if node.target is ops:
            return True
    return False


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert isinstance(node.target, str)
    parent_name, name = get_parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def flatten_replacement(x, dim_start):
    x = x.transpose(0, 1)
    return torch.flatten(x, dim_start + 1)


def flatten_rewrite(node: fx.Node, graph: fx.Graph):
    with graph.inserting_after(node):
        new_node = graph.call_function(flatten_replacement, node.args, node.kwargs)
        node.replace_all_uses_with(new_node)
    graph.erase_node((node))


def functional_rewrite(node: fx.Node, graph: fx.Graph, new_module: torch.nn.Module):
    with graph.inserting_after(node):
        new_node = graph.call_module(hydro.fuse_ops.AdaptiveAvgPool2d)
        node.replace_all_uses_with(new_node)
    graph.erase_node((node))


def _get_fused_op(module: nn.Module, B: int = 1):
    repr = module.extra_repr()
    args, kwargs = parse_params_repr(repr)
    fused_op = functools.partial(OPS_MAP[type(module)], B=B)(*args, **kwargs)
    return fused_op


def _get_fused_op_for_func(node: fx.Node, B: int = 1):
    args, kwargs = node.args[1:], node.kwargs

    fused_op = functools.partial(FUNCTION_MAP[str(node)], B=B)(*args, **kwargs)
    return fused_op


def fuse_model(model: fx.GraphModule, B: int = 1, inplace=False) -> torch.nn.Module:
    """
    Input:
        model: traced Pytorch model.
        B: number of grouped models.
        inplace: whether to modify the original model.
    """
    if not inplace:
        model = copy.deepcopy(model)

    assert B > 0, "Invalid value: `B`."
    assert isinstance(model, fx.GraphModule), "Model need to be traced first."
    modules = dict(model.named_modules())

    for _, node in enumerate(model.graph.nodes):
        if matches_module_pattern(_SUPPORTED_FUSIBLE_MODULE_LIST, node, modules):
            ori_module = modules[node.target]

            if isinstance(ori_module, nn.TransformerEncoder):
                args, kwargs = parse_params_repr(ori_module.extra_repr())
                assert len(kwargs) == 0, "Currently, only support default LayerNorm"

                layer_args, layer_kwargs = parse_params_repr(ori_module.layers[0].extra_repr())
                fused_encoder_layer = functools.partial(OPS_MAP[type(ori_module.layers[0])], B=B)(*layer_args, **layer_kwargs)
                fused_op = nn.TransformerEncoder(fused_encoder_layer, args[0])
                # encoder_norm = nn.LayerNorm(ori_module.layers[0].d_model)
                # fused_op = nn.TransformerEncoder(fused_encoder_layer, args[0], norm=encoder_norm)
            else:
                fused_op = _get_fused_op(ori_module, B=B)
            replace_node_module(node, modules, fused_op)

        if matches_function_pattern(_SUPPORTED_FUSIBLE_FUNC_LIST, node):
            if str(node) == "flatten":
                flatten_rewrite(node, model.graph)
            elif str(node) == "adaptive_avg_pool2d":
                with model.graph.inserting_before(node):
                    new_module_str = str(node._prev).split("_")[0] + ".adaptive_avg_pool2d"
                    output_size = node.args[1:]
                    if isinstance(output_size, tuple) and len(output_size) == 1:
                        output_size = output_size[0]

                    model.add_submodule(new_module_str, hydro.fuse_ops.AdaptiveAvgPool2d(output_size, B=B))
                    new_node = model.graph.call_module(new_module_str, node.args[:1])
                    node.replace_all_uses_with(new_node)
                model.graph.erase_node((node))

    model.delete_all_unused_submodules()
    model.recompile()
    return model


# print(node.target)
# print(node._prev)
# print(node._next)
# print(modules[node.target.])
# fused_op = _get_fused_op_for_func(node, B=B)
# FUNCTION_MAP[str(node)]

# parent_name, name = get_parent_name(new_node.target)
# new_module = hydro.fuse_ops.AdaptiveAvgPool2d(node.args[1], B=B)
# modules[new_node.target] = new_module
# setattr(modules[parent_name], name, new_module)

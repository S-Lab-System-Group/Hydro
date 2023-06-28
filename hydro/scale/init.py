from typing import Union

import logging
import torch
import torch.nn as nn
import torch.fx as fx

import hydro.fuse_ops as fnn
from hydro.scale import matches_module_pattern

logger = logging.getLogger(__name__)

DEFAULT_KAIMING_A = 1

_SUPPORTED_INIT_MODULE_LIST = [
    nn.Conv1d,
    nn.Conv2d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.Linear,
    nn.Embedding,
    nn.LayerNorm,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
    fnn.Conv1d,
    fnn.Conv2d,
    fnn.ConvTranspose2d,
    fnn.Linear,
    fnn.Embedding,
    fnn.LayerNorm,
    fnn.TransformerEncoderLayer,
]


def _reinitialize_Linear(layer: nn.Module):
    """
    Linear layer initialization with Kaiming Normalization.
    """
    assert isinstance(layer, nn.Linear) or isinstance(layer, fnn.Linear)

    if isinstance(layer, fnn.Linear):
        # Equivalent to the original model without fusion
        for b in range(layer.B):
            # nn.init.constant_(layer.weight[b], 0.0001)  # For testing purpose
            nn.init.kaiming_normal_(layer.weight[b], a=DEFAULT_KAIMING_A, mode="fan_out")
    else:
        nn.init.kaiming_normal_(layer.weight, a=DEFAULT_KAIMING_A)


def _reinitialize_Conv(layer: nn.Module):
    """
    Conv layer initialization with Kaiming Normalization.
    """
    assert isinstance(layer, nn.modules.conv._ConvNd) or isinstance(layer, fnn.conv._ConvNd)

    if isinstance(layer, fnn.conv._ConvNd):
        # Equivalent to the original model without fusion
        for b in range(layer.B):
            nn.init.kaiming_normal_(layer.weight[b], a=DEFAULT_KAIMING_A)
    else:
        nn.init.kaiming_normal_(layer.weight, a=DEFAULT_KAIMING_A)


def _reinitialize_Embedding(layer: nn.Module):
    """
    Embedding layer initialization with Normalization.
    """
    assert isinstance(layer, nn.Embedding) or isinstance(layer, fnn.Embedding)

    # nn.init.normal_(layer.weight)
    nn.init.kaiming_normal_(layer.weight, a=DEFAULT_KAIMING_A)
    if layer.padding_idx is not None:
        with torch.no_grad():
            layer.weight[layer.padding_idx].fill_(0)


def _reinitialize_LayerNorm(layer: nn.Module):
    """
    LayerNorm layer initialization with Normalization.
    """
    assert isinstance(layer, nn.LayerNorm) or isinstance(layer, fnn.LayerNorm)

    # nn.init.normal_(layer.weight)
    nn.init.kaiming_normal_(layer.weight, a=DEFAULT_KAIMING_A)
    nn.init.zeros_(layer.bias)


#####################################################################


def _reinitialize_output_module(layer: nn.Module, std: float = 1.0):
    """
    Zero initialization for the output layer.
    """
    assert isinstance(layer, nn.Linear) or isinstance(layer, fnn.Linear)

    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _reinitialize_module(layer: nn.Module, std: float = 1.0):
    """
    MuP Examples:
        - BERT: use modified `normal_`, need to manually call `layer.apply(layer._init_weights)` after calling `set_base_shape(model, base)`

        - GPT2: use modified `normal_`, More depth-related consideration

        - MLP: kaiming_normal_

        - ResNet: kaiming_normal_

        - Transformer: normal_, std=(1/d_model)**0.5 in TransformerEncoderLayer
                                std=(1/embed_dim)**0.5  in MultiheadAttention
    """

    layer_name = layer.__class__.__name__.lower()

    if "linear" in layer_name:
        _reinitialize_Linear(layer)
    elif "conv" in layer_name:
        _reinitialize_Conv(layer)
    elif "embedding" in layer_name:
        _reinitialize_Embedding(layer)
    elif "layernorm" in layer_name:
        _reinitialize_LayerNorm(layer)
    else:
        raise NotImplementedError(f"Unsupported layer type: {layer_name}")
        nn.init.kaiming_normal_(layer.weight, a=1, mode="fan_in")

    # nn.init.normal_(layer.weight, std=std)
    if layer.bias is not None:
        # nn.init.normal_(layer.bias, std=std)
        nn.init.zeros_(layer.bias)


# def reinitialize_model(model: Union[fx.GraphModule, nn.Module], scaling_ratio: Union[int, float] = 2):
#     """
#     Input:
#         model: traced Pytorch model.
#         scaling_ratio: the ratio to shrink the target model
#     """
#     for idx, layer in enumerate(model.modules()):
#         print(idx, layer)
#         if hasattr(layer, "weight"):
#             # print(idx, layer)
#             # Back to the default initialization
#             # layer.reset_parameters()
#             _reinitialize_module(layer, std=scaling_ratio ** 0.5)

#     # Assign the output layer data to be zero
#     _reinitialize_output_module(layer, std=scaling_ratio)

#     return model


def reinitialize_model(model: fx.GraphModule, scaling_ratio: Union[int, float] = 2):
    """
    Input:
        model: traced Pytorch model.
        scaling_ratio: the ratio to shrink the target model
    """
    assert scaling_ratio > 0, "Invalid value: `scaling_ratio`."
    assert isinstance(model, fx.GraphModule), "Model need to be traced first."

    graph_len = len(model.graph.nodes)
    modules = dict(model.named_modules())

    for idx, node in enumerate(model.graph.nodes):
        if matches_module_pattern(_SUPPORTED_INIT_MODULE_LIST, node, modules):
            ori_module = modules[node.target]
            if idx == graph_len - 2:  # Dealing with output layer
                _reinitialize_output_module(ori_module, std=scaling_ratio)
            else:
                _reinitialize_module(ori_module, std=scaling_ratio**0.5)

    model.delete_all_unused_submodules()
    model.recompile()
    return model

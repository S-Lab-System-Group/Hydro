# https://github.com/huggingface/transformers/blob/main/src/transformers/utils/fx.py

import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx.proxy import ParameterProxy

import transformers
from transformers import PreTrainedModel
from transformers.utils import fx as tfx


logger = logging.getLogger(__name__)


class HydroTracer(Tracer):
    """
    Tracer that is able to symbolically trace models from the library. To do that, it uses the HFProxy instead of the
    regular PyTorch torch.fx.Proxy.
    """

    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes: bool = True
    allow_insert_stateless_mods: bool = True
    _TORCH_METHODS_TO_PATCH = ["arange", "zeros", "ones", "full", "full_like", "eye", "empty", "tensor"]

    def __init__(self, autowrap_modules=(math,), autowrap_functions=()):

        super().__init__(autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions)


def get_concrete_args(model: nn.Module, input_names: List[str]):
    sig = inspect.signature(model.forward)

    if not (set(input_names) <= set(sig.parameters.keys())):
        formatted_input_names = input_names[0] if len(input_names) == 1 else ", ".join(input_names)
        formatted_allowed_input_names = ", ".join(sig.parameters.keys())
        raise ValueError(
            f"The model does not have input(s) named: {formatted_input_names}, expected a subset of the following:"
            f" {formatted_allowed_input_names}"
        )

    return {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}


def symbolic_trace(model: Union[torch.nn.Module, Callable[..., Any]], input_names: Optional[List[str]] = None) -> GraphModule:

    """
    Performs symbolic tracing on the model.

    Args:
        model ([`PretrainedModel`]):
            The model to trace.
        input_names (`List[str]`, *optional*):
            The names of the inputs of the traced model. If unset, model.dummy_inputs.keys() are used instead.

    Returns:
        `torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example:

        ```python
        from transformers.utils.fx import symbolic_trace

        traced_model = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        ```
    """

    # Huggingface transformer models -> Directly use transformers.utils.fx
    if isinstance(model, PreTrainedModel):
        logger.info("Tracing Huggingface Transformer model")
        traced = tfx.symbolic_trace(model, input_names)
        return traced

    if isinstance(model, torch.nn.Module):
        logger.info("Tracing General Model")
        tracer = Tracer()
        # tracer = HydroTracer()
        traced_graph = tracer.trace(model, concrete_args=None)
        traced = torch.fx.GraphModule(model, traced_graph)
        name = model.__class__.__name__
        return GraphModule(tracer.root, traced_graph, name)

    # TODO
    raise NotImplementedError

    # traced.config = model.config
    # traced.class_for_deserialization = model.__class__
    # traced.device = model.device

    # return traced

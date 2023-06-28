from typing import Any, Callable, List, Optional, Union

import logging
from importlib.util import find_spec
from packaging.version import Version

import torch
import torch.fx
from torch.fx import GraphModule, Tracer


if Version(torch.__version__) < Version("2.0"):
    torch_dynamo = False
else:
    torch_dynamo = True
    from torch._dynamo.eval_frame import optimize

transformers_available = find_spec("transformers")
if transformers_available is not None:
    import transformers
    from transformers import PreTrainedModel
    from transformers.utils import fx as tfx


logger = logging.getLogger(__name__)


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
        traced_model = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        ```
    """

    # TODO: Add support for torch dynamo
    # if torch_dynamo:
    #     # Trace with torch dynamo (graph capture only)
    #     logger.info("Tracing model with torch dynamo")
    #     traced = optimize(backend=lambda x: x)(model)
    #     return traced

    if transformers_available is not None:
        # Huggingface transformer models -> Directly use transformers.utils.fx
        if isinstance(model, PreTrainedModel):
            logger.info("Tracing Huggingface Transformer model")
            traced = tfx.symbolic_trace(model, input_names)
            return traced

    if isinstance(model, torch.nn.Module):
        logger.info("Tracing general model with torch.fx")
        tracer = Tracer()
        traced_graph = tracer.trace(model, concrete_args=None)
        traced = torch.fx.GraphModule(model, traced_graph)
        name = model.__class__.__name__
        return GraphModule(tracer.root, traced_graph, name)

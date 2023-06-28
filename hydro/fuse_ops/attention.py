from typing import Optional, Tuple
import math
import functools

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import torch.nn.functional as F

import torch
from torch import nn
import torch.utils.checkpoint
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer


from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, load_tf_weights_in_gpt2, GPT2PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class GPT2AttentionHydro(GPT2Attention):
    """Transformers 4.30"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_mult = self.embed_dim / self.num_heads

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # if self.scale_attn_weights:
        #     attn_weights = attn_weights / torch.full(
        #         [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        #     )

        if self.scale_attn_weights:
            attn_weights = (
                self.attn_mult
                * attn_weights
                / torch.full([], value.size(-1), dtype=attn_weights.dtype, device=attn_weights.device)
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class GPT2PreTrainedModelHydro(GPT2PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MuReadout) and readout_zero_init:
            module.weight.data.zero_()
        elif isinstance(module, (nn.Linear, Conv1D)):
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if hasattr(module.weight, "infshape"):
                normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        c_proj_std = self.config.initializer_range / math.sqrt(2 * self.config.n_layer)
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # p.data.normal_(mean=0.0, std=c_proj_std)
                if hasattr(p, "infshape"):
                    normal_(p, mean=0.0, std=c_proj_std)
                else:
                    p.data.normal_(mean=0.0, std=c_proj_std)

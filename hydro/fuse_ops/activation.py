from typing import Optional, Tuple
import math
import functools

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_


class MultiheadAttention(Module):
    r"""Pytorch 2.0
    Allows the model to jointly attend to information
    from different representation subspaces as described in the paper: `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    """
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        B=1,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # NOTE: Modified from here
        assert self._qkv_same_embed_dim, "Currently only supports common QKV same embed_dim"
        assert B > 0
        self.B = B
        from .linear import Linear

        # self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        # self.register_parameter("q_proj_weight", None)
        # self.register_parameter("k_proj_weight", None)
        # self.register_parameter("v_proj_weight", None)

        Linear = functools.partial(Linear, B=B)
        self.scaling = float(self.head_dim) ** -0.5
        self.bias = bias
        self.linear_q = Linear(embed_dim, embed_dim, bias)
        self.linear_k = Linear(embed_dim, embed_dim, bias)
        self.linear_v = Linear(embed_dim, embed_dim, bias)
        self.linear_o = Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _get_xavier_uniform_data(self):
        data = torch.zeros((3 * self.embed_dim, self.embed_dim))
        torch.nn.init.xavier_uniform_(data)
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(data)
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        return nn.init._no_grad_uniform_(data, -a, a).reshape((3, self.embed_dim, self.embed_dim))

    def _reset_parameters(self):
        for b in range(self.B):
            tmp_weight = self._get_xavier_uniform_data()
            self.linear_q.weight.data[b] = tmp_weight[0]
            self.linear_k.weight.data[b] = tmp_weight[1]
            self.linear_v.weight.data[b] = tmp_weight[2]

        if self.bias:
            torch.nn.init.constant_(self.linear_q.bias, 0.0)
            torch.nn.init.constant_(self.linear_k.bias, 0.0)
            torch.nn.init.constant_(self.linear_v.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and float masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
                Default: ``False``.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        N = query.shape[0]

        if self.B > 0:
            N = query.size(1)
            query = query.contiguous().view(query.shape[0], -1, query.shape[-1])
            key = key.contiguous().view(key.shape[0], -1, key.shape[-1])
            value = value.contiguous().view(value.shape[0], -1, value.shape[-1])

        query = self.linear_q(query.contiguous())
        key = self.linear_k(key.contiguous())
        value = self.linear_v(value.contiguous())

        if self.B > 0:
            query = query.view(self.B * N, -1, query.shape[-1])
            key = key.view(self.B * N, -1, key.shape[-1])
            value = value.view(self.B * N, -1, value.shape[-1])

        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        query *= self.scaling

        query = query.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        query = query.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)

        key = key.contiguous().view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.contiguous().view(bsz * self.num_heads, src_len, self.head_dim)

        value = value.contiguous().view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.contiguous().view(bsz * self.num_heads, src_len, self.head_dim)

        o_weights = torch.bmm(query, key.transpose(1, 2))

        if attn_mask is not None:
            if attn_mask.dim() > 4 + min(self.B, 1) or attn_mask.dim() < 2:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

            while attn_mask.dim() < 4 + min(self.B, 1):
                attn_mask = attn_mask.unsqueeze(0)
            if not (
                (self.B == 0 or (attn_mask.size(0) in [1, self.B]))
                and (
                    attn_mask.size(-4) in [1, N]
                    and attn_mask.size(-3) in [1, self.num_heads]
                    and attn_mask.size(-2) == query.size(-2)
                    and attn_mask.size(-1) == key.size(-2)
                )
            ):
                raise RuntimeError("The size of the attn_mask is not correct.")
            # attn_mask's dim is 5 now.

            old_shape = o_weights.shape
            if self.B > 0:
                o_weights = o_weights.view((self.B, N, -1, tgt_len, src_len))
            else:
                o_weights = o_weights.view((N, -1, tgt_len, src_len))

            if attn_mask.dtype == torch.bool:
                o_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                o_weights += attn_mask
            o_weights = o_weights.view(old_shape)

        o_weights = F.softmax(o_weights, dim=-1)
        o_weights = F.dropout(o_weights, p=self.dropout, training=self.training)
        o = torch.bmm(o_weights, value)
        if self.B > 0:
            o = o.contiguous().view(self.B, N, self.num_heads, tgt_len, self.head_dim)
            o = o.transpose(2, 3).contiguous().view(self.B, N, tgt_len, embed_dim)
        else:
            o = o.contiguous().view(bsz, self.num_heads, tgt_len, self.head_dim)
            o = o.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        o = self.linear_o(o)

        if need_weights:
            # average attention weights over heads
            o_weights = o_weights.view(bsz, self.num_heads, tgt_len, src_len)
            o_weights = o_weights.sum(dim=1) / self.num_heads
            if self.B > 0:
                o_weights = o_weights.view(self.B, N, tgt_len, src_len)
            return o, o_weights
        else:
            return o, None

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        res = torch.tril(torch.ones(seq_len, seq_len))
        return res.view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def extra_repr(self):
        return "in_features={}, head_num={}, bias={}, B={}".format(
            self.embed_dim,
            self.num_heads,
            self.bias,
            self.B,
        )

    def snatch_parameters(self, other, b=0):
        assert isinstance(other, nn.MultiheadAttention)
        assert other._qkv_same_embed_dim
        assert other.bias_k is None and other.bias_v is None
        assert self.embed_dim == other.embed_dim
        assert self.num_heads == other.num_heads

        tmp_weight = other.in_proj_weight.reshape(3, self.embed_dim, self.embed_dim)
        if b > 0:
            self.linear_q.weight.data[b - 1] = tmp_weight[0].transpose(0, 1).view(self.linear_q.weight.data[b - 1].shape)
            self.linear_k.weight.data[b - 1] = tmp_weight[1].transpose(0, 1).view(self.linear_k.weight.data[b - 1].shape)
            self.linear_v.weight.data[b - 1] = tmp_weight[2].transpose(0, 1).view(self.linear_v.weight.data[b - 1].shape)
            self.linear_o.weight.data[b - 1] = other.out_proj.weight.data.transpose(0, 1).view(
                self.linear_o.weight.data[b - 1].shape
            )
            self.linear_o.bias.data[b - 1] = other.out_proj.bias.data.view(self.linear_o.bias.data[b - 1].shape)
        else:
            self.linear_q.weight.data = tmp_weight[0].view(self.linear_q.weight.data.shape)
            self.linear_k.weight.data = tmp_weight[1].view(self.linear_k.weight.data.shape)
            self.linear_v.weight.data = tmp_weight[2].view(self.linear_v.weight.data.shape)
            self.linear_o.weight.data = other.out_proj.weight.data.view(self.linear_o.weight.data.shape)
            self.linear_o.bias.data = other.out_proj.bias.data.view(self.linear_o.bias.data.shape)

        if self.bias:
            tmp_bias = other.in_proj_bias.reshape(3, self.embed_dim)
            if b > 0:
                self.linear_q.bias.data[b - 1] = tmp_bias[0].view(self.linear_q.bias.data[b - 1].shape)
                self.linear_k.bias.data[b - 1] = tmp_bias[1].view(self.linear_k.bias.data[b - 1].shape)
                self.linear_v.bias.data[b - 1] = tmp_bias[2].view(self.linear_v.bias.data[b - 1].shape)
            else:
                self.linear_q.bias.data = tmp_bias[0].view(self.linear_q.bias.data.shape)
                self.linear_k.bias.data = tmp_bias[1].view(self.linear_k.bias.data.shape)
                self.linear_v.bias.data = tmp_bias[2].view(self.linear_v.bias.data.shape)

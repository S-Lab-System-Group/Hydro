import math
import torch
from torch import Tensor
from torch.nn import Module, Parameter, init

torch.nn.Linear


class Linear(Module):
    r"""Pytorch 2.0
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features", "B"]
    in_features: int
    out_features: int
    weight: Tensor
    B: int

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, B=1, tie_weight=False
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.B = B
        self.tie_weight = tie_weight
        if tie_weight:  # For Huggingface tie weights support
            self.weight = Parameter(torch.empty((B, out_features, in_features), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty((B, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty((B, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for b in range(self.B):
            init.kaiming_uniform_(self.weight[b], a=math.sqrt(5), mode="fan_out")
            if self.bias is not None:
                _, fan_out = init._calculate_fan_in_and_fan_out(self.weight[b])
                bound = 1 / math.sqrt(fan_out) if fan_out > 0 else 0
                init.uniform_(self.bias[b], -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        old_shape = list(input.shape)
        input = input.reshape(old_shape[0], -1, old_shape[-1])
        if self.bias is None:
            if self.tie_weight:
                res = torch.bmm(input, self.weight.data.transpose(1, 2))
            else:
                res = torch.bmm(input, self.weight)
        else:
            res = torch.baddbmm(self.bias, input, self.weight)
        old_shape[-1] = self.out_features
        return res.reshape(old_shape)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, B={}".format(
            self.in_features, self.out_features, self.bias is not None, self.B
        )

    def snatch_parameters(self, other, b):
        assert isinstance(other, torch.nn.Linear)
        assert 0 <= b < self.B
        self.weight.data[b] = other.weight.data.transpose(0, 1)
        if self.bias is not None:
            self.bias.data[b] = other.bias.data.unsqueeze(0)

    def keep_partial_parameters(self, keep_list):
        self.B = len(keep_list)
        self.weight.data = self.weight.data[keep_list, :, :]
        if self.bias is not None:
            self.bias.data = self.bias.data[keep_list, :, :]

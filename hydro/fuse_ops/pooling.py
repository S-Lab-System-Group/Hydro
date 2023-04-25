from typing import List, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.common_types import (
    _size_any_t,
    _size_1_t,
    _size_2_t,
    _size_any_opt_t,
    _size_2_opt_t,
)


class _MaxPoolNd(Module):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
        "B",
    ]
    return_indices: bool
    ceil_mode: bool
    B: int

    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        B: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.B = B

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding} "
            ", dilation={dilation}, ceil_mode={ceil_mode}, B={B}".format(**self.__dict__)
        )

    def keep_partial_parameters(self, keep_list):
        self.B = len(keep_list)


class MaxPool2d(_MaxPoolNd):
    r"""Pytorch 2.0
    Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: Tensor) -> Tensor:
        N, B, C, H, W = input.size()
        res = F.max_pool2d(
            input.view(N * B, C, H, W),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        if self.return_indices:
            y, indices = res
        else:
            y = res
        NxB, C, H, W = y.size()
        y = y.view(N, B, C, H, W)
        return (y, indices.view(N, B, C, H, W)) if self.return_indices else y


class _AdaptiveAvgPoolNd(Module):
    __constants__ = ["output_size", "B"]

    def __init__(self, output_size: _size_any_opt_t, B: int = 1) -> None:
        super().__init__()
        self.output_size = output_size
        self.B = B

    def extra_repr(self) -> str:
        return "output_size={}, B={}".format(self.output_size, self.B)

    def keep_partial_parameters(self, keep_list):
        self.B = len(keep_list)


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    r"""Pytorch 2.0
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1})` or :math:`(C, S_{0}, S_{1})`, where
          :math:`S=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

    """

    output_size: _size_2_opt_t

    def forward(self, input: Tensor) -> Tensor:
        N, B, C, H, W = input.size()
        output = F.adaptive_avg_pool2d(input.view(N * B, C, H, W), self.output_size)
        NxB, C, H, W = output.size()
        return output.view(N, B, C, H, W)

from torch import Tensor
from torch.nn.modules import Module
from torch.nn import functional as F


class _DropoutNd(Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False, B: int = 1) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.B = B

    def extra_repr(self) -> str:
        return "p={}, inplace={}, B={}".format(self.p, self.inplace, self.B)


class Dropout(_DropoutNd):
    r"""Pytorch 2.0
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)


class Dropout2d(_DropoutNd):
    r"""Pytorch 2.0
    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    .. warning ::
        Due to historical reasons, this class will perform 1D channel-wise dropout
        for 3D inputs (as done by :class:`nn.Dropout1d`). Thus, it currently does NOT
        support inputs without a batch dimension of shape :math:`(C, H, W)`. This
        behavior will change in a future release to interpret 3D inputs as no-batch-dim
        inputs. To maintain the old behavior, switch to :class:`nn.Dropout1d`.

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, L)`.
        - Output: :math:`(N, C, H, W)` or :math:`(N, C, L)` (same shape as input).

    Examples::

        >>> m = nn.Dropout2d(p=0.2)
        >>> input = torch.randn(20, 16, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        shape = list(input.size())
        new_shape = [shape[0] * shape[1]] + shape[2:]
        y = F.dropout2d(input.view(new_shape), self.p, self.training, self.inplace)
        return y.view(shape)

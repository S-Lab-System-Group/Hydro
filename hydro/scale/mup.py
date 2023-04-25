# Copyright 2022 Microsoft Corporation.
import yaml

from copy import copy, deepcopy
from torch import nn
from torch.nn import Linear
from torch.nn.modules.conv import _ConvNd

__BSH_COMMENT__ = """\
# This is a base shape file encoded in yaml
# - `null` indicates a dimension is "finite", i.e. a non-"width" dimension
# - a number indicates the base dimension of an "infinite" dimension, i.e. some notion of "width"
"""


def rescale_linear_bias(linear):
    """Rescale bias in nn.Linear layers to convert SP initialization to μP initialization.

    Warning: This method is NOT idempotent and should be called only once
    unless you know what you are doing.
    """
    if hasattr(linear, "_has_rescaled_params") and linear._has_rescaled_params:
        raise RuntimeError(
            "`rescale_linear_bias` has been called once before already. Unless you know what you are doing, usually you should not be calling `rescale_linear_bias` more than once.\n"
            "If you called `set_base_shapes` on a model loaded from a checkpoint, or just want to re-set the base shapes of an existing model, make sure to set the flag `rescale_params=False`.\n"
            "To bypass this error and *still rescale biases*, set `linear._has_rescaled_params=False` before this call."
        )
    if linear.bias is None:
        return
    fanin_mult = linear.weight.infshape[1].width_mult()
    linear.bias.data *= fanin_mult ** 0.5
    linear._has_rescaled_params = True


def get_shapes(model):
    return {name: param.shape for name, param in model.named_parameters()}


def get_infshapes(model):
    return {name: param.infshape for name, param in model.named_parameters()}


def save_base_shapes(model_or_shapes, file):
    if isinstance(model_or_shapes, nn.Module):
        sh = get_infshapes(model_or_shapes)
    elif isinstance(model_or_shapes, dict):
        sh = deepcopy(model_or_shapes)
    else:
        raise ValueError()
    sh = {k: s.base_shape() for k, s in sh.items()}
    s = yaml.dump(sh, None, indent=4)
    s = __BSH_COMMENT__ + s
    with open(file, "w") as f:
        f.write(s)


def load_base_shapes(filename):
    """Get a dict of `InfShape` from a filename."""
    with open(filename, "r") as f:
        d = yaml.safe_load(f)
    return {k: InfShape.from_base_shape(v) for k, v in d.items()}


def _dataparallel_hack(base_shapes, shapes):
    """Fix module name discrepancy caused by (Distributed)DataParallel module.

    The parameters of a (Distributed)DataParallel module all have names that
    start with 'module'. This causes a mismatch from non-DataParallel modules.
    This function tries to match `base_shapes` to `shapes`: if the latter starts
    with 'module', then make the former too; likewise if not.
    """
    if all(k.startswith("module.") for k in shapes) and all(not k.startswith("module.") for k in base_shapes):
        return {"module." + k: v for k, v in base_shapes.items()}, shapes
    if all(not k.startswith("module.") for k in shapes) and all(k.startswith("module.") for k in base_shapes):
        return {k.strip("module."): v for k, v in base_shapes.items()}, shapes
    return base_shapes, shapes


def _extract_shapes(x):
    """
    Input:
        x: can be any of the following:
            - `nn.Module`
            - dict of shapes
            - dict of `InfShape`
            - str of path to a base shapes (.bsh) file
    Output:
        If `x` is dict of `InfShape`, then output itself.
        If `x` is path, then output a dict of `InfShapes` loaded from `x`.
        Else, output the shapes (not `InfShape`) associated to `x`
    """
    if isinstance(x, nn.Module):
        x_shapes = get_shapes(x)
    elif isinstance(x, dict):
        x_shapes = deepcopy(x)
    elif isinstance(x, str):
        # x is file name
        x_shapes = load_base_shapes(x)
    else:
        raise ValueError(f"unhandled x type: {type(x)}")
    return x_shapes


def _zip_infshape_dict(base_shapes, shapes):
    """make a dict of `InfShape` from two dicts of shapes.
    Inputs:
        base_shapes: dict of base shapes or InfShape objects
        shapes: dict of shapes
    Output:
        dict of `InfShape` using `zip_infshape`
    """
    base_shapes, shapes = _dataparallel_hack(base_shapes, shapes)
    basenames = set(base_shapes.keys())
    names = set(shapes.keys())
    assert basenames == names, (
        f"`base_shapes` has extra names {basenames - names}. " f"`shapes` has extra names {names - basenames}."
    )
    infshapes = {}
    for name, bsh in base_shapes.items():
        infshapes[name] = zip_infshape(bsh, shapes[name])
    return infshapes


def zip_infshapes(base, target):
    """make a dict of `InfShape` from models or dicts.
    Inputs:
        base: a base `nn.Module` or a dict of shapes
        target: a target `nn.Module` or a dict of shapes
    Output:
        dict of `InfShape` using `zip_infshape`
    """
    base_shapes = _extract_shapes(base)
    target_shapes = _extract_shapes(target)
    return _zip_infshape_dict(base_shapes, target_shapes)


def clear_dims(infshape_dict):
    """
    Input:
        infshape_dict: dict of `InfShape`
    Output:
        the same dict but where all `InfDim` in all `InfShape`
        have their `dim` attribute set to None
    """
    d = deepcopy(infshape_dict)
    for _, v in d.items():
        for infdim in v:
            infdim.dim = None
    return d


def make_base_shapes(base_shapes, delta_shapes, savefile=None):
    """Make a base shape object from a base model/shapes and a delta model/shapes.

    Inputs:
        base:
            a base `nn.Module` or a dict of shapes
        delta:
            a "delta" model or a dict of shapes, for the sole purpose of
            determining which dimensions are "width" and will be scaled up and
            down in the target model.
        savefile:
            if a string, then the resulting base shape object is serialized to
            this location via yaml encoding.
    Outputs:
        base infshapes
    """
    bsh = clear_dims(zip_infshapes(base_shapes, delta_shapes))
    if savefile is not None:
        save_base_shapes(bsh, savefile)
    return bsh


def apply_infshapes(model, infshapes):
    for name, p in model.named_parameters():
        p.infshape = infshapes[name]


def set_base_shapes(model, base, rescale_params=True, delta=None, savefile=None):
    """Sets the `p.infshape` attribute for each parameter `p` of `model`.

    Inputs:
        model: nn.Module instance
        base: The base model.
            Can be nn.Module, a dict of shapes, a str, or None.
            If None, then defaults to `model`
            If str, then treated as filename for yaml encoding of a dict of base shapes.
        rescale_params:
            assuming the model is initialized using the default pytorch init (or
            He initialization etc that scale the same way with fanin): If True
            (default), rescales parameters to have the correct (μP) variances.
        do_assert: 
    Output:
        same object as `model`, after setting the `infshape` attribute of each parameter.
    """
    if base is None:
        base = model
    base_shapes = _extract_shapes(base)
    if delta is not None:
        delta_shapes = _extract_shapes(delta)
        base_shapes = _zip_infshape_dict(base_shapes, delta_shapes)
    shapes = get_shapes(model)
    infshapes = _zip_infshape_dict(base_shapes, shapes)
    if savefile is not None:
        save_base_shapes(infshapes, savefile)
    apply_infshapes(model, infshapes)
    if rescale_params:
        for name, module in model.named_modules():
            if isinstance(module, (Linear, _ConvNd)):
                rescale_linear_bias(module)
    return model


class InfDim:
    """A dimension with a base dimension, used for calculating μP scaling.

    An `InfDim` object is made up of 2 numbers: a dimension and a base
    dimension. If the base dimension is None, then this object represents a
    "finite", or "non-width" dimension. Otherwise, it represents an "infinite",
    or "width" dimension.
    """

    def __init__(self, base_dim, dim):
        self.base_dim = base_dim
        self.dim = dim

    def isinf(self):
        return self.base_dim is not None

    def width_mult(self):
        """Width multiplier used for calculating μP scaling.

        If finite, return 1.
        If infinite, return dim / base_dim.
        """
        if self.isinf():
            return self.dim / self.base_dim
        return 1

    def __repr__(self):
        return f"InfDim({self.base_dim}, {self.dim})"

    def __str__(self):
        if self.isinf():
            return repr(self)
        return f"FinDim({self.dim})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfDim):
            return False
        return self.base_dim == other.base_dim and self.dim == other.dim


class InfShape(tuple):
    """A tuple of `InfDim`s.

    This is intended to be attached to each parameter tensor `p` as `p.infshape`.
    """

    def __init__(self, *args, **kwargs):
        tuple.__init__(*args, **kwargs)
        for dim in self:
            if not isinstance(dim, InfDim):
                raise ValueError("Elements of InfShape needs to be of class InfDim")
        # set main to be the last dimension that is infinite
        # for inf x inf this is fanin
        # for inf x fin or fin x inf it's the unique inf dim
        # user can set this manually if necessary
        self.main_idx = self.main = None
        for i, dim in list(enumerate(self))[::-1]:
            if dim.isinf():
                self.main_idx = i
                self.main = dim
                break

    def fanin_fanout(self):
        assert len(self) >= 2, "fanin, fanout undefined for 1-dimensional weights"
        return self[1], self[0]

    def fanin_fanout_mult_ratio(self):
        fanin, fanout = self.fanin_fanout()
        return fanin.width_mult() / fanout.width_mult()

    def ninf(self):
        return sum(1 for dim in self if dim.isinf())

    def width_mult(self):
        if self.main is not None:
            return self.main.width_mult()
        return 1

    def base_shape(self):
        return [d.base_dim for d in self]

    def shape(self):
        return [d.dim for d in self]

    def __repr__(self):
        r = tuple.__repr__(self)[1:-1]
        return f"InfShape([{r}])"

    def serialize(self):
        d = {"base_shape": [], "shape": []}
        for infdim in self:
            d["shape"].append(infdim.dim)
            d["base_shape"].append(infdim.base_dim)
        return d

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfShape):
            return False
        return all(d == dd for d, dd in zip(self, other))

    @classmethod
    def deserialize(cls, d):
        infshape = []
        for base_dim, dim in zip(d["base_shape"], d["shape"]):
            infshape.append(InfDim(base_dim, dim))
        return InfShape(infshape)

    @classmethod
    def from_base_shape(cls, bsh):
        return InfShape([InfDim(bd, None) for bd in bsh])


def zip_infshape(base_dims, dims, fin_if_same=True):
    infshape = []
    for bd, d in zip(base_dims, dims):
        if isinstance(bd, InfDim):
            # retain bd's base_dim but overwrite dim
            infdim = copy(bd)
            infdim.dim = d
            infshape.append(infdim)
        elif isinstance(bd, int):
            if bd == d and fin_if_same:
                infshape.append(InfDim(None, d))
            else:
                infshape.append(InfDim(bd, d))
        else:
            raise ValueError(f"unhandled base_dim type: {type(bd)}")
    return InfShape(infshape)


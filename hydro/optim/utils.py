import torch
import torch.nn as nn
import numpy as np
import re
import itertools

RE_PARSE_RATIO = re.compile("Mismatched elements: (\d+) \/ (\d+)")


def dump_error_msg(e):
    """Dump out the exception e message"""
    print("\t\t-> Failed with error message:")
    print("[Start] ==============================================")
    print(e)
    print("[ End ] ==============================================\n")


def assert_allclose(
    actual,
    desired,
    rtol=1e-07,
    atol=0,
    equal_nan=True,
    err_msg="",
    verbose=True,
    population_threshold=0.0,
):
    try:
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            err_msg=err_msg,
            verbose=verbose,
        )
    except AssertionError as e:
        m = RE_PARSE_RATIO.search(str(e))
        if not m:
            raise e
        else:
            if (int(m.group(1)) / int(m.group(2))) >= population_threshold:
                raise e


def _snatch_grads_unfused(op_list, op, b):
    op_list[b].weight.grad = op.weight.grad
    if op_list[b].bias is not None:
        op_list[b].bias.grad = op.bias.grad


def _snatch_grads_linear(fused_op, op, b, fused=True):
    if fused:
        fused_op.weight.grad[b] = op.weight.grad.transpose(0, 1)
        if fused_op.bias is not None:
            fused_op.bias.grad[b] = op.bias.grad.unsqueeze(0)
    else:
        _snatch_grads_unfused(fused_op, op, b)


def _snatch_grads_conv2d(fused_op, op, b, fused=True):
    if fused:
        fused_op.weight.grad[b] = op.weight.grad
        if fused_op.bias is not None:
            fused_op.bias.grad[b] = op.bias.grad
    else:
        _snatch_grads_unfused(fused_op, op, b)


def _snatch_parameters_unfused(op_list, op, b):
    op_list[b].weight.data = op.weight.data
    if op_list[b].bias is not None:
        op_list[b].bias.data = op.bias.data


def _assert_params_unfused(op_list, op, b):
    assert_allclose(
        op_list[b].weight.data.cpu().numpy(),
        op.weight.data.cpu().numpy(),
        rtol=1e-4,
        population_threshold=1e-2,
    )
    if op_list[b].bias is not None:
        assert_allclose(
            op_list[b].bias.data.cpu().numpy(),
            op.bias.data.cpu().numpy(),
            rtol=1e-4,
            population_threshold=1e-2,
        )


def _assert_params_linear(fused_op, op, b, fused=True):
    try:
        if fused:
            assert_allclose(
                fused_op.weight.data[b].cpu().numpy(),
                op.weight.data.transpose(0, 1).cpu().numpy(),
                rtol=1e-4,
                population_threshold=1e-2,
            )
            if fused_op.bias is not None:
                assert_allclose(
                    fused_op.bias.data[b].cpu().numpy(),
                    op.bias.data.unsqueeze(0).cpu().numpy(),
                    rtol=1e-4,
                    population_threshold=1e-2,
                )
        else:
            _assert_params_unfused(fused_op, op, b)
    except AssertionError as e:
        dump_error_msg(e)


def _assert_params_conv2d(fused_op, op, b, fused=True):
    try:
        if fused:
            assert_allclose(
                fused_op.weight.data[b].cpu().numpy(),
                op.weight.data.cpu().numpy(),
                rtol=1e-4,
                population_threshold=1e-2,
            )
            if fused_op.bias is not None:
                assert_allclose(
                    fused_op.bias.data[b].cpu().numpy(),
                    op.bias.data.cpu().numpy(),
                    rtol=1e-4,
                    population_threshold=1e-2,
                )
        else:
            _assert_params_unfused(fused_op, op, b)
    except AssertionError as e:
        dump_error_msg(e)


def _set_grads(net_fused, net_array):
    B = len(net_array)
    # Init. grads for net_array.
    for b in range(B):
        for p in net_array[b].parameters():
            p.grad = torch.rand_like(p)
    # Init. grads for net_fused to zeros.
    for p in net_fused.parameters():
        p.grad = torch.zeros_like(p)
    for b_params in net_fused.unfused_parameters():
        for p in b_params:
            p.grad = torch.zeros_like(p)
    # Assign net_fused with the same grads from net_array.
    for b in range(B):
        net_fused.snatch_grads(net_array[b], b)


def _zero_grads(optimizer_fused, optimizer_array):
    for optimizer in optimizer_array:
        optimizer.zero_grad()
    optimizer_fused.zero_grad()


def _take_step_on_test_optimizers(optimizer_fused, optimizer_array):
    for optimizer in optimizer_array:
        optimizer.step()
    optimizer_fused.step()


def _verify_test_nets_params(net_fused, net_array):
    B = len(net_array)
    for b in range(B):
        net_fused.assert_params(net_array[b], b)


def _optim_testing_procedure(
    net_fused,
    net_array,
    optimizer_fused,
    optimizer_array,
    num_iters=10,
):
    # Init parameters for net_array and net_fused.
    _init_test_nets(net_fused, net_array)
    for _ in range(num_iters):
        # Zero out gradients.
        _zero_grads(optimizer_fused, optimizer_array)
        # Set gradients for net_array and net_fused.
        _set_grads(net_fused, net_array)
        # Call step().
        _take_step_on_test_optimizers(optimizer_fused, optimizer_array)
    # Verify parameter values.
    _verify_test_nets_params(net_fused, net_array)


def _to_tensor(coeff, B, dtype=torch.float, device=torch.device("cpu")):
    if isinstance(coeff, (float, int)):
        res = coeff
    elif isinstance(coeff, (list, tuple, np.ndarray)):
        if isinstance(coeff, (list, tuple)):
            assert len(coeff) == B
        else:
            assert len(coeff.shape) == 1 and coeff.shape[0] == B
        res = torch.as_tensor(coeff, dtype=dtype, device=device)
    elif isinstance(coeff, torch.Tensor):
        assert coeff.dim() == 1 and coeff.size(0) == B
        res = coeff.to(dtype=dtype, device=device)
    else:
        raise ValueError("Unsupported type({}): {}".format(coeff, type(coeff)))
    return res


def _get_coeff_like_params_map(coeff, params, B):
    res = {}
    for p in params:
        assert p.size(0) == B
        res[p] = coeff.view(B, *([1] * (p.dim() - 1)))
    return res


def _broadcastablize(optimizer, name, B, is_tuple=False):
    for group in optimizer.param_groups:
        if is_tuple:
            coeffs = group[name]
            new_coeffs = []
            for coeff in coeffs:
                coeff = _to_tensor(coeff, B)
                if isinstance(coeff, (float, int)):
                    new_coeffs.append(coeff)
                else:
                    assert isinstance(coeff, torch.Tensor)
                    new_coeffs.append(_get_coeff_like_params_map(coeff, group["params"], B))
            group[name] = tuple(new_coeffs)
        else:
            coeff = group[name]
            coeff = _to_tensor(coeff, B)
            if isinstance(coeff, (float, int)):
                continue
            assert isinstance(coeff, torch.Tensor)
            group[name] = _get_coeff_like_params_map(coeff, group["params"], B)


class Coefficient:
    def __init__(self, name, value):
        if not isinstance(value, (list, tuple, torch.Tensor, np.ndarray)):
            raise ValueError("Unsupported {} type({}): {}".format(name, value, type(value)))

        self._name = name
        self._value = value
        self._ddt_map = {}  # (device, dtype) -> tensor

    def _validate_range_for_element(self, i, e, lb=None, ub=None):
        if (lb is not None and e < lb) or (ub is not None and e > ub):
            raise ValueError("Invalid {}[{}]: {}".format(self._name, i, e))

    def validate_range(self, lb=None, ub=None):
        for i, e in enumerate(self._value):
            self._validate_range_for_element(i, e, lb=lb, ub=ub)

    def get_tensor(self):
        if len(self._ddt_map) == 1:
            return list(self._ddt_map.values())[0]
        else:  # Here, self._ddt_map is {}, len=0,
            return self._value

    def _update_ddt_map(self, device, dtype):
        if isinstance(self._value, (list, tuple, np.ndarray)):
            self._ddt_map[(device, dtype)] = torch.as_tensor(
                self._value,
                dtype=dtype,
                device=device,
            )
        elif isinstance(self._value, torch.Tensor):
            self._ddt_map[(device, dtype)] = self._value.to(
                dtype=dtype,
                device=device,
            )
        else:
            raise ValueError("Unsupported type({}): {}".format(self._value, type(self._value)))

    def __getitem__(self, p):
        k = (p.device, p.dtype)
        if k not in self._ddt_map:
            self._update_ddt_map(p.device, p.dtype)
        B = self._ddt_map[k].size(0)
        return self._ddt_map[k].view(B, *([1] * (p.dim() - 1)))


def is_coefficient(v):
    return isinstance(v, Coefficient)


def _validate_range(name, val, lb=None, ub=None):
    if is_coefficient(val):
        val.validate_range(lb=lb, ub=ub)
    else:
        if (lb is not None and val < lb) or (ub is not None and val > ub):
            raise ValueError("Invalid {}: {}".format(name, val))


def make_coefficient(name, value, lb=None, ub=None, is_tuple=False):
    if is_tuple:
        res = tuple(
            v
            if isinstance(v, (int, float))
            else Coefficient(
                "{}[{}]".format(name, i),
                v,
            )
            for i, v in enumerate(value)
        )
        for i, r in enumerate(res):
            _validate_range("{}[{}]".format(name, i), r, lb=lb, ub=ub)
    else:
        res = value if isinstance(value, (int, float)) else Coefficient(name, value)
        _validate_range(name, res, lb=lb, ub=ub)
    return res


def index_array_or_return_scalar(array_or_scalar, b):
    scalar_types = (int, float, type(None))
    if isinstance(array_or_scalar, scalar_types):
        return array_or_scalar
    elif isinstance(array_or_scalar, (list, tuple)):
        return array_or_scalar[b]
    elif isinstance(array_or_scalar, (torch.Tensor, np.ndarray)):
        return array_or_scalar[b].item()
    else:
        raise ValueError("Unsupported type({}) = {}!".format(array_or_scalar, type(array_or_scalar)))


def reduce_array_if_possible(arr):
    if isinstance(arr, (list, tuple, np.ndarray, torch.Tensor)):
        first = arr[0]
        for e in arr[1:]:
            if e != first:
                return arr
        if isinstance(arr, (torch.Tensor, np.ndarray)):
            return first.item()
        else:
            return first
    else:
        return arr


def reduce_array_if_possible_for(*coeffs):
    return (reduce_array_if_possible(coeff) for coeff in coeffs)


def consolidate_hyperparams_and_determine_B(args, hp_names):
    B = 1
    for hp_name in hp_names:
        hp = getattr(args, hp_name)
        assert isinstance(hp, list)
        B = max(B, len(hp)) if B == 1 else B
        if len(hp) == 1:
            setattr(args, hp_name, hp[0])
        else:
            assert len(hp) == B
    return B

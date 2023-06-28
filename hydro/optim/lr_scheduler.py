import math
import warnings
import weakref
from functools import wraps
from bisect import bisect_right

import torch
from torch.optim import Optimizer

from .utils import (
    reduce_array_if_possible_for,
    _to_tensor,
    _get_coeff_like_params_map,
    index_array_or_return_scalar,
    is_coefficient,
    make_coefficient,
    Coefficient,
)

__all__ = [
    "StepLR",
    "LRScheduler",
]

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1, B=1, verbose=False):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if isinstance(last_epoch, int) and last_epoch == -1:
            for group in optimizer.param_groups:
                if isinstance(group["lr"], dict):
                    group.setdefault(
                        "initial_lr",
                        {p: lr.detach().clone() for p, lr in group["lr"].items()},
                    )
                else:
                    group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified " "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(map(lambda group: group["initial_lr"], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.verbose = verbose
        self.B = B

        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate."""
        if is_verbose:
            if epoch is None:
                print("Adjusting learning rate" " of group {} to {:.4e}.".format(group, lr))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print("Epoch {}: adjusting learning rate" " of group {} to {:.4e}.".format(epoch_str, group, lr))

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                (epoch,) = reduce_array_if_possible_for(epoch)
                epoch = _to_tensor(epoch, self.B, dtype=torch.long)
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        # print(f"Epoch: {self.last_epoch}  LR: {self._last_lr[0]._value}")


# Including _LRScheduler for backwards compatibility
# Subclass instead of assign because we want __name__ of _LRScheduler to be _LRScheduler (assigning would make it LRScheduler).
class _LRScheduler(LRScheduler):
    pass


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


class StepLR(LRScheduler):
    """Pytorch 2.0
    Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, B=1, verbose=False):
        step_size, gamma, last_epoch = reduce_array_if_possible_for(step_size, gamma, last_epoch)
        step_size, gamma, last_epoch = (
            _to_tensor(step_size, B, dtype=torch.long),
            _to_tensor(gamma, B),
            _to_tensor(last_epoch, B, dtype=torch.long),
        )
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, B, verbose)

    def _calculate_lr_update_mask(self):
        if isinstance(self.last_epoch, int):
            epoch_nonzero_mask = torch.full(
                (self.B,),
                self.last_epoch != 0,
                dtype=torch.bool,
            )
        else:
            epoch_nonzero_mask = self.last_epoch != 0

        if isinstance(self.step_size, torch.Tensor) and isinstance(self.last_epoch, int):
            last_epoch = torch.full((self.B,), self.last_epoch, dtype=torch.long)
        else:
            last_epoch = self.last_epoch
        epoch_mod_step_size_zero_mask = last_epoch % self.step_size == 0
        if isinstance(epoch_mod_step_size_zero_mask, bool):
            epoch_mod_step_size_zero_mask = torch.full(
                (self.B,),
                epoch_mod_step_size_zero_mask,
                dtype=torch.bool,
            )
        return torch.logical_and(epoch_nonzero_mask, epoch_mod_step_size_zero_mask)

    def _calculate_multiplier(self, lr_update_mask):
        if isinstance(self.gamma, (int, float)):
            if lr_update_mask.all():
                multiplier = self.gamma
            else:
                multiplier = torch.ones((self.B,), dtype=torch.float)
                multiplier[lr_update_mask] = self.gamma
        else:
            multiplier = torch.ones((self.B,), dtype=torch.float)
            multiplier[lr_update_mask] = self.gamma[lr_update_mask]
        return multiplier

    def _update_lr(self, this_lr, params, multiplier):
        if isinstance(this_lr, dict):
            if isinstance(multiplier, (int, float)):
                res = {p: lr * multiplier for p, lr in this_lr.items()}
            else:
                multiplier_map = _get_coeff_like_params_map(multiplier, params, self.B)
                res = {p: lr * multiplier_map[p].to(p.device) for p, lr in this_lr.items()}
        elif isinstance(this_lr, list):
            if isinstance(multiplier, (int, float)):
                updated_lr = [lr * multiplier for lr in this_lr]
                res = make_coefficient("learning rate", updated_lr, lb=0.0, ub=float("inf"))
            else:
                raise NotImplementedError(f"got {this_lr}")
                # updated_lr = torch.mul(lr_tensor, multiplier.to(lr_tensor.device))
                # res = make_coefficient("learning rate", updated_lr, lb=0.0, ub=float("inf"))
        elif isinstance(this_lr, float):
            if isinstance(multiplier, (int, float)):
                updated_lr = this_lr * multiplier
            else:
                updated_lr = [this_lr * mul for mul in multiplier]
            res = make_coefficient("learning rate", updated_lr, lb=0.0, ub=float("inf"))
        elif isinstance(this_lr, Coefficient):
            lr_tensor = this_lr.get_tensor()
            if isinstance(multiplier, (int, float)):
                if isinstance(lr_tensor, list):
                    updated_lr = [lr * multiplier for lr in lr_tensor]
                else:
                    updated_lr = multiplier * lr_tensor
                res = make_coefficient("learning rate", updated_lr, lb=0.0, ub=float("inf"))
            else:
                updated_lr = torch.mul(lr_tensor, multiplier.to(lr_tensor.device))
                res = make_coefficient("learning rate", updated_lr, lb=0.0, ub=float("inf"))
                # multiplier_map = _get_coeff_like_params_map(multiplier, params, self.B)
                # res = {p: this_lr * mul.to(p.device) for p, mul in multiplier_map.items()}
        else:
            raise NotImplementedError(f"got {this_lr}")
        return res

    def get_lr(self):
        with torch.no_grad():
            if not self._get_lr_called_within_step:
                warnings.warn(
                    "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                    UserWarning,
                )

            lr_update_mask = self._calculate_lr_update_mask()
            if lr_update_mask.any():
                multiplier = self._calculate_multiplier(lr_update_mask)
                return [self._update_lr(group["lr"], group["params"], multiplier) for group in self.optimizer.param_groups]
            else:
                return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        with torch.no_grad():
            multiplier = self.gamma ** (self.last_epoch // self.step_size)
            return [
                self._update_lr(base_lr, group["params"], multiplier)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

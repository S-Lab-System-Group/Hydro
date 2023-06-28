from typing import List, Optional, Union

import math
import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    _use_grad_for_differentiable,
    _get_value,
    _stack_if_compiling,
    _dispatch_sqrt,
    _default_to_fused_or_foreach,
    _capturable_doc,
    _differentiable_doc,
    _foreach_doc,
    _fused_doc,
    _maximize_doc,
)
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

from .utils import Coefficient, is_coefficient, make_coefficient, reduce_array_if_possible_for

__all__ = ["AdamW", "adamw"]


class AdamW(Optimizer):
    r"""Pytorch 2.0
    Implements AdamW algorithm.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        scaling_num: Union[int, float] = -1,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        lr, eps, beta1, beta2, weight_decay = reduce_array_if_possible_for(lr, eps, betas[0], betas[1], weight_decay)
        betas = (beta1, beta2)
        lr = make_coefficient("learning rate", lr, lb=0.0, ub=float("inf"))
        eps = make_coefficient("epsilon value", eps, lb=0.0, ub=float("inf"))
        betas = make_coefficient("beta parameter at index", betas, lb=0.0, ub=1.0, is_tuple=True)
        weight_decay = make_coefficient("weight_decay value", weight_decay, lb=0.0, ub=float("inf"))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            scaling_num=scaling_num,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Suppor AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            if not all(p.is_cuda and torch.is_floating_point(p) for pg in self.param_groups for p in pg["params"]):
                raise RuntimeError("`fused=True` requires all the params to be CUDA, floating point Tensor")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((1,), dtype=torch.float, device=p.device)
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if amsgrad:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            state_steps.append(state["step"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                scaling_num=group["scaling_num"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


AdamW.__doc__ = r"""Implements AdamW algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Decoupled Weight Decay Regularization`_.
    """ + r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        {maximize}
        {foreach}
        {capturable}
        {differentiable}
        {fused}
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """.format(
    maximize=_maximize_doc,
    foreach=_foreach_doc,
    fused=_fused_doc,
    capturable=_capturable_doc,
    differentiable=_differentiable_doc,
)


def adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    amsgrad: bool,
    beta1: Union[float, Coefficient],
    beta2: Union[float, Coefficient],
    lr: Union[float, Coefficient],
    weight_decay: Union[float, Coefficient],
    eps: Union[float, Coefficient],
    maximize: bool,
    scaling_num: Union[int, float],
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        foreach = False  # not implemented in Hydro
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adamw
        raise NotImplementedError("Currently, `fused_adamw` is not implemented in Hydro")
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw
        raise NotImplementedError("Currently, `multi_tensor_adamw` is not implemented in Hydro")
    else:
        func = _single_tensor_adamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        scaling_num=scaling_num,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _single_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: Union[float, Coefficient],
    beta2: Union[float, Coefficient],
    lr: Union[float, Coefficient],
    weight_decay: Union[float, Coefficient],
    eps: Union[float, Coefficient],
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    scaling_num: Union[int, float],
):
    # assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."
            raise NotImplementedError("Capturable Adam not implemented for Hydro")
        else:
            assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        if is_coefficient(weight_decay) and is_coefficient(lr):
            param.mul_(1 - lr[param] * weight_decay[param])
        elif is_coefficient(weight_decay):
            param.mul_(1 - lr * weight_decay[param])
        elif is_coefficient(lr):
            param.mul_(1 - lr[param] * weight_decay)
        else:
            param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        if is_coefficient(beta1):
            exp_avg.mul_(beta1[param]).add_((1 - beta1[param]) * grad)
        else:
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if is_coefficient(beta2):
            exp_avg_sq.mul_(beta2[param]).add_((1 - beta2[param]) * grad * grad)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable:
            step = step_t

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sqs_i = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sqs_i = max_exp_avg_sqs[i]
                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sqs_i, exp_avg_sq))
                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = step_t.item()

            if is_coefficient(beta1):
                bias_correction1 = 1 - beta1[param] ** step
            else:
                bias_correction1 = 1 - beta1**step

            if is_coefficient(beta2):
                bias_correction2_sqrt = (1 - beta2[param] ** step).sqrt()
            else:
                bias_correction2_sqrt = math.sqrt(1 - beta2**step)

            if scaling_num > 0 and is_coefficient(lr):
                if param.infshape.ninf() == 2:
                    step_size = scaling_num * lr[param] / bias_correction1
                else:
                    step_size = lr[param] / bias_correction1
            elif is_coefficient(lr):
                step_size = lr[param] / bias_correction1
            else:
                step_size = lr / bias_correction1

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                if is_coefficient(eps):
                    denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps[param])
                else:
                    denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                if is_coefficient(eps):
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps[param])
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            if torch.is_tensor(step_size):
                param.add_(-step_size * (exp_avg / denom))
            else:
                param.addcdiv_(exp_avg, denom, value=-step_size)


def _multi_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    scaling_num: Union[int, float],
):
    if len(params) == 0:
        return

    if capturable:
        assert all(
            p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)
        ), "If capturable=True, params and state_steps must be CUDA tensors."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for (
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_max_exp_avg_sqs,
        device_state_steps,
    ) in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(tuple(device_grads))  # type: ignore[assignment]

        device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
        device_exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avgs]
        device_exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avg_sqs]
        device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]

        # update steps
        torch._foreach_add_(device_state_steps, 1)

        # Perform stepweight decay
        torch._foreach_mul_(device_params, 1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        torch._foreach_mul_(device_exp_avgs, beta1)
        torch._foreach_add_(device_exp_avgs, device_grads, alpha=1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        if capturable:
            # TODO: use foreach_pow if/when foreach_pow is added
            bias_correction1 = [torch.pow(beta1, step) for step in device_state_steps]
            bias_correction2 = [torch.pow(beta2, step) for step in device_state_steps]
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            torch._foreach_neg_(bias_correction1)
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            step_size = torch._foreach_div(bias_correction1, lr)
            torch._foreach_reciprocal_(step_size)
            torch._foreach_neg_(step_size)

            bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                max_exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                torch._foreach_div_(
                    max_exp_avg_sq_sqrt,
                    torch._foreach_mul(bias_correction2_sqrt, step_size),
                )
                eps_over_step_size = torch._foreach_div(step_size, eps)
                torch._foreach_reciprocal_(eps_over_step_size)
                denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps_over_step_size)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
                torch._foreach_div_(exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
                eps_over_step_size = torch._foreach_div(step_size, eps)
                torch._foreach_reciprocal_(eps_over_step_size)
                denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)

            torch._foreach_addcdiv_(device_params, device_exp_avgs, denom)
        else:
            bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                max_exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
                torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction2_sqrt)
                denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
                torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
                denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

            torch._foreach_addcdiv_(device_params, device_exp_avgs, denom, step_size)


def _fused_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,  # Needed for consistency.
    differentiable: bool,
    scaling_num: Union[int, float],
) -> None:
    if differentiable:
        raise RuntimeError("_fused_adamw is not differentiable")
    grad_scale_dict = {grad_scale.device: grad_scale} if grad_scale is not None else None
    found_inf_dict = {found_inf.device: found_inf} if found_inf is not None else None
    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for device, dtype in grouped_tensors:
        (
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
        ) = grouped_tensors[(device, dtype)]
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            if device not in grad_scale_dict:
                grad_scale_dict[device] = grad_scale.to(device, non_blocking=True)
            device_grad_scale = grad_scale_dict[device]
        if found_inf is not None:
            if found_inf not in found_inf_dict:
                found_inf_dict[device] = found_inf.to(device, non_blocking=True)
            device_found_inf = found_inf_dict[device]
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adamw_(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
            amsgrad=amsgrad,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            torch._foreach_sub_(device_state_steps, [device_found_inf] * len(device_state_steps))

"""LM-head optimizers used by the row-geometry comparison."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer


class HeadAdamW(AdamW):
    """Stock AdamW for the LM head."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        parameters = tuple(params)
        if len(parameters) != 1 or parameters[0].ndim != 2:
            raise ValueError("the LM-head optimizer must own exactly one matrix")
        group = {
            "params": list(parameters),
            "role": "lm_head",
            "base_lr": float(lr),
        }
        super().__init__(
            [group],
            lr=float(lr),
            betas=betas,
            eps=float(eps),
            weight_decay=float(weight_decay),
        )

    @torch.no_grad()
    def lm_head_step(self) -> Tensor:
        """Reconstruct the latest signed step, excluding decoupled weight decay."""

        group = self.param_groups[0]
        parameter = group["params"][0]
        gradient = parameter.grad
        state = self.state[parameter]
        if gradient is None or "step" not in state:
            raise RuntimeError("head diagnostics require a completed optimizer step")

        step = int(state["step"].item())
        beta1, beta2 = (float(value) for value in group["betas"])
        signal = state["exp_avg"] / (1.0 - beta1**step)
        second_moment = state["exp_avg_sq"] / (1.0 - beta2**step)
        update = signal / (second_moment.sqrt() + float(group["eps"]))
        return (-float(group["lr"]) * update).detach()


class RowNorm(Optimizer):
    """Row-normalized EMA with decoupled LM-head weight decay.

    For gradient ``G`` and stored momentum ``M``, each step computes
    ``M = beta * M + (1 - beta) * G`` and normalizes the bias-corrected
    moment ``M / (1 - beta**step)``. It then applies
    ``W = (1 - lr * weight_decay) * W - lr * normalized_M``.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        beta: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        values = (lr, beta, eps, weight_decay)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("RowNorm hyperparameters must be finite")
        if lr < 0 or not 0 <= beta < 1 or eps <= 0 or weight_decay < 0:
            raise ValueError(
                "RowNorm requires lr >= 0, beta in [0, 1), eps > 0, and weight_decay >= 0"
            )
        parameters = tuple(params)
        if len(parameters) != 1 or parameters[0].ndim != 2:
            raise ValueError("RowNorm must own exactly one LM-head matrix")
        defaults = {
            "lr": float(lr),
            "base_lr": float(lr),
            "role": "lm_head",
            "beta": float(beta),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "bias_correction": True,
        }
        super().__init__(parameters, defaults)

    @staticmethod
    def _normalize_rows(buffer: Tensor, eps: float) -> Tensor:
        return buffer / (torch.linalg.vector_norm(buffer, dim=1, keepdim=True) + eps)

    @staticmethod
    def _bias_correction_factor(beta: float, step: int) -> float:
        if step < 1:
            raise ValueError("RowNorm step must be positive for bias correction")
        if beta == 0:
            return 1.0
        return -math.expm1(step * math.log(beta))

    @classmethod
    def _normalized_ema(
        cls,
        buffer: Tensor,
        *,
        beta: float,
        step: int,
        eps: float,
    ) -> Tensor:
        correction = cls._bias_correction_factor(beta, step)
        # This is algebraically identical to normalizing buffer / correction,
        # but avoids materializing another full LM-head-sized tensor.
        return cls._normalize_rows(buffer, eps * correction)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta = float(group["beta"])
            learning_rate = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("RowNorm does not support sparse gradients")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                    state["step"] = 0
                elif "step" not in state:
                    raise RuntimeError(
                        "cannot resume a legacy RowNorm optimizer state without a step counter"
                    )
                buffer = state["momentum_buffer"]
                buffer.mul_(beta).add_(gradient, alpha=1.0 - beta)
                state["step"] += 1
                update = self._normalized_ema(
                    buffer,
                    beta=beta,
                    step=int(state["step"]),
                    eps=float(group["eps"]),
                )
                parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(update, alpha=-learning_rate)
        return loss

    @torch.no_grad()
    def lm_head_step(self) -> Tensor:
        """Reconstruct the latest signed step, excluding decoupled weight decay."""

        group = self.param_groups[0]
        parameter = group["params"][0]
        gradient = parameter.grad
        state = self.state[parameter]
        if gradient is None or "momentum_buffer" not in state:
            raise RuntimeError("head diagnostics require a completed optimizer step")
        if "step" not in state:
            raise RuntimeError(
                "cannot report RowNorm diagnostics without a step counter"
            )
        buffer = state["momentum_buffer"]
        step = int(state["step"])
        update = self._normalized_ema(
            buffer,
            beta=float(group["beta"]),
            step=step,
            eps=float(group["eps"]),
        )
        return (-float(group["lr"]) * update).detach()

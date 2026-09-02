"""Muon optimizer and its Newton--Schulz matrix transform."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def orthogonalize_newton_schulz(matrix: Tensor, steps: int, eps: float) -> Tensor:
    """Apply the quintic Newton--Schulz transform used by the paper's Muon."""

    if matrix.ndim != 2:
        raise ValueError(f"Muon expects matrices, got shape {tuple(matrix.shape)}")
    transposed = matrix.shape[0] > matrix.shape[1]
    value = matrix.mT if transposed else matrix
    value = value / (torch.linalg.vector_norm(value) + eps)
    a, b, c = NS_COEFFICIENTS
    for _ in range(steps):
        gram = value @ value.mT
        value = a * value + (b * gram + c * gram @ gram) @ value
    return value.mT if transposed else value


class Muon(Optimizer):
    """FP32 Muon with optional Nesterov momentum and Algorithm 8 shape scaling."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        momentum: float,
        nesterov: bool,
        weight_decay: float,
        ns_steps: int,
        eps: float,
    ) -> None:
        if lr < 0 or not 0 <= momentum < 1 or weight_decay < 0 or ns_steps < 0 or eps < 0:
            raise ValueError("invalid Muon hyperparameters")
        if not isinstance(nesterov, bool):
            raise TypeError("Muon nesterov must be a bool")
        defaults = dict(
            lr=float(lr),
            base_lr=float(lr),
            role="backbone",
            momentum=float(momentum),
            nesterov=nesterov,
            weight_decay=float(weight_decay),
            ns_steps=int(ns_steps),
            eps=float(eps),
        )
        super().__init__(params, defaults)
        if any(parameter.ndim != 2 for group in self.param_groups for parameter in group["params"]):
            raise ValueError("Muon can only own two-dimensional hidden matrices")

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(gradient)
                direction = (
                    gradient.add(buffer, alpha=momentum)
                    if group["nesterov"]
                    else buffer
                )
                update = orthogonalize_newton_schulz(
                    direction, int(group["ns_steps"]), float(group["eps"])
                )
                shape_scale = math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
                update.mul_(shape_scale)
                parameter.mul_(1.0 - lr * float(group["weight_decay"]))
                parameter.add_(update, alpha=-lr)
        return loss

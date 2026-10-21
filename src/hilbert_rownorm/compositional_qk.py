"""Compositional Muon for attention query--key pairs (Keigwin et al., 2026).

The attention probabilities see the query and key weights only through the
per-head product W_Q,h^T W_K,h. The half-split rule constrains the operator
norm of each half of that product's first-order change by whitening each
factor's momentum with the partner's inverse Gram root before and after
orthogonalization. In ``nn.Linear`` (out, in) layout, per head,

    delta W_Q,h = -(lr * mp / 2) C_K^{-1} msign(C_K^{-1} M_Q,h),
    C_K = (W_K,h W_K,h^T + damping * I)^{1/2},

and symmetrically for the key. As damping tends to zero this gives
||delta W_Q,h^T W_K,h||_op <= lr * mp / 2, the constraint that bounding the
Hilbert distance of attention distributions places on the update. The
``isotropic`` variant replaces C_K^{-1} by one scalar per head,
(||W_K,h||_F^2 / head_dim + damping)^{-1/2}, which commutes with RoPE
rotations but meets the bound only approximately.

Momentum, Nesterov, decoupled weight decay, FP32 updates, and the
Newton--Schulz transform follow this repository's Muon. No shape scaling is
applied: for head_dim <= model dim, Muon's factor is one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from .muon import orthogonalize_newton_schulz

WHITENING_MODES = ("isotropic", "full")


class CompositionalQK(Optimizer):
    """Half-split Compositional Muon on per-layer (query, key) weight pairs."""

    def __init__(
        self,
        pairs: Iterable[tuple[Tensor, Tensor]],
        *,
        head_dim: int,
        lr: float,
        momentum: float,
        nesterov: bool,
        weight_decay: float,
        mp: float,
        damping: float,
        whitening: str,
        ns_steps: int,
        eps: float,
    ) -> None:
        self.pairs = [tuple(pair) for pair in pairs]
        values = (lr, momentum, weight_decay, mp, damping, eps)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("CompositionalQK hyperparameters must be finite")
        if lr < 0 or not 0 <= momentum < 1 or weight_decay < 0 or mp <= 0 or damping < 0:
            raise ValueError("invalid CompositionalQK hyperparameters")
        if not isinstance(nesterov, bool):
            raise TypeError("CompositionalQK nesterov must be a bool")
        if not isinstance(ns_steps, int) or ns_steps < 0 or eps < 0:
            raise ValueError("invalid CompositionalQK Newton-Schulz settings")
        if whitening not in WHITENING_MODES:
            raise ValueError(f"whitening must be one of {WHITENING_MODES}, got {whitening!r}")
        if not isinstance(head_dim, int) or head_dim < 1:
            raise ValueError("head_dim must be a positive integer")
        if not self.pairs or any(len(pair) != 2 for pair in self.pairs):
            raise ValueError("CompositionalQK needs at least one (query, key) pair")
        for query, key in self.pairs:
            if query.ndim != 2 or query.shape != key.shape or query.shape[0] % head_dim:
                raise ValueError(
                    "query and key must be equal-shape (heads * head_dim, dim) matrices"
                )
        self.head_dim = head_dim
        defaults = dict(
            lr=float(lr),
            base_lr=float(lr),
            role="attention_qk",
            momentum=float(momentum),
            nesterov=nesterov,
            weight_decay=float(weight_decay),
            mp=float(mp),
            damping=float(damping),
            whitening=whitening,
            ns_steps=int(ns_steps),
            eps=float(eps),
        )
        super().__init__([weight for pair in self.pairs for weight in pair], defaults)

    def _heads(self, matrix: Tensor) -> Tensor:
        return matrix.view(-1, self.head_dim, matrix.shape[-1])

    def _direction(self, parameter: Tensor, momentum: float, nesterov: bool) -> Tensor:
        state = self.state[parameter]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(parameter)
        buffer = state["momentum_buffer"]
        buffer.mul_(momentum).add_(parameter.grad)
        return parameter.grad.add(buffer, alpha=momentum) if nesterov else buffer

    def _partner_inverse_root(self, weight: Tensor, group: dict[str, Any]) -> Tensor:
        """``(W_h W_h^T + damping I)^{-1/2}`` per head, or its scalar approximation."""

        heads = self._heads(weight.detach().float())
        damping = float(group["damping"])
        identity = torch.eye(self.head_dim, device=heads.device, dtype=heads.dtype)
        if group["whitening"] == "isotropic":
            scale = (heads.square().sum(dim=(1, 2)) / self.head_dim + damping).rsqrt()
            return scale[:, None, None] * identity
        gram = heads @ heads.mT + damping * identity
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        inverse_roots = eigenvalues.clamp_min(1e-12).rsqrt()
        return (eigenvectors * inverse_roots.unsqueeze(-2)) @ eigenvectors.mT

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        lr = float(group["lr"])
        momentum = float(group["momentum"])
        step_scale = lr * 0.5 * float(group["mp"])
        for query, key in self.pairs:
            if query.grad is None or key.grad is None:
                continue
            if query.grad.is_sparse or key.grad.is_sparse:
                raise RuntimeError("CompositionalQK does not support sparse gradients")
            query_direction = self._direction(query, momentum, group["nesterov"])
            key_direction = self._direction(key, momentum, group["nesterov"])
            partner_inverse = {
                id(query): self._partner_inverse_root(key, group),
                id(key): self._partner_inverse_root(query, group),
            }
            for parameter, direction in ((query, query_direction), (key, key_direction)):
                inverse = partner_inverse[id(parameter)]
                whitened = inverse @ self._heads(direction.float())
                update = orthogonalize_newton_schulz(
                    whitened, int(group["ns_steps"]), float(group["eps"])
                )
                update = (inverse @ update).reshape(parameter.shape)
                parameter.mul_(1.0 - lr * float(group["weight_decay"]))
                parameter.add_(update.to(parameter.dtype), alpha=-step_scale)
        return loss

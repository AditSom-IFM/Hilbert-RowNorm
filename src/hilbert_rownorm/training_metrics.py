"""Distributed training metrics accumulated between tracker reports."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class TrainingIntervalMetrics:
    """Host values for one completed interval across every training rank."""

    loss: float
    grad_norm: float
    grad_norm_max: float
    clipped_fraction: float
    would_clip_fraction: float
    updates: int
    peak_allocated_gib: float
    peak_reserved_gib: float


class TrainingIntervalAccumulator:
    """Retain detached device scalars and reduce them only when reporting."""

    def __init__(
        self,
        *,
        device: torch.device,
        max_gradient_norm: float,
        gradient_clipping: bool,
    ) -> None:
        self.device = device
        self.max_gradient_norm = max_gradient_norm
        self.gradient_clipping = gradient_clipping
        self.losses: list[Tensor] = []
        self.grad_norms: list[Tensor] = []

    def add(self, loss: Tensor, grad_norm: Tensor) -> None:
        self.losses.append(loss.detach())
        self.grad_norms.append(grad_norm.detach())

    def collect(self) -> TrainingIntervalMetrics:
        """Reduce the buffered interval with the same collectives on every rank."""

        updates = len(self.losses)
        losses = torch.stack(self.losses)
        grad_norms = torch.stack(self.grad_norms)
        would_clip_flags = grad_norms > self.max_gradient_norm
        clipped_count = (
            would_clip_flags.sum()
            if self.gradient_clipping
            else would_clip_flags.new_zeros(())
        )
        sums = torch.stack(
            (
                losses.sum(),
                grad_norms.sum(),
                clipped_count,
                would_clip_flags.sum(),
            )
        )
        maximum = grad_norms.max()
        memory = torch.zeros(2, device=self.device)
        if self.device.type == "cuda":
            memory[0] = torch.cuda.max_memory_allocated(self.device) / 2**30
            memory[1] = torch.cuda.max_memory_reserved(self.device) / 2**30
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(sums)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        loss, grad_norm, clipped, would_clip = (
            sums / (updates * world_size)
        ).tolist()
        allocated, reserved = memory.tolist()
        self.losses.clear()
        self.grad_norms.clear()
        return TrainingIntervalMetrics(
            loss=loss,
            grad_norm=grad_norm,
            grad_norm_max=maximum.item(),
            clipped_fraction=clipped,
            would_clip_fraction=would_clip,
            updates=updates,
            peak_allocated_gib=allocated,
            peak_reserved_gib=reserved,
        )

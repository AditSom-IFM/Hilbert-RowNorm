"""Geometry measurements for the most recent LM-head optimizer step."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class HeadStepProvider(Protocol):
    """An optimizer that can reconstruct its latest LM-head step."""

    def lm_head_step(self) -> Tensor:
        """Return the signed, learning-rate-scaled step before weight decay."""


def exact_diameter_update_schedule(
    total_updates: int,
    *,
    every_log: bool,
    log_every: int,
) -> tuple[int, ...]:
    """Return logging updates, including the final update, when enabled."""

    if not isinstance(total_updates, int) or total_updates < 1:
        raise ValueError("total updates must be positive")
    if not every_log:
        return ()
    if not isinstance(log_every, int) or log_every < 1:
        raise ValueError("exact diameter every log requires a positive log interval")
    updates = set(range(log_every, total_updates + 1, log_every))
    updates.add(total_updates)
    return tuple(sorted(updates))


@torch.inference_mode()
def exact_row_diameter(matrix: Tensor, *, block_rows: int) -> float:
    """Compute ``max_ij ||row_i-row_j||_2`` with bounded temporary storage."""

    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("diameter input must be a non-empty matrix")
    if not matrix.is_floating_point():
        raise TypeError("diameter input must be floating point")
    if not isinstance(block_rows, int) or block_rows < 1:
        raise ValueError("diameter block rows must be positive")
    value = matrix.detach().to(dtype=torch.float32)
    maximum = value.new_zeros(())
    rows = value.shape[0]
    for left_start in range(0, rows, block_rows):
        left = value[left_start : left_start + block_rows]
        for right_start in range(left_start, rows, block_rows):
            right = value[right_start : right_start + block_rows]
            distances = torch.cdist(
                left,
                right,
                p=2,
                compute_mode="use_mm_for_euclid_dist",
            )
            maximum = torch.maximum(maximum, distances.max())
    result = float(maximum.item())
    if not math.isfinite(result):
        raise ValueError("diameter input must contain only finite values")
    return result


class HeadGeometry:
    """Expose the latest head step and its scheduled exact diameter."""

    def __init__(
        self,
        optimizers: Sequence[object],
        *,
        rank: int,
        total_updates: int,
        exact_diameter_every_log: bool,
        log_every: int,
        diameter_block_rows: int,
        hilbert_probe_tokens: int,
    ) -> None:
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be non-negative")
        if not isinstance(diameter_block_rows, int) or diameter_block_rows < 1:
            raise ValueError("diameter block rows must be positive")
        if not isinstance(hilbert_probe_tokens, int) or hilbert_probe_tokens < 0:
            raise ValueError("Hilbert probe tokens must be non-negative")

        self.rank = rank
        self.diameter_block_rows = diameter_block_rows
        self.exact_diameter_updates = exact_diameter_update_schedule(
            total_updates,
            every_log=exact_diameter_every_log,
            log_every=log_every,
        )
        enabled = bool(self.exact_diameter_updates or hilbert_probe_tokens)
        providers = [
            optimizer for optimizer in optimizers if isinstance(optimizer, HeadStepProvider)
        ]
        if enabled and len(providers) != 1:
            raise ValueError("head geometry requires exactly one LM-head optimizer")
        self.provider = providers[0] if enabled else None

    def exact_diameter_due(self, update: int) -> bool:
        """Return whether the exact diameter is scheduled at ``update``."""

        return update in self.exact_diameter_updates

    def optimizer_step(self, enabled: bool) -> Tensor | None:
        """Return the latest signed head step on every training rank."""

        if not enabled:
            return None
        if self.provider is None:
            raise RuntimeError("head geometry is not enabled")
        return self.provider.lm_head_step()

    def exact_diameter(self, enabled: bool) -> float | None:
        """Compute the latest exact step diameter on rank zero."""

        if not enabled or self.rank != 0:
            return None
        step = self.optimizer_step(True)
        assert step is not None
        return exact_row_diameter(step, block_rows=self.diameter_block_rows)

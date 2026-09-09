"""Geometry measurements for the most recent LM-head optimizer step."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

# Historical paper measurements used uncentered FP32 cdist (v1).
DIAMETER_ALGORITHM = "centered_cdist_v2"


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
    """Measure ``max_ij ||row_i-row_j||_2`` by an exhaustive blocked search.

    Translation removes a common row offset before the matrix-multiplication
    distance calculation. Each block's maximizing candidate is then evaluated
    by direct subtraction of the original rows. FP64 input stays FP64; lower
    precision inputs are promoted to FP32. Storage is O(V d + block_rows**2).

    The compatibility name ``exact`` denotes all-pair coverage, not exact
    arithmetic: floating-point rounding can still affect candidate selection.
    This v2 arithmetic differs from the frozen, uncentered FP32 paper metrics.
    """

    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("diameter input must be a non-empty matrix")
    if not matrix.is_floating_point():
        raise TypeError("diameter input must be floating point")
    if not isinstance(block_rows, int) or block_rows < 1:
        raise ValueError("diameter block rows must be positive")
    dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    value = matrix.detach().to(dtype=dtype)
    if not torch.isfinite(value).all():
        raise ValueError("diameter input must contain only finite values")
    centered = value - value[:1]
    maximum = value.new_zeros(())
    rows = value.shape[0]
    for left_start in range(0, rows, block_rows):
        left = centered[left_start : left_start + block_rows]
        for right_start in range(left_start, rows, block_rows):
            right = centered[right_start : right_start + block_rows]
            distances = torch.cdist(
                left,
                right,
                p=2,
                compute_mode="use_mm_for_euclid_dist",
            )
            candidate = distances.flatten().argmax().reshape(1)
            left_index = candidate.div(right.shape[0], rounding_mode="floor") + left_start
            right_index = candidate.remainder(right.shape[0]) + right_start
            difference = value.index_select(0, left_index) - value.index_select(0, right_index)
            distance = torch.linalg.vector_norm(difference)
            maximum = torch.maximum(maximum, distance)
    result = float(maximum.item())
    if not math.isfinite(result):
        raise ValueError("diameter input must contain only finite values")
    return result


class HeadGeometry:
    """Expose the latest head step and its scheduled exhaustive diameter."""

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
        """Compute the latest exhaustive step diameter on rank zero."""

        if not enabled or self.rank != 0:
            return None
        step = self.optimizer_step(True)
        assert step is not None
        return exact_row_diameter(step, block_rows=self.diameter_block_rows)

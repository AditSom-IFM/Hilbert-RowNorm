"""Pure construction of the metric records emitted by the training tracker."""

from __future__ import annotations

import math
from typing import Any

from .hilbert_diagnostics import HilbertStepRms
from .training_metrics import TrainingIntervalMetrics


def add_training_metrics(
    record: dict[str, Any],
    metrics: TrainingIntervalMetrics,
    *,
    update: int,
    previous_update: int,
    tokens_per_step: int,
    interval_seconds: float,
    elapsed_seconds: float,
) -> None:
    """Append one reduced training interval using the stable W&B field order."""

    record.update(
        {
            "train/loss": metrics.loss,
            "train/perplexity": math.exp(metrics.loss),
            "train/grad_norm": metrics.grad_norm,
            "train/grad_norm_max": metrics.grad_norm_max,
            "train/clipped_fraction": metrics.clipped_fraction,
            "train/would_clip_fraction": metrics.would_clip_fraction,
            "performance/tokens_per_second": (
                (update - previous_update) * tokens_per_step / interval_seconds
            ),
            "performance/seconds_per_update": interval_seconds / metrics.updates,
            "performance/elapsed_seconds": elapsed_seconds,
            "memory/peak_allocated_gib": metrics.peak_allocated_gib,
            "memory/peak_reserved_gib": metrics.peak_reserved_gib,
        }
    )


def add_validation_metrics(
    record: dict[str, Any],
    *,
    loss: float,
    tokens: int,
    seconds: float,
    hilbert: HilbertStepRms | None,
) -> None:
    """Append validation loss and the optional Hilbert RMS."""

    record.update(
        {
            "validation/loss": loss,
            "validation/perplexity": math.exp(loss),
            "validation/tokens": tokens,
            "validation/seconds": seconds,
            "events/evaluation": 1,
        }
    )
    if hilbert is not None:
        record.update(
            {
                "validation/hilbert_step/rms": hilbert.rms,
                "validation/hilbert_step/tokens": hilbert.tokens,
            }
        )


def add_head_diameter(record: dict[str, Any], diameter: float) -> None:
    """Append the exact diameter of the latest LM-head step."""

    record["diagnostics/lm_head_step/diameter_exact"] = diameter


def add_checkpoint_metrics(record: dict[str, Any], seconds: float) -> None:
    """Append the checkpoint event and its measured duration."""

    record.update(
        {
            "events/checkpoint": 1,
            "performance/checkpoint_seconds": seconds,
        }
    )


def build_final_summary(
    *,
    updates: int,
    tokens_per_step: int,
    parameter_count: int,
    elapsed_seconds: float,
    validation_loss: float | None,
    validation_hilbert: HilbertStepRms | None,
) -> dict[str, float | int]:
    """Build the successful-run summary without reading tracker state."""

    tokens = updates * tokens_per_step
    summary: dict[str, float | int] = {
        "final_update": updates,
        "final_tokens": tokens,
        "final_tpp": tokens / parameter_count,
        "elapsed_seconds": elapsed_seconds,
        "end_to_end_tokens_per_second": tokens / elapsed_seconds,
    }
    if validation_loss is not None:
        summary["final_validation_loss"] = validation_loss
    if validation_hilbert is not None:
        summary.update(
            {
                "final_validation_hilbert_step_rms": validation_hilbert.rms,
                "final_validation_hilbert_step_tokens": validation_hilbert.tokens,
            }
        )
    return summary

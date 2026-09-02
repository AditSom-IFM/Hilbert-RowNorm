from dataclasses import FrozenInstanceError

import pytest
import torch

import hilbert_rownorm.training_metrics as training_metrics_module
from hilbert_rownorm.training_metrics import (
    TrainingIntervalAccumulator,
    TrainingIntervalMetrics,
)


def test_accumulator_returns_named_metrics_and_clears_detached_buffers() -> None:
    accumulator = TrainingIntervalAccumulator(
        device=torch.device("cpu"),
        max_gradient_norm=1.0,
        gradient_clipping=True,
    )
    first_loss = torch.tensor(2.0, requires_grad=True)
    first_grad_norm = torch.tensor(0.5, requires_grad=True)
    accumulator.add(first_loss, first_grad_norm)
    accumulator.add(torch.tensor(4.0), torch.tensor(2.0))

    assert not accumulator.losses[0].requires_grad
    assert not accumulator.grad_norms[0].requires_grad

    metrics = accumulator.collect()

    assert metrics == TrainingIntervalMetrics(
        loss=3.0,
        grad_norm=1.25,
        grad_norm_max=2.0,
        clipped_fraction=0.5,
        would_clip_fraction=0.5,
        updates=2,
        peak_allocated_gib=0.0,
        peak_reserved_gib=0.0,
    )
    assert accumulator.losses == []
    assert accumulator.grad_norms == []
    with pytest.raises(FrozenInstanceError):
        metrics.loss = 0.0


def test_accumulator_uses_exact_ddp_collective_order(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], object | None]] = []

    def all_reduce(values: torch.Tensor, op=None) -> None:
        calls.append((tuple(values.shape), op))
        if len(calls) == 1:
            values.add_(values.new_tensor((10.0, 5.5, 2.0, 2.0)))
        elif len(calls) == 2:
            values.fill_(4.0)
        else:
            values.copy_(values.new_tensor((7.0, 9.0)))

    monkeypatch.setattr(training_metrics_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(training_metrics_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(training_metrics_module.dist, "all_reduce", all_reduce)
    accumulator = TrainingIntervalAccumulator(
        device=torch.device("cpu"),
        max_gradient_norm=1.0,
        gradient_clipping=True,
    )
    accumulator.add(torch.tensor(2.0), torch.tensor(0.5))
    accumulator.add(torch.tensor(4.0), torch.tensor(2.0))

    metrics = accumulator.collect()

    assert calls == [
        ((4,), None),
        ((), torch.distributed.ReduceOp.MAX),
        ((2,), torch.distributed.ReduceOp.MAX),
    ]
    assert metrics == TrainingIntervalMetrics(
        loss=4.0,
        grad_norm=2.0,
        grad_norm_max=4.0,
        clipped_fraction=0.75,
        would_clip_fraction=0.75,
        updates=2,
        peak_allocated_gib=7.0,
        peak_reserved_gib=9.0,
    )
    assert accumulator.losses == []
    assert accumulator.grad_norms == []

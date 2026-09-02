from dataclasses import fields

import pytest
import torch

import hilbert_rownorm.hilbert_diagnostics as diagnostics
from hilbert_rownorm.hilbert_diagnostics import (
    HilbertRmsAccumulator,
    HilbertStepRms,
)


def test_hilbert_rms_is_the_rms_of_fp32_logit_ranges() -> None:
    step_logits = torch.tensor(
        [[1.0, -2.0, 4.0], [3.0, 3.0, -1.0]],
        dtype=torch.float64,
    )
    accumulator = HilbertRmsAccumulator(torch.device("cpu"))
    accumulator.add(step_logits[:1])
    accumulator.add(step_logits[1:])

    result = accumulator.compute()
    ranges = step_logits.float().amax(dim=1) - step_logits.float().amin(dim=1)

    assert [field.name for field in fields(HilbertStepRms)] == ["rms", "tokens"]
    assert result.rms == pytest.approx(
        ranges.square().sum(dtype=torch.float64).div(2).sqrt().item()
    )
    assert result.tokens == 2


def test_hilbert_rms_reduces_only_the_reported_squared_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = torch.tensor([[1.0, 3.0]])
    remote = torch.tensor([[2.0, 2.0], [-4.0, 0.0]])
    accumulator = HilbertRmsAccumulator(torch.device("cpu"))
    accumulator.add(local)
    remote_ranges = remote.amax(dim=1) - remote.amin(dim=1)
    calls: list[tuple[tuple[int, ...], torch.dtype]] = []

    def all_reduce(value: torch.Tensor) -> None:
        calls.append((tuple(value.shape), value.dtype))
        if value.dtype == torch.float64:
            value += remote_ranges.square().sum(dtype=torch.float64)
        else:
            value += remote.shape[0]

    monkeypatch.setattr(diagnostics.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(diagnostics.dist, "all_reduce", all_reduce)

    result = accumulator.compute()
    combined = torch.cat((local, remote))
    ranges = combined.amax(dim=1) - combined.amin(dim=1)

    assert result.rms == pytest.approx(ranges.square().mean().sqrt().item())
    assert result.tokens == 3
    assert calls == [((), torch.float64), ((), torch.int64)]


@pytest.mark.parametrize("step_logits", [torch.ones(3), torch.ones(0, 2), torch.ones(2, 0)])
def test_hilbert_rms_rejects_empty_or_invalid_shapes(
    step_logits: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="non-empty shape"):
        HilbertRmsAccumulator(torch.device("cpu")).add(step_logits)


def test_hilbert_rms_requires_hidden_states() -> None:
    with pytest.raises(ValueError, match="without hidden states"):
        HilbertRmsAccumulator(torch.device("cpu")).compute()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_hilbert_rms_rejects_nonfinite_step_logits(value: float) -> None:
    accumulator = HilbertRmsAccumulator(torch.device("cpu"))
    accumulator.add(torch.tensor([[0.0, value]]))

    with pytest.raises(ValueError, match="require finite step logits"):
        accumulator.compute()

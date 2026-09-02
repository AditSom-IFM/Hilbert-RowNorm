from __future__ import annotations

import pytest
import torch

import hilbert_rownorm.head_geometry as geometry
from hilbert_rownorm.head_geometry import (
    HeadGeometry,
    exact_diameter_update_schedule,
    exact_row_diameter,
)


class Provider:
    def __init__(self, step: torch.Tensor) -> None:
        self.step = step
        self.calls = 0

    def lm_head_step(self) -> torch.Tensor:
        self.calls += 1
        return self.step


def test_exact_diameter_schedule_is_periodic_and_always_includes_final_update() -> None:
    assert exact_diameter_update_schedule(
        23,
        every_log=True,
        log_every=10,
    ) == (10, 20, 23)
    assert exact_diameter_update_schedule(
        23,
        every_log=False,
        log_every=0,
    ) == ()


def test_exact_diameter_schedule_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="total updates must be positive"):
        exact_diameter_update_schedule(0, every_log=False, log_every=0)
    with pytest.raises(ValueError, match="positive log interval"):
        exact_diameter_update_schedule(23, every_log=True, log_every=0)


def test_exact_row_diameter_matches_all_pairs_and_is_translation_invariant() -> None:
    rows = torch.tensor(
        [[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0], [4.0, -1.0, 2.0], [0.0, 0.0, 0.0]]
    )
    expected = torch.pdist(rows).max().item()

    for block_rows in (1, 2, 3, 8):
        assert exact_row_diameter(rows, block_rows=block_rows) == pytest.approx(
            expected
        )
    shifted = rows + torch.tensor([7.0, -4.0, 2.0])
    assert exact_row_diameter(shifted, block_rows=2) == pytest.approx(expected)


def test_exact_row_diameter_uses_paper_numerical_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.dtype, torch.dtype, float, str]] = []
    original = torch.cdist

    def cdist(
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        p: float,
        compute_mode: str,
    ) -> torch.Tensor:
        calls.append((left.dtype, right.dtype, p, compute_mode))
        return original(left, right, p=p, compute_mode=compute_mode)

    monkeypatch.setattr(geometry.torch, "cdist", cdist)

    exact_row_diameter(torch.arange(15, dtype=torch.float64).reshape(5, 3), block_rows=2)

    assert len(calls) == 6
    assert all(
        call == (
            torch.float32,
            torch.float32,
            2,
            "use_mm_for_euclid_dist",
        )
        for call in calls
    )


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        (torch.ones(0, 2), ValueError, "non-empty matrix"),
        (torch.ones(2, 0), ValueError, "non-empty matrix"),
        (torch.ones(2, 2, dtype=torch.int64), TypeError, "floating point"),
        (torch.tensor([[0.0, float("nan")]]), ValueError, "only finite"),
    ],
)
def test_exact_row_diameter_rejects_invalid_matrices(
    matrix: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        exact_row_diameter(matrix, block_rows=2)


def test_head_geometry_exposes_step_and_rank_zero_diameter() -> None:
    step = torch.tensor([[0.0, 0.0], [3.0, 4.0], [-3.0, 4.0]])
    rank_zero_provider = Provider(step)
    rank_zero = HeadGeometry(
        [rank_zero_provider],
        rank=0,
        total_updates=23,
        exact_diameter_every_log=True,
        log_every=10,
        diameter_block_rows=2,
        hilbert_probe_tokens=128,
    )

    assert rank_zero.exact_diameter_due(10)
    assert rank_zero.exact_diameter_due(23)
    assert not rank_zero.exact_diameter_due(22)
    assert rank_zero.optimizer_step(True) is step
    assert rank_zero.exact_diameter(True) == pytest.approx(6.0)
    assert rank_zero_provider.calls == 2

    rank_one_provider = Provider(step)
    rank_one = HeadGeometry(
        [rank_one_provider],
        rank=1,
        total_updates=23,
        exact_diameter_every_log=True,
        log_every=10,
        diameter_block_rows=2,
        hilbert_probe_tokens=0,
    )
    assert rank_one.exact_diameter(True) is None
    assert rank_one_provider.calls == 0


def test_head_geometry_requires_one_provider_only_when_enabled() -> None:
    disabled = HeadGeometry(
        [],
        rank=0,
        total_updates=3,
        exact_diameter_every_log=False,
        log_every=0,
        diameter_block_rows=2,
        hilbert_probe_tokens=0,
    )
    assert disabled.provider is None
    assert disabled.optimizer_step(False) is None

    provider = Provider(torch.ones(2, 2))
    for optimizers in ([], [provider, Provider(torch.ones(2, 2))]):
        with pytest.raises(ValueError, match="exactly one LM-head optimizer"):
            HeadGeometry(
                optimizers,
                rank=0,
                total_updates=3,
                exact_diameter_every_log=False,
                log_every=0,
                diameter_block_rows=2,
                hilbert_probe_tokens=1,
            )

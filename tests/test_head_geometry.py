from __future__ import annotations

import pytest
import torch

import hilbert_rownorm.head_geometry as geometry
from hilbert_rownorm.head_geometry import (
    DIAMETER_ALGORITHM,
    HeadGeometry,
    exact_diameter_update_schedule,
    exact_row_diameter,
)


def direct_diameter(rows: torch.Tensor) -> float:
    """Small FP64 reference using differences, without the cdist formula."""

    values = rows.double()
    differences = values[:, None, :] - values[None, :, :]
    return differences.square().sum(dim=-1).sqrt().max().item()


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
    expected = direct_diameter(rows)

    for block_rows in (1, 2, 3, 8):
        assert exact_row_diameter(rows, block_rows=block_rows) == pytest.approx(
            expected
        )
    shifted = rows + torch.tensor([7.0, -4.0, 2.0])
    assert exact_row_diameter(shifted, block_rows=2) == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exact_row_diameter_versions_translation_and_precision(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
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

    rows = torch.arange(15, dtype=dtype).reshape(5, 3) + 10
    original_rows = rows.clone()
    assert exact_row_diameter(rows, block_rows=2) == pytest.approx(direct_diameter(rows))

    assert DIAMETER_ALGORITHM == "centered_cdist_v2"
    assert torch.equal(rows, original_rows)
    assert len(calls) == 6
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert all(
        call == (
            expected_dtype,
            expected_dtype,
            2,
            "use_mm_for_euclid_dist",
        )
        for call in calls
    )


@pytest.mark.parametrize(
    ("dtype", "offset", "relative_tolerance"),
    [(torch.float32, 10.0, 2e-7), (torch.float64, 1e8, 1e-14)],
)
def test_exact_row_diameter_resolves_small_differences_under_a_common_offset(
    dtype: torch.dtype,
    offset: float,
    relative_tolerance: float,
) -> None:
    rows = torch.tensor([[0, 0], [0.0032, 0], [0, 0.0032]], dtype=dtype) + offset
    expected = direct_diameter(rows)

    assert expected > 0
    for block_rows in (1, 2, 3, 8):
        assert exact_row_diameter(rows, block_rows=block_rows) == pytest.approx(
            expected, rel=relative_tolerance, abs=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("row_count", [1, 7])
def test_exact_row_diameter_is_zero_for_single_or_identical_rows(
    dtype: torch.dtype,
    row_count: int,
) -> None:
    rows = torch.tensor([[10.1, -20.2, 30.3]], dtype=dtype).repeat(row_count, 1)

    for block_rows in (1, 2, 5, 8):
        assert exact_row_diameter(rows, block_rows=block_rows) == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_row_diameter_matches_direct_reference_across_blocks(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(17)
    rows = torch.randn(19, 7, dtype=dtype, generator=generator) + 100
    expected = direct_diameter(rows)
    tolerance = 2e-7 if dtype == torch.float32 else 1e-14
    results = [exact_row_diameter(rows, block_rows=block_rows) for block_rows in (1, 3, 8, 32)]

    assert results == pytest.approx([expected] * len(results), rel=tolerance, abs=0)


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        (torch.ones(0, 2), ValueError, "non-empty matrix"),
        (torch.ones(2, 0), ValueError, "non-empty matrix"),
        (torch.ones(2, 2, dtype=torch.int64), TypeError, "floating point"),
        (torch.tensor([[0.0, float("nan")]]), ValueError, "only finite"),
        (torch.tensor([[0.0], [float("inf")]]), ValueError, "only finite"),
        (torch.tensor([[-float("inf")], [0.0]]), ValueError, "only finite"),
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

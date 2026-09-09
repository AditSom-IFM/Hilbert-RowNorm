from __future__ import annotations

from argparse import Namespace
from dataclasses import FrozenInstanceError

import pytest

from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.run_plan import (
    RunPlan,
    build_run_plan,
    evaluation_batches,
    validate_arguments,
)


def _arguments(**overrides: object) -> Namespace:
    values = {
        "global_batch": 16,
        "micro_batch": 2,
        "tokens_per_parameter": 1,
        "warmup_fraction": 0.5,
        "min_lr_ratio": 0.1,
        "max_gradient_norm": 1.0,
        "max_steps": 3,
        "log_every": 10,
        "eval_tokens": 100,
    }
    values.update(overrides)
    return Namespace(**values)


def test_run_plan_derives_full_schedule_and_truncated_run() -> None:
    config = ModelConfig(
        dim=4,
        n_layers=1,
        head_dim=2,
        vocab_size=4,
        max_seq_len=2,
    )
    args = _arguments()

    validate_arguments(args, world_size=2)
    plan = build_run_plan(args, config, world_size=2)

    assert plan == RunPlan(
        accumulation_steps=4,
        tokens_per_step=32,
        target_tokens=288,
        schedule_steps=9,
        run_steps=3,
        warmup_steps=4,
    )
    assert plan.scheduled_tokens == 288
    with pytest.raises(FrozenInstanceError):
        plan.run_steps = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"global_batch": 0}, "global_batch must be positive"),
        ({"micro_batch": 3}, r"world_size \* micro_batch must divide global_batch"),
        ({"tokens_per_parameter": 0}, "tokens_per_parameter must be positive"),
        ({"warmup_fraction": 1.0}, r"warmup_fraction must be in \[0, 1\)"),
        ({"min_lr_ratio": 1.1}, r"min_lr_ratio must be in \[0, 1\]"),
        ({"max_gradient_norm": float("inf")}, "max_gradient_norm must be finite"),
        ({"eval_tokens": 0}, "eval_tokens must be positive"),
        ({"eval_tokens": -1}, "eval_tokens must be positive"),
    ],
)
def test_run_argument_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_arguments(_arguments(**overrides), world_size=2)


def test_run_plan_rejects_empty_or_nonpositive_runs() -> None:
    config = ModelConfig(dim=2, n_layers=1, head_dim=2, vocab_size=2, max_seq_len=2)

    with pytest.raises(ValueError, match="at least one optimizer update"):
        build_run_plan(_arguments(global_batch=64), config, world_size=1)
    with pytest.raises(ValueError, match="max_steps must be positive"):
        build_run_plan(_arguments(max_steps=0), config, world_size=1)


def test_exact_diameter_every_log_requires_positive_interval() -> None:
    with pytest.raises(ValueError, match="requires --log-every > 0"):
        validate_arguments(
            _arguments(
                exact_head_diameter_every_log=True,
                log_every=0,
            ),
            world_size=2,
        )


@pytest.mark.parametrize(
    ("requested_tokens", "expected"),
    [(1, (1, 32)), (63, (1, 32)), (64, (2, 64))],
)
def test_evaluation_batches_preserve_flooring_with_a_one_batch_minimum(
    requested_tokens: int,
    expected: tuple[int, int],
) -> None:
    assert evaluation_batches(requested_tokens, tokens_per_batch=32) == expected


@pytest.mark.parametrize(
    ("requested_tokens", "tokens_per_batch", "message"),
    [
        (0, 32, "requested evaluation tokens must be positive"),
        (-1, 32, "requested evaluation tokens must be positive"),
        (32, 0, "evaluation tokens per batch must be positive"),
        (32, -1, "evaluation tokens per batch must be positive"),
    ],
)
def test_evaluation_batches_rejects_nonpositive_token_counts(
    requested_tokens: int,
    tokens_per_batch: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluation_batches(requested_tokens, tokens_per_batch)

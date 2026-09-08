"""Validated update and token counts for one training run."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from .config import ModelConfig


@dataclass(frozen=True)
class RunPlan:
    """Immutable counts derived from the model, launch size, and token budget."""

    accumulation_steps: int
    tokens_per_step: int
    target_tokens: int
    schedule_steps: int
    run_steps: int
    warmup_steps: int

    @property
    def scheduled_tokens(self) -> int:
        return self.schedule_steps * self.tokens_per_step


def validate_arguments(args: argparse.Namespace, world_size: int) -> None:
    """Validate run arguments whose meaning does not depend on the model preset."""

    if args.global_batch < 1:
        raise ValueError("global_batch must be positive")
    if args.micro_batch < 1 or args.global_batch % (world_size * args.micro_batch):
        raise ValueError("world_size * micro_batch must divide global_batch")
    if args.tokens_per_parameter < 1:
        raise ValueError("tokens_per_parameter must be positive")
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if not math.isfinite(args.max_gradient_norm) or args.max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be finite and positive")
    if args.eval_tokens < 1:
        raise ValueError("eval_tokens must be positive")
    diameter_every_log = getattr(args, "exact_head_diameter_every_log", False)
    diameter_block_rows = getattr(args, "exact_head_diameter_block_rows", 4_096)
    hilbert_probe_tokens = getattr(args, "hilbert_step_probe_tokens", 0)
    if diameter_block_rows < 1:
        raise ValueError("exact_head_diameter_block_rows must be positive")
    if diameter_every_log and getattr(args, "log_every", 0) < 1:
        raise ValueError("exact_head_diameter_every_log requires --log-every > 0")
    if hilbert_probe_tokens < 0:
        raise ValueError("hilbert_step_probe_tokens must be non-negative")


def build_run_plan(
    args: argparse.Namespace,
    config: ModelConfig,
    world_size: int,
) -> RunPlan:
    """Derive optimizer-update counts after arguments and model selection are valid."""

    accumulation_steps = args.global_batch // (world_size * args.micro_batch)
    tokens_per_step = args.global_batch * config.max_seq_len
    target_tokens = config.parameter_count * args.tokens_per_parameter
    schedule_steps = target_tokens // tokens_per_step
    if schedule_steps < 1:
        raise ValueError("the token budget must contain at least one optimizer update")
    run_steps = (
        schedule_steps if args.max_steps is None else min(schedule_steps, args.max_steps)
    )
    if run_steps < 1:
        raise ValueError("max_steps must be positive")
    warmup_steps = int(schedule_steps * args.warmup_fraction)
    return RunPlan(
        accumulation_steps=accumulation_steps,
        tokens_per_step=tokens_per_step,
        target_tokens=target_tokens,
        schedule_steps=schedule_steps,
        run_steps=run_steps,
        warmup_steps=warmup_steps,
    )


def evaluation_batches(requested_tokens: int, tokens_per_batch: int) -> tuple[int, int]:
    """Floor positive token requests to full batches, with a one-batch minimum."""

    if requested_tokens < 1:
        raise ValueError("requested evaluation tokens must be positive")
    if tokens_per_batch < 1:
        raise ValueError("evaluation tokens per batch must be positive")
    batches = max(1, requested_tokens // tokens_per_batch)
    return batches, batches * tokens_per_batch

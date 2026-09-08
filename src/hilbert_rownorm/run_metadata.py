"""Stable metadata payloads describing one pretraining run."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from typing import Any

import torch
from torch.optim import Optimizer

from .config import ModelConfig
from .data import FineWebBatchLoader
from .head_geometry import DIAMETER_ALGORITHM
from .run_plan import RunPlan
from .tracking import optimizer_configuration

TRACKED_ARGUMENTS = (
    "model",
    "optimizer",
    "lm_head_optimizer",
    "exact_head_diameter_every_log",
    "exact_head_diameter_block_rows",
    "hilbert_step_probe_tokens",
    "micro_batch",
    "global_batch",
    "tokens_per_parameter",
    "seed",
    "max_steps",
    "warmup_fraction",
    "min_lr_ratio",
    "max_gradient_norm",
    "no_gradient_clipping",
    "log_every",
    "eval_every",
    "eval_tokens",
    "save_every",
    "gradient_checkpointing",
    "require_cuda",
)


def precision_metadata(compute_dtype: torch.dtype | None) -> dict[str, Any]:
    """Describe arithmetic, not the dtype of a tensor after an explicit cast."""

    compute = "FP32" if compute_dtype is None else str(compute_dtype).removeprefix("torch.")
    if compute_dtype == torch.bfloat16:
        compute = "BF16"
    return {
        "precision": f"FP32 params / {compute} compute / FP32 loss",
        "precision_details": {
            "parameters": "float32",
            "optimizer_states": "float32",
            "autocast": None
            if compute_dtype is None
            else str(compute_dtype).removeprefix("torch."),
            "logit_matmul": compute,
            "logits_after_cast": "float32",
            "cross_entropy": "float32",
            "hilbert_matmul": compute,
            "hilbert_range": "float32",
            "hilbert_sum_of_squares": "float64",
        },
    }


def build_run_config(
    args: Namespace,
    config: ModelConfig,
    optimizer_recipe_id: str,
    optimizer_config: dict[str, Any],
    optimizers: list[Optimizer],
    train_loader: FineWebBatchLoader,
    val_loader: FineWebBatchLoader,
    world_size: int,
    plan: RunPlan,
    actual_val_tokens: int,
    device_name: str,
    *,
    compute_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Build the exact tracking configuration used to describe a run."""

    return {
        **{name: getattr(args, name) for name in TRACKED_ARGUMENTS},
        "dataset": "FineWeb-100B (GPT-2 tokenizer)",
        "optimizer_recipe_id": optimizer_recipe_id,
        "optimizer_config_values": optimizer_config,
        "model_config": asdict(config),
        "parameter_count": config.parameter_count,
        "optimizer_groups": optimizer_configuration(optimizers),
        "train_shards": len(train_loader.shards),
        "validation_shards": len(val_loader.shards),
        "world_size": world_size,
        "gradient_accumulation_steps": plan.accumulation_steps,
        "tokens_per_update": plan.tokens_per_step,
        "target_tokens": plan.target_tokens,
        "scheduled_tokens": plan.scheduled_tokens,
        "schedule_steps": plan.schedule_steps,
        "run_steps": plan.run_steps,
        "schedule": "linear_warmup_cosine_decay",
        "warmup_steps": plan.warmup_steps,
        "max_gradient_norm": args.max_gradient_norm,
        "gradient_clipping": not args.no_gradient_clipping,
        "actual_eval_tokens": actual_val_tokens,
        **precision_metadata(compute_dtype),
        "diameter_algorithm": DIAMETER_ALGORITHM,
        "hilbert_panel_selection": "first_tokens_per_rank_v1",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": device_name,
    }

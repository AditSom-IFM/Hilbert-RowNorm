"""Linear orchestration for one pretraining run."""

from __future__ import annotations

import time
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from .checkpointing import save_checkpoint
from .config import ModelConfig, model_preset
from .data import FineWebBatchLoader
from .head_geometry import HeadGeometry
from .model import TransformerLM
from .optimizer_factory import build_optimizers
from .optimizer_recipes import resolve_optimizer_recipe
from .run_metadata import build_run_config
from .run_plan import RunPlan, build_run_plan, evaluation_batches, validate_arguments
from .tracking import Tracker
from .training import evaluate, schedule_scale, train_step


@dataclass(frozen=True)
class PreparedRun:
    """Objects assembled before tracking and optimizer updates begin."""

    args: Namespace
    rank: int
    world_size: int
    device: torch.device
    config: ModelConfig
    plan: RunPlan
    model: TransformerLM
    train_model: nn.Module
    optimizers: list[Optimizer]
    head_geometry: HeadGeometry
    train_loader: FineWebBatchLoader
    val_loader: FineWebBatchLoader
    val_batches: int
    actual_val_tokens: int
    device_name: str
    run_config: dict[str, Any]


def prepare_run(
    args: Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> PreparedRun:
    """Build run resources in their original deterministic initialization order."""

    validate_arguments(args, world_size)
    config = model_preset(args.model)
    plan = build_run_plan(args, config, world_size)

    torch.manual_seed(args.seed + rank)
    model = TransformerLM(
        config,
        device=device,
        compute_dtype=torch.bfloat16 if device.type == "cuda" else None,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    optimizer_recipe = resolve_optimizer_recipe(
        args.model, args.optimizer, args.lm_head_optimizer
    )
    optimizers = build_optimizers(model, args.optimizer, optimizer_recipe.config)
    head_geometry = HeadGeometry(
        optimizers,
        rank=rank,
        total_updates=plan.run_steps,
        exact_diameter_every_log=getattr(args, "exact_head_diameter_every_log", False),
        log_every=getattr(args, "log_every", 0),
        diameter_block_rows=getattr(args, "exact_head_diameter_block_rows", 4_096),
        hilbert_probe_tokens=getattr(args, "hilbert_step_probe_tokens", 0),
    )
    train_model: nn.Module = model
    if world_size > 1:
        train_model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    loader_options = dict(
        batch_size=args.micro_batch,
        sequence_length=config.max_seq_len,
        rank=rank,
        world_size=world_size,
    )
    train_loader = FineWebBatchLoader(
        args.data,
        "fineweb_train_*.bin",
        gradient_accumulation_steps=plan.accumulation_steps,
        seed=args.seed,
        **loader_options,
    )
    val_loader = FineWebBatchLoader(
        args.data, "fineweb_val_*.bin", shuffle=False, **loader_options
    )
    val_batches, actual_val_tokens = evaluation_batches(
        args.eval_tokens, val_loader.global_tokens_per_batch
    )
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(
        args,
        config,
        optimizer_recipe.identifier,
        optimizer_recipe.config,
        optimizers,
        train_loader,
        val_loader,
        world_size,
        plan,
        actual_val_tokens,
        device_name,
    )
    return PreparedRun(
        args=args,
        rank=rank,
        world_size=world_size,
        device=device,
        config=config,
        plan=plan,
        model=model,
        train_model=train_model,
        optimizers=optimizers,
        head_geometry=head_geometry,
        train_loader=train_loader,
        val_loader=val_loader,
        val_batches=val_batches,
        actual_val_tokens=actual_val_tokens,
        device_name=device_name,
        run_config=run_config,
    )


def start_tracker(run: PreparedRun) -> Tracker:
    """Start tracking after every run resource and metadata payload is ready."""

    args = run.args
    return Tracker(
        rank=run.rank,
        device=run.device,
        optimizers=run.optimizers,
        parameter_count=run.config.parameter_count,
        tokens_per_step=run.plan.tokens_per_step,
        validation_tokens=run.actual_val_tokens,
        max_gradient_norm=args.max_gradient_norm,
        gradient_clipping=not args.no_gradient_clipping,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.output.name,
        group=f"{args.model}-{args.optimizer}",
        mode=args.wandb_mode,
        directory=args.output,
        config=run.run_config,
    )


def print_run_summary(run: PreparedRun) -> None:
    """Print the unchanged rank-zero run summary."""

    if run.rank == 0:
        args = run.args
        print(
            f"model={args.model} parameters={run.config.parameter_count:,} "
            f"optimizer={args.optimizer} schedule_steps={run.plan.schedule_steps:,} "
            f"run_steps={run.plan.run_steps:,} global_batch={args.global_batch:,} "
            f"tokens_per_step={run.plan.tokens_per_step:,} "
            f"accumulation={run.plan.accumulation_steps} "
            f"device={run.device} device_name={run.device_name!r} "
            f"world_size={run.world_size} output={args.output}"
        )


def execute_updates(run: PreparedRun, tracker: Tracker) -> None:
    """Execute optimizer updates, validation, diagnostics, and checkpoints."""

    args = run.args
    plan = run.plan
    autocast_dtype = torch.bfloat16 if run.device.type == "cuda" else None
    for step in range(plan.run_steps):
        update = step + 1
        should_log = update == plan.run_steps or (
            args.log_every > 0 and update % args.log_every == 0
        )
        scale = schedule_scale(
            step, plan.schedule_steps, plan.warmup_steps, args.min_lr_ratio
        )
        loss, grad_norm = train_step(
            run.train_model,
            run.optimizers,
            run.train_loader,
            accumulation_steps=plan.accumulation_steps,
            lr_scale=scale,
            max_gradient_norm=args.max_gradient_norm,
            device=run.device,
            autocast_dtype=autocast_dtype,
            gradient_clipping=not args.no_gradient_clipping,
        )
        tracker.add(loss, grad_norm)
        exact_diameter_due = run.head_geometry.exact_diameter_due(update)
        head_diameter = run.head_geometry.exact_diameter(
            exact_diameter_due and tracker.active
        )

        should_evaluate = args.eval_every > 0 and (
            update % args.eval_every == 0 or update == plan.run_steps
        )
        val_loss = validation_hilbert = evaluation_seconds = None
        if should_evaluate:
            hilbert_probe_tokens = getattr(args, "hilbert_step_probe_tokens", 0)
            head_step = run.head_geometry.optimizer_step(
                hilbert_probe_tokens > 0
            )
            then = time.perf_counter()
            val_loss, validation_hilbert = evaluate(
                run.train_model,
                run.val_loader,
                run.val_batches,
                run.device,
                autocast_dtype,
                head_step=head_step,
                hilbert_probe_tokens=hilbert_probe_tokens,
            )
            evaluation_seconds = time.perf_counter() - then

        checkpoint_path = None
        checkpoint_seconds = None
        if args.save_every > 0 and (
            update % args.save_every == 0 or update == plan.run_steps
        ):
            checkpoint_path = args.output / (
                "final.pt" if update == plan.run_steps else "latest.pt"
            )
            then = time.perf_counter()
            save_checkpoint(
                checkpoint_path,
                run.model,
                run.optimizers,
                update,
                args,
                run.rank,
            )
            checkpoint_seconds = time.perf_counter() - then

        tracker.report(
            update,
            scale,
            log_train=should_log,
            head_diameter=head_diameter,
            validation_loss=val_loss,
            validation_hilbert=validation_hilbert,
            validation_seconds=evaluation_seconds,
            checkpoint_path=checkpoint_path,
            checkpoint_seconds=checkpoint_seconds,
        )


def run_pretraining(
    args: Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Prepare, execute, and tear down one pretraining run."""

    tracker: Tracker | None = None
    final_updates = 0
    exit_code = 1
    try:
        run = prepare_run(args, rank, world_size, device)
        tracker = start_tracker(run)
        print_run_summary(run)
        execute_updates(run, tracker)
        final_updates = run.plan.run_steps
        exit_code = 0
    finally:
        if tracker is not None:
            tracker.finish(final_updates, exit_code)
        if dist.is_initialized():
            dist.destroy_process_group()

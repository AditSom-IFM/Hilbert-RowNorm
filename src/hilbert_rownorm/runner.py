"""Linear orchestration for one pretraining run."""

from __future__ import annotations

import json
import logging
import os
import time
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from .checkpointing import (
    ResumePoint,
    load_checkpoint,
    restore_checkpoint,
    resume_fingerprint,
    save_checkpoint,
    select_resume_checkpoint,
)
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
    fingerprint: dict[str, Any] | None = None
    resume: ResumePoint | None = None
    already_complete: bool = False

    @property
    def start_update(self) -> int:
        return 0 if self.resume is None else self.resume.update


# Files a run writes into its output directory, moved aside when a resumed
# run finds no checkpoint and has to start over.
RUN_RECORDS = (
    "metrics.jsonl",
    "config.json",
    "summary.json",
    "resumes.jsonl",
    "wandb",
)


def archive_unfinished_attempt(output: Path) -> Path | None:
    """Move an earlier attempt's records aside before starting over in ``output``."""

    present = [output / name for name in RUN_RECORDS if (output / name).exists()]
    present += sorted(output.glob("metrics.discarded-*.jsonl"))
    present += sorted(output.glob("summary.before-resume-*.json"))
    if not present:
        return None
    index = 1
    while (output / "abandoned" / f"attempt-{index}").exists():
        index += 1
    archive = output / "abandoned" / f"attempt-{index}"
    archive.mkdir(parents=True)
    for path in present:
        os.replace(path, archive / path.name)
    return archive


def _completed_summary(output: Path) -> bool:
    try:
        return json.loads((output / "summary.json").read_text()).get("exit_code") == 0
    except (OSError, ValueError, AttributeError):
        return False


def prepare_output_directory(
    output: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    *,
    resume: bool,
) -> tuple[Path | None, bool]:
    """Reserve the run directory and, when resuming, choose the checkpoint.

    Rank zero does the file-system work; every rank receives the chosen
    checkpoint and whether a finished summary is already present, or the
    same error.
    """

    if not resume:
        create_output_directory(output, rank, world_size, device)
        return None, False
    message: list[Any] = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=True)
            selected = select_resume_checkpoint(output)
            if selected is None:
                archive = archive_unfinished_attempt(output)
                if archive is not None:
                    print(
                        f"no checkpoint in {output}; moved earlier records to {archive}",
                        flush=True,
                    )
            message[0] = {
                "checkpoint": None if selected is None else str(selected),
                "complete": selected is not None and _completed_summary(output),
            }
        except Exception as error:  # noqa: BLE001 - every rank must see the failure
            message[0] = {"error": f"cannot resume in {output}: {type(error).__name__}: {error}"}
    if world_size > 1:
        dist.broadcast_object_list(message, src=0, device=device)
    result = message[0]
    if "error" in result:
        raise RuntimeError(result["error"])
    checkpoint = result["checkpoint"]
    return (None if checkpoint is None else Path(checkpoint)), result["complete"]


def create_output_directory(
    output: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Reserve a fresh run directory and share rank-zero failures with peers."""

    failure: list[str | None] = [None]
    creation_error: OSError | None = None
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            creation_error = error
            failure[0] = (
                f"cannot create fresh output directory {output}: {error}; "
                "choose an unused, writable --output directory"
            )
    if world_size > 1:
        dist.broadcast_object_list(failure, src=0, device=device)
    if failure[0] is not None:
        raise RuntimeError(failure[0]) from creation_error


def head_overrides(args: Namespace) -> dict[str, object]:
    """Collect LM-head sweep values given on the command line."""

    names = {
        "lr": "lm_head_lr",
        "betas": "lm_head_betas",
        "beta": "lm_head_beta",
        "weight_decay": "lm_head_weight_decay",
    }
    values = {key: getattr(args, name, None) for key, name in names.items()}
    return {key: value for key, value in values.items() if value is not None}


def attention_qk_overrides(args: Namespace) -> dict[str, object]:
    """Collect query/key optimizer values given on the command line."""

    names = {
        "lr": "qk_lr",
        "momentum": "qk_momentum",
        "weight_decay": "qk_weight_decay",
        "mp": "cm_mp",
        "whitening": "cm_whitening",
    }
    values = {key: getattr(args, name, None) for key, name in names.items()}
    return {key: value for key, value in values.items() if value is not None}


def prepare_run(
    args: Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> PreparedRun:
    """Build run resources in their original deterministic initialization order."""

    validate_arguments(args, world_size)
    config = model_preset(args.model)
    if getattr(args, "no_qk_norm", False):
        config = replace(config, qk_norm=False)
    plan = build_run_plan(args, config, world_size)
    checkpoint, complete = prepare_output_directory(
        args.output, rank, world_size, device, resume=getattr(args, "resume", False)
    )

    torch.manual_seed(args.seed + rank)
    model = TransformerLM(
        config,
        device=device,
        compute_dtype=torch.bfloat16 if device.type == "cuda" else None,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    optimizer_recipe = resolve_optimizer_recipe(
        args.model,
        args.optimizer,
        args.lm_head_optimizer,
        head_overrides=head_overrides(args),
        attention_qk=getattr(args, "attention_qk_optimizer", "muon"),
        attention_qk_overrides=attention_qk_overrides(args),
        qk_norm=not getattr(args, "no_qk_norm", False),
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
        compute_dtype=getattr(model, "compute_dtype", None),
    )
    fingerprint = resume_fingerprint(run_config, train_loader, val_loader)
    resume = None
    if checkpoint is not None:
        resume = _resume(
            checkpoint,
            model=model,
            optimizers=optimizers,
            train_loader=train_loader,
            fingerprint=fingerprint,
            plan=plan,
            rank=rank,
            world_size=world_size,
            device=device,
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
        fingerprint=fingerprint,
        resume=resume,
        already_complete=complete and resume is not None and resume.update == plan.run_steps,
    )


def _resume(
    path: Path,
    *,
    model: TransformerLM,
    optimizers: list[Optimizer],
    train_loader: FineWebBatchLoader,
    fingerprint: dict[str, Any],
    plan: RunPlan,
    rank: int,
    world_size: int,
    device: torch.device,
) -> ResumePoint:
    """Restore a checkpoint on every rank; fail on all ranks if any rank fails."""

    error: Exception | None = None
    point = None
    try:
        payload = load_checkpoint(path)
        point = restore_checkpoint(
            payload,
            path=path,
            model=model,
            optimizers=optimizers,
            train_loader=train_loader,
            fingerprint=fingerprint,
            run_steps=plan.run_steps,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        del payload
    except Exception as exception:  # noqa: BLE001 - re-raised below on every rank
        error = exception
    if world_size > 1 and dist.is_initialized():
        failed = torch.tensor(int(error is not None), device=device)
        dist.all_reduce(failed)
        if failed.item() and error is None:
            raise RuntimeError(f"another rank failed to restore {path}")
    if error is not None:
        raise error
    assert point is not None
    return point


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
        resume=run.resume,
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
    for step in range(run.start_update, plan.run_steps):
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
        head_diameter = run.head_geometry.exact_diameter(exact_diameter_due)

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

        # Record this update before checkpointing it, so a resumed run's
        # local metrics never miss an update that the checkpoint covers.
        tracker.report(
            update,
            scale,
            log_train=should_log,
            head_diameter=head_diameter,
            validation_loss=val_loss,
            validation_hilbert=validation_hilbert,
            validation_seconds=evaluation_seconds,
        )

        if args.save_every > 0 and (
            update % args.save_every == 0 or update == plan.run_steps
        ):
            then = time.perf_counter()
            checkpoint_path = save_checkpoint(
                args.output,
                update=update,
                model=run.model,
                optimizers=run.optimizers,
                train_loader=run.train_loader,
                fingerprint=run.fingerprint or {},
                tracker_state=tracker.checkpoint_state(),
                args=args,
                keep=getattr(args, "keep_checkpoints", 2),
                rank=run.rank,
                world_size=run.world_size,
                device=run.device,
            )
            checkpoint_seconds = time.perf_counter() - then
            tracker.report(
                update,
                scale,
                log_train=False,
                checkpoint_path=checkpoint_path,
                checkpoint_seconds=None if checkpoint_path is None else checkpoint_seconds,
                checkpoint_failed=checkpoint_path is None,
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
        if run.already_complete:
            if rank == 0:
                print(f"run in {args.output} is already complete; nothing to do", flush=True)
            exit_code = 0
            return
        tracker = start_tracker(run)
        print_run_summary(run)
        execute_updates(run, tracker)
        final_updates = run.plan.run_steps
        exit_code = 0
    finally:
        try:
            if tracker is not None:
                try:
                    tracker.finish(final_updates, exit_code)
                except Exception:
                    if exit_code == 0:
                        raise
                    logging.getLogger(__name__).warning(
                        "Tracking shutdown failed while handling the run failure",
                        exc_info=True,
                    )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

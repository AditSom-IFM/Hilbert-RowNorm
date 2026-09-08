"""Interval metrics saved locally on rank zero, with optional W&B logging."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer

from .hilbert_diagnostics import HilbertStepRms
from .tracking_records import (
    add_checkpoint_metrics,
    add_head_diameter,
    add_training_metrics,
    add_validation_metrics,
    build_final_summary,
)
from .training_metrics import TrainingIntervalAccumulator

INHERITED_RUN_VARIABLES = (
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_RESUME_FROM",
    "WANDB_FORK_FROM",
    "WANDB_SWEEP_ID",
    "WANDB_LAUNCH",
    "WANDB_LAUNCH_CONFIG_PATH",
)


def fresh_wandb_identity(project: str) -> dict[str, str]:
    """Prevent fresh training from resuming a run or modifying the paper mirror."""

    if project == "Hilbert-RowNorm":
        raise ValueError(
            "Hilbert-RowNorm is the curated paper project; use "
            "Hilbert-RowNorm-training or another training project"
        )
    inherited = [name for name in INHERITED_RUN_VARIABLES if os.environ.get(name)]
    if inherited:
        raise ValueError("fresh training requires unsetting " + ", ".join(inherited))
    return {"id": uuid4().hex, "resume": "never"}


def optimizer_configuration(optimizers: list[Optimizer]) -> list[dict[str, Any]]:
    """Serialize the few optimizer values needed to reproduce a run."""

    result = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            values = {
                "optimizer": type(optimizer).__name__,
                "role": group["role"],
                "peak_lr": float(group["base_lr"]),
                "weight_decay": float(group["weight_decay"]),
            }
            for key in (
                "betas",
                "beta",
                "bias_correction",
                "eps",
                "momentum",
                "nesterov",
                "ns_steps",
            ):
                if key in group:
                    value = group[key]
                    values[key] = list(value) if isinstance(value, tuple) else value
            result.append(values)
    return result


def _role_lrs(optimizers: list[Optimizer]) -> tuple[dict[str, float], dict[str, float]]:
    current: dict[str, float] = {}
    peak: dict[str, float] = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            role = group.get("role")
            if not isinstance(role, str) or role in current:
                raise ValueError("optimizer groups need unique semantic roles")
            current[role] = float(group["lr"])
            peak[role] = float(group["base_lr"])
    return current, peak


class Tracker:
    """Aggregate without per-step host sync; write local records and optional W&B."""

    def __init__(
        self,
        *,
        rank: int,
        device: torch.device,
        optimizers: list[Optimizer],
        parameter_count: int,
        tokens_per_step: int,
        validation_tokens: int,
        max_gradient_norm: float,
        gradient_clipping: bool,
        project: str,
        entity: str | None,
        name: str,
        group: str,
        mode: str,
        directory: Path,
        config: dict[str, Any],
    ) -> None:
        self.rank = rank
        self.device = device
        self.optimizers = optimizers
        self.parameter_count = parameter_count
        self.tokens_per_step = tokens_per_step
        self.validation_tokens = validation_tokens
        self._training_metrics = TrainingIntervalAccumulator(
            device=device,
            max_gradient_norm=max_gradient_norm,
            gradient_clipping=gradient_clipping,
        )
        self.started = self.last_log_time = time.perf_counter()
        self.last_log_update = 0
        self.last_validation_loss: float | None = None
        self.last_validation_hilbert: HilbertStepRms | None = None
        self.run = None
        self.directory = directory
        self._metrics_file: TextIO | None = None

        error = None
        if rank == 0:
            try:
                self._metrics_file = (directory / "metrics.jsonl").open("x")
                with (directory / "config.json").open("x") as config_file:
                    json.dump(config, config_file, indent=2, allow_nan=False)
                    config_file.write("\n")
                if mode != "disabled":
                    self._start_wandb(
                        project=project,
                        entity=entity,
                        name=name,
                        group=group,
                        mode=mode,
                        directory=directory,
                        config=config,
                    )
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
                self._finish_wandb_after_error()
                self._close_metrics()
        if dist.is_initialized():
            failed = torch.tensor(int(error is not None), device=device)
            dist.broadcast(failed, src=0)
            if failed.item():
                raise RuntimeError(error or "tracking initialization failed on rank zero")
        elif error:
            raise RuntimeError(error)
        self.started = self.last_log_time = time.perf_counter()

    def _start_wandb(
        self,
        *,
        project: str,
        entity: str | None,
        name: str,
        group: str,
        mode: str,
        directory: Path,
        config: dict[str, Any],
    ) -> None:
        identity = fresh_wandb_identity(project)
        import wandb

        self.run = wandb.init(
            **identity,
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type="train",
            mode=mode,
            dir=str(directory),
            config=config,
            force=mode == "online",
            settings=wandb.Settings(init_timeout=60, disable_git=True, disable_code=True),
        )
        if self.run is None:
            raise RuntimeError("wandb.init() did not return a run")
        self.run.define_metric("progress/tokens")
        self.run.define_metric("progress/update")
        for prefix in ("train", "validation", "optimizer", "performance", "memory", "events"):
            self.run.define_metric(f"{prefix}/*", step_metric="progress/tokens")
        self.run.define_metric("diagnostics/*", step_metric="progress/update")
        self.run.define_metric("validation/loss", step_metric="progress/tokens", summary="min")

    def _finish_wandb_after_error(self) -> None:
        """Mark a started run failed without replacing the primary exception."""

        if self.run is not None:
            try:
                self.run.finish(exit_code=1)
            except Exception:
                logging.getLogger(__name__).warning(
                    "W&B shutdown failed while handling a tracking error", exc_info=True
                )

    def _close_metrics(self) -> None:
        """Close the sink, preserving any exception already being handled."""

        error_in_flight = sys.exc_info()[0] is not None
        if self._metrics_file is not None and not self._metrics_file.closed:
            try:
                self._metrics_file.close()
            except Exception:
                if not error_in_flight:
                    raise
                logging.getLogger(__name__).warning(
                    "Local metrics close failed while handling a tracking error", exc_info=True
                )

    @property
    def active(self) -> bool:
        """Whether this rank owns an active W&B run."""

        return self.run is not None

    def add(self, loss: Tensor, grad_norm: Tensor) -> None:
        self._training_metrics.add(loss, grad_norm)

    def report(
        self,
        update: int,
        schedule_scale: float,
        *,
        log_train: bool,
        head_diameter: float | None = None,
        validation_loss: float | None = None,
        validation_hilbert: HilbertStepRms | None = None,
        validation_seconds: float | None = None,
        checkpoint_path: Path | None = None,
        checkpoint_seconds: float | None = None,
    ) -> None:
        if (
            not log_train
            and head_diameter is None
            and validation_loss is None
            and checkpoint_path is None
        ):
            return
        train = self._training_metrics.collect() if log_train else None
        if self.rank != 0:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        tokens = update * self.tokens_per_step
        current_lrs, peak_lrs = _role_lrs(self.optimizers)
        record: dict[str, Any] = {
            "progress/update": update,
            "progress/tokens": tokens,
            "progress/tpp": tokens / self.parameter_count,
            "optimizer/schedule_scale": schedule_scale,
        }
        for role in current_lrs:
            record[f"optimizer/lr_{role}"] = current_lrs[role]
            record[f"optimizer/peak_lr_{role}"] = peak_lrs[role]

        if head_diameter is not None:
            add_head_diameter(record, head_diameter)

        if train is not None:
            seconds = now - self.last_log_time
            add_training_metrics(
                record,
                train,
                update=update,
                previous_update=self.last_log_update,
                tokens_per_step=self.tokens_per_step,
                interval_seconds=seconds,
                elapsed_seconds=now - self.started,
            )
            print(
                f"step={update} tokens={tokens} loss={train.loss:.5f} "
                f"grad_norm={train.grad_norm:.3f} lr={current_lrs['backbone']:.3e} "
                f"interval_tok/s={record['performance/tokens_per_second']:,.0f} "
                f"peak_allocated_gib={train.peak_allocated_gib:.2f} "
                f"peak_reserved_gib={train.peak_reserved_gib:.2f}",
                flush=True,
            )
            self.last_log_time, self.last_log_update = now, update

        if validation_loss is not None and validation_seconds is not None:
            self.last_validation_loss = validation_loss
            add_validation_metrics(
                record,
                loss=validation_loss,
                tokens=self.validation_tokens,
                seconds=validation_seconds,
                hilbert=validation_hilbert,
            )
            if validation_hilbert is not None:
                self.last_validation_hilbert = validation_hilbert
            print(f"step={update} val_loss={validation_loss:.5f}", flush=True)

        if checkpoint_path is not None and checkpoint_seconds is not None:
            add_checkpoint_metrics(record, checkpoint_seconds)
            if self.run is not None:
                self.run.summary.update(
                    {
                        "checkpoint_file": checkpoint_path.name,
                        "checkpoint_update": update,
                    }
                )
        assert self._metrics_file is not None
        self._metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
        self._metrics_file.flush()
        if self.run is not None:
            self.run.log(record)

    def finish(self, updates: int, exit_code: int) -> None:
        if self.rank != 0:
            return
        try:
            summary: dict[str, Any] = {"exit_code": exit_code}
            if exit_code == 0:
                final = build_final_summary(
                    updates=updates,
                    tokens_per_step=self.tokens_per_step,
                    parameter_count=self.parameter_count,
                    elapsed_seconds=time.perf_counter() - self.started,
                    validation_loss=self.last_validation_loss,
                    validation_hilbert=self.last_validation_hilbert,
                )
                summary.update(final)
                if self.run is not None:
                    self.run.summary.update(final)
            self._close_metrics()
            with (self.directory / "summary.json").open("x") as summary_file:
                json.dump(summary, summary_file, indent=2, allow_nan=False)
                summary_file.write("\n")
        except Exception:
            self._finish_wandb_after_error()
            raise
        else:
            if self.run is not None:
                self.run.finish(exit_code=exit_code)
        finally:
            self._close_metrics()

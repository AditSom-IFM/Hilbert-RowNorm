"""Resumable training checkpoints.

A checkpoint holds everything the next optimizer update depends on: the
model weights, every optimizer's state, the training-data position, each
rank's random-number state, and the tracker values that continue across
attempts. Files are written atomically under ``<output>/checkpoints`` and
loaded with ``torch.load(weights_only=True)``.

A run resumes only when its configuration matches the checkpoint's, apart
from the settings in ``RESUMABLE_CHANGES``, which do not change the
arithmetic of an update.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from .data import FineWebBatchLoader

CHECKPOINT_FORMAT = "hilbert-rownorm-training-checkpoint"
CHECKPOINT_VERSION = 1
CHECKPOINT_DIRECTORY = "checkpoints"
_CHECKPOINT_NAME = re.compile(r"^update-(\d{8,})\.pt$")
_TEMPORARY_NAME = re.compile(r"^\.update-\d{8,}\.pt\.tmp-\d+$")

# Run-configuration keys that may change between a checkpoint and the run
# resuming it. Batch splitting, logging, diagnostics, run length, and
# software versions do not change which examples or hyperparameters an
# update uses.
RESUMABLE_CHANGES = frozenset(
    {
        "micro_batch",
        "world_size",
        "gradient_accumulation_steps",
        "max_steps",
        "run_steps",
        "log_every",
        "eval_every",
        "save_every",
        "keep_checkpoints",
        "resume",
        "gradient_checkpointing",
        "require_cuda",
        "exact_head_diameter_every_log",
        "exact_head_diameter_block_rows",
        "hilbert_step_probe_tokens",
        "torch_version",
        "cuda_version",
        "device_name",
    }
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumePoint:
    """The checkpoint a run continues from, after it has been applied."""

    path: Path
    update: int
    tracker: dict[str, Any]


def checkpoint_directory(output: Path) -> Path:
    return output / CHECKPOINT_DIRECTORY


def checkpoint_path(output: Path, update: int) -> Path:
    return checkpoint_directory(output) / f"update-{update:08d}.pt"


def list_checkpoints(output: Path) -> list[tuple[int, Path]]:
    """Return ``(update, path)`` for every checkpoint file, newest first."""

    directory = checkpoint_directory(output)
    if not directory.is_dir():
        return []
    found = []
    for path in directory.iterdir():
        match = _CHECKPOINT_NAME.match(path.name)
        if match and path.is_file():
            found.append((int(match.group(1)), path))
    return sorted(found, reverse=True)


def _json_value(value: Any) -> Any:
    """Normalize a configuration value the way ``config.json`` stores it."""

    return json.loads(json.dumps(value, default=str))


def resume_fingerprint(
    run_config: Mapping[str, Any],
    train_loader: FineWebBatchLoader,
    val_loader: FineWebBatchLoader,
) -> dict[str, Any]:
    """Collect the configuration that must match for a run to resume."""

    fingerprint = {
        key: value for key, value in run_config.items() if key not in RESUMABLE_CHANGES
    }
    fingerprint["train_data_identity"] = train_loader.identity()
    fingerprint["validation_data_identity"] = val_loader.identity()
    return _json_value(fingerprint)


def fingerprint_differences(saved: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Describe each setting whose saved and current values differ."""

    differences = []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key, "<missing>") != current.get(key, "<missing>"):
            differences.append(
                f"{key}: checkpoint={saved.get(key, '<missing>')!r}, "
                f"run={current.get(key, '<missing>')!r}"
            )
    return differences


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Return this process's Python, NumPy, and torch random-number state."""

    kind, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "kind": kind,
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    version, internal, gaussian = state["python"]
    random.setstate((version, tuple(internal), gaussian))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["kind"],
            numpy_state["keys"].numpy().astype(np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def _all_gather(value: Any, world_size: int) -> list[Any]:
    if world_size == 1 or not dist.is_initialized():
        return [value]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _broadcast(value: Any, rank: int, world_size: int, device: torch.device) -> Any:
    """Send rank zero's value to every rank."""

    if world_size == 1 or not dist.is_initialized():
        return value
    message = [value if rank == 0 else None]
    dist.broadcast_object_list(message, src=0, device=device)
    return message[0]


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomically(payload: dict[str, Any], path: Path) -> None:
    """Write ``payload`` so ``path`` is either absent or a complete file."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prune_checkpoints(output: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` checkpoints and return what was deleted."""

    if keep < 1:
        raise ValueError("keep must be positive")
    removed = []
    for _, path in list_checkpoints(output)[keep:]:
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def _json_arguments(args: Namespace) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in sorted(vars(args).items())}


def save_checkpoint(
    output: Path,
    *,
    update: int,
    model: nn.Module,
    optimizers: Sequence[Optimizer],
    train_loader: FineWebBatchLoader,
    fingerprint: Mapping[str, Any],
    tracker_state: Mapping[str, Any],
    args: Namespace,
    keep: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Path | None:
    """Write one checkpoint from rank zero; every rank returns the same result.

    Every rank must call this at the same update. Rank zero writes the file
    and keeps the newest ``keep`` checkpoints. A write failure, for example
    a full disk, is logged and returns ``None`` so training can continue; the
    previous checkpoints stay untouched.
    """

    local = {"loader": train_loader.state_dict(), "rng": capture_rng_state(device)}
    gathered = _all_gather(local, world_size)
    loader_states = [entry["loader"] for entry in gathered]
    if any(state != loader_states[0] for state in loader_states):
        raise RuntimeError(f"training-data positions differ across ranks: {loader_states}")

    result: dict[str, str | None] = {"path": None, "error": None}
    if rank == 0:
        path = checkpoint_path(output, update)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "format": CHECKPOINT_FORMAT,
                "version": CHECKPOINT_VERSION,
                "update": update,
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "world_size": world_size,
                "fingerprint": dict(fingerprint),
                "model": model.state_dict(),
                "optimizers": [
                    {"type": type(optimizer).__name__, "state": optimizer.state_dict()}
                    for optimizer in optimizers
                ],
                "train_loader": loader_states[0],
                "rng": [entry["rng"] for entry in gathered],
                "tracker": dict(tracker_state),
                "args": _json_arguments(args),
            }
            write_atomically(payload, path)
            result["path"] = str(path)
        except Exception as error:  # noqa: BLE001 - reported to every rank below
            result["error"] = f"{type(error).__name__}: {error}"
        else:
            try:
                prune_checkpoints(output, keep)
            except OSError as error:
                _LOGGER.warning("could not delete old checkpoints: %s", error)
    result = _broadcast(result, rank, world_size, device)
    if result["error"] is not None:
        if rank == 0:
            _LOGGER.warning(
                "checkpoint at update %d was not saved; training continues: %s",
                update,
                result["error"],
            )
        return None
    return Path(result["path"])


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read a checkpoint lazily and check that it is a training checkpoint."""

    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a Hilbert-RowNorm training checkpoint")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} has checkpoint version {payload.get('version')!r}; "
            f"this code reads version {CHECKPOINT_VERSION}"
        )
    return payload


def select_resume_checkpoint(output: Path) -> Path | None:
    """Return the newest readable checkpoint, setting unreadable ones aside.

    Runs on rank zero only. An unreadable file, such as one cut short by a
    crash outside the atomic write, is renamed with a ``.corrupt`` suffix so
    it is neither chosen again nor counted when old checkpoints are deleted.
    Leftover temporary files from interrupted writes are removed.
    """

    directory = checkpoint_directory(output)
    if directory.is_dir():
        for path in directory.iterdir():
            if _TEMPORARY_NAME.match(path.name):
                path.unlink(missing_ok=True)
    for update, path in list_checkpoints(output):
        try:
            payload = load_checkpoint(path)
            saved_update = payload.get("update")
            if saved_update != update:
                raise ValueError(f"file name says update {update}, contents say {saved_update}")
        except Exception as error:  # noqa: BLE001 - any unreadable file is skipped
            quarantined = path.with_name(path.name + ".corrupt")
            _LOGGER.warning("skipping unreadable checkpoint %s: %s", path, error)
            path.rename(quarantined)
            continue
        return path
    return None


def _comparable_group(group: Mapping[str, Any]) -> dict[str, Any]:
    # ``lr`` follows the schedule; ``params`` are positional indices.
    return {key: _json_value(value) for key, value in group.items() if key not in {"lr", "params"}}


def _check_optimizer_groups(saved: Mapping[str, Any], optimizer: Optimizer, name: str) -> None:
    saved_groups = saved["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError(
            f"{name} has {len(saved_groups)} saved groups, run has {len(optimizer.param_groups)}"
        )
    for index, (old, new) in enumerate(zip(saved_groups, optimizer.param_groups, strict=True)):
        old_values, new_values = _comparable_group(old), _comparable_group(new)
        shared = set(old_values) & set(new_values)
        changed = {
            key: (old_values[key], new_values[key])
            for key in sorted(shared)
            if old_values[key] != new_values[key]
        }
        if changed:
            raise ValueError(
                f"{name} group {index} hyperparameters changed (saved, run): {changed}"
            )
        if len(old["params"]) != len(new["params"]):
            raise ValueError(f"{name} group {index} owns a different number of parameters")


def restore_checkpoint(
    payload: Mapping[str, Any],
    *,
    path: Path,
    model: nn.Module,
    optimizers: Sequence[Optimizer],
    train_loader: FineWebBatchLoader,
    fingerprint: Mapping[str, Any],
    run_steps: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> ResumePoint:
    """Check a loaded checkpoint against the run, then restore its state."""

    differences = fingerprint_differences(payload["fingerprint"], fingerprint)
    if differences:
        raise ValueError(
            f"cannot resume from {path}: the run configuration changed\n  "
            + "\n  ".join(differences)
        )
    update = payload["update"]
    if not isinstance(update, int) or not 0 < update <= run_steps:
        raise ValueError(f"checkpoint update {update!r} is outside this run's {run_steps} updates")

    saved_optimizers = payload["optimizers"]
    names = [type(optimizer).__name__ for optimizer in optimizers]
    saved_names = [entry["type"] for entry in saved_optimizers]
    if names != saved_names:
        raise ValueError(f"optimizers differ: checkpoint {saved_names}, run {names}")
    for entry, optimizer, name in zip(saved_optimizers, optimizers, names, strict=True):
        _check_optimizer_groups(entry["state"], optimizer, name)

    model.load_state_dict(payload["model"], strict=True)
    for entry, optimizer in zip(saved_optimizers, optimizers, strict=True):
        optimizer.load_state_dict(entry["state"])
    train_loader.load_state_dict(dict(payload["train_loader"]))
    rng_states = payload["rng"]
    if len(rng_states) == world_size:
        restore_rng_state(rng_states[rank], device)
    elif rank == 0:
        _LOGGER.warning(
            "checkpoint has random state for %d ranks, run has %d; keeping fresh random state",
            len(rng_states),
            world_size,
        )
    return ResumePoint(path=path, update=update, tracker=dict(payload["tracker"]))

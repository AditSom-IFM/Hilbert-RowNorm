"""Checkpoint persistence for pretraining runs."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from .model import TransformerLM


def save_checkpoint(
    path: Path,
    model: TransformerLM,
    optimizers: list[Optimizer],
    step: int,
    args: Namespace,
    rank: int,
) -> None:
    """Save one rank-zero checkpoint, then synchronize all initialized ranks."""

    if rank == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                "step": step,
                "args": vars(args),
                "model_config": asdict(model.config),
            },
            path,
        )
    if dist.is_initialized():
        dist.barrier()

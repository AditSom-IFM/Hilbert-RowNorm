"""Distributed Hilbert diagnostics for a fixed panel of hidden states."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class HilbertStepRms:
    """RMS Hilbert perturbation and the number of measured tokens."""

    rms: float
    tokens: int


class HilbertRmsAccumulator:
    """Accumulate ``range(S h)^2`` without retaining vocabulary-sized logits."""

    def __init__(self, device: torch.device) -> None:
        self.square_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.tokens = 0

    @torch.no_grad()
    def add(self, step_logits: Tensor) -> None:
        if step_logits.ndim != 2 or min(step_logits.shape) < 1:
            raise ValueError(
                "step logits must have non-empty shape [tokens, vocabulary_size]"
            )
        values = step_logits.detach().float()
        perturbations = values.amax(dim=1) - values.amin(dim=1)
        self.square_sum += perturbations.square().sum(dtype=torch.float64)
        self.tokens += perturbations.numel()

    @torch.no_grad()
    def compute(self) -> HilbertStepRms:
        """Reduce a snapshot, retaining local totals for later adds/computes."""

        square_sum = self.square_sum.clone()
        tokens = torch.tensor(
            self.tokens,
            device=self.square_sum.device,
            dtype=torch.int64,
        )
        if dist.is_initialized():
            dist.all_reduce(square_sum)
            dist.all_reduce(tokens)
        count = int(tokens.item())
        if count < 1:
            raise ValueError("cannot compute Hilbert diagnostics without hidden states")
        rms = float(torch.sqrt(square_sum / count).item())
        if not math.isfinite(rms):
            raise ValueError("Hilbert diagnostics require finite step logits")
        return HilbertStepRms(rms=rms, tokens=count)

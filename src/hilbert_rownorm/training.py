"""Training and evaluation operations for the pretraining harness."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from .data import FineWebBatchLoader
from .hilbert_diagnostics import HilbertRmsAccumulator, HilbertStepRms
from .optimizer_factory import set_lr_scale


def schedule_scale(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """Linearly warm up, then cosine-decay to a fraction of the peak LR."""

    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if step < warmup_steps:
        return step / warmup_steps
    decay_updates = total_steps - warmup_steps
    if decay_updates == 1:
        return min_lr_ratio
    progress = min(1.0, (step - warmup_steps) / (decay_updates - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _distributed_mean(value: Tensor) -> Tensor:
    tensor = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(tensor)
        tensor /= dist.get_world_size()
    return tensor


def _accumulate_gradients(
    model: nn.Module,
    loader: FineWebBatchLoader,
    *,
    accumulation_steps: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> Tensor:
    """Accumulate one synchronized global batch without modifying its gradients."""

    loss_sum = torch.zeros((), device=device)
    no_sync = getattr(model, "no_sync", None)

    for micro_step in range(accumulation_steps):
        inputs, targets = loader.next_batch()
        inputs = inputs.to(device)
        targets = targets.to(device)
        sync = micro_step + 1 == accumulation_steps or not callable(no_sync)
        sync_context = nullcontext() if sync else no_sync()
        autocast = (
            torch.autocast(device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with sync_context, autocast:
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        (loss / accumulation_steps).backward()
        loss_sum += loss.detach()
    return loss_sum / accumulation_steps


def train_step(
    model: nn.Module,
    optimizers: list[Optimizer],
    loader: FineWebBatchLoader,
    *,
    accumulation_steps: int,
    lr_scale: float,
    max_gradient_norm: float,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
    gradient_clipping: bool = True,
) -> tuple[Tensor, Tensor]:
    """Run one optimizer update and return its pre-clipping global gradient norm."""

    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss = _accumulate_gradients(
        model,
        loader,
        accumulation_steps=accumulation_steps,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(),
        max_gradient_norm if gradient_clipping else math.inf,
        error_if_nonfinite=True,
    )
    set_lr_scale(optimizers, lr_scale)
    for optimizer in optimizers:
        optimizer.step()
    return loss, grad_norm.detach()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: FineWebBatchLoader,
    batches: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    *,
    head_step: Tensor | None = None,
    hilbert_probe_tokens: int = 0,
) -> tuple[float, HilbertStepRms | None]:
    if hilbert_probe_tokens < 0:
        raise ValueError("Hilbert probe tokens must be non-negative")
    if (head_step is None) != (hilbert_probe_tokens == 0):
        raise ValueError("Hilbert probes require both a head step and positive token count")
    model.eval()
    loader.reset()
    total = torch.zeros((), device=device)
    hilbert_diagnostics = (
        HilbertRmsAccumulator(device) if head_step is not None else None
    )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_probe_tokens, remainder = divmod(hilbert_probe_tokens, world_size)
    local_probe_tokens += int(rank < remainder)
    observed_probe_tokens = 0
    for _ in range(batches):
        inputs, targets = loader.next_batch()
        inputs = inputs.to(device)
        targets = targets.to(device)
        autocast = (
            torch.autocast(device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        collect_hidden = (
            hilbert_diagnostics is not None
            and observed_probe_tokens < local_probe_tokens
        )
        with autocast:
            if collect_hidden:
                output = model(inputs, return_hidden=True)
                if not isinstance(output, tuple):
                    raise TypeError("model must return logits and hidden states")
                logits, hidden = output
                remaining = local_probe_tokens - observed_probe_tokens
                selected = hidden.reshape(-1, hidden.shape[-1])[:remaining]
                assert head_step is not None
                step_logits = F.linear(selected, head_step)
            else:
                output = model(inputs)
                if isinstance(output, tuple):
                    raise TypeError("model returned hidden states unexpectedly")
                logits = output
        logits = logits.float()
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if collect_hidden:
            hilbert_diagnostics.add(step_logits)
            observed_probe_tokens += step_logits.shape[0]
    hilbert = hilbert_diagnostics.compute() if hilbert_diagnostics is not None else None
    if hilbert is not None and hilbert.tokens != hilbert_probe_tokens:
        raise RuntimeError(
            f"Hilbert panel requested {hilbert_probe_tokens} tokens, "
            f"but validation supplied {hilbert.tokens}"
        )
    return _distributed_mean(total / batches).item(), hilbert

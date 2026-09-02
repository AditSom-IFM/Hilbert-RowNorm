"""Single-node distributed process initialization."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def initialize_distributed(require_cuda: bool) -> tuple[int, int, torch.device]:
    """Initialize the current process from the standard torchrun environment."""

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for this launch; refusing to fall back to CPU")
    if require_cuda and world_size > 1 and not dist.is_nccl_available():
        raise RuntimeError("NCCL is required for multi-GPU CUDA training")
    if cuda_available:
        if require_cuda and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device must support BF16")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} GPUs are visible"
            )
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world_size, device

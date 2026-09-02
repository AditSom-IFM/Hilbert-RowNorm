from __future__ import annotations

import pytest
import torch

import hilbert_rownorm.train as train_module
from hilbert_rownorm.distributed import initialize_distributed


def test_train_reexports_distributed_initialization() -> None:
    assert train_module._distributed is initialize_distributed


def test_distributed_defaults_to_one_cpu_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert initialize_distributed(require_cuda=False) == (
        0,
        1,
        torch.device("cpu"),
    )


def test_require_cuda_refuses_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="refusing to fall back to CPU"):
        initialize_distributed(require_cuda=True)


def test_cuda_process_uses_local_rank_and_nccl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)
    selected_devices: list[torch.device] = []
    backends: list[str] = []
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)
    monkeypatch.setattr(torch.distributed, "init_process_group", backends.append)

    rank, world_size, device = initialize_distributed(require_cuda=True)

    assert (rank, world_size, device) == (3, 4, torch.device("cuda", 1))
    assert selected_devices == [device]
    assert backends == ["nccl"]

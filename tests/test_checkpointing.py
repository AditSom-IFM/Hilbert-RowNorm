from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

import hilbert_rownorm.checkpointing as checkpointing
from hilbert_rownorm.config import ModelConfig


class FakeModel:
    def __init__(self) -> None:
        self.config = ModelConfig(
            dim=4,
            n_layers=1,
            head_dim=2,
            vocab_size=8,
            max_seq_len=16,
        )
        self.state = {"weight": torch.tensor([1.0])}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.state


class FakeOptimizer:
    def __init__(self, index: int) -> None:
        self.index = index

    def state_dict(self) -> dict[str, int]:
        return {"index": self.index}


def test_save_checkpoint_preserves_payload_and_barrier_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []
    saved: list[tuple[dict[str, object], Path]] = []

    def save(payload: dict[str, object], path: Path) -> None:
        events.append("save")
        saved.append((payload, path))

    monkeypatch.setattr(checkpointing.torch, "save", save)
    monkeypatch.setattr(checkpointing.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpointing.dist, "barrier", lambda: events.append("barrier"))
    model = FakeModel()
    optimizers = [FakeOptimizer(1), FakeOptimizer(2)]
    args = Namespace(model="tiny", output=tmp_path)
    path = tmp_path / "latest.pt"

    checkpointing.save_checkpoint(  # type: ignore[arg-type]
        path,
        model,
        optimizers,
        step=7,
        args=args,
        rank=0,
    )

    assert events == ["save", "barrier"]
    payload, saved_path = saved[0]
    assert saved_path == path
    assert list(payload) == ["model", "optimizers", "step", "args", "model_config"]
    assert payload["model"] is model.state
    assert payload["optimizers"] == [{"index": 1}, {"index": 2}]
    assert payload["step"] == 7
    assert payload["args"] is vars(args)
    assert payload["model_config"] == {
        "dim": 4,
        "n_layers": 1,
        "head_dim": 2,
        "mlp_expansion": 4,
        "vocab_size": 8,
        "max_seq_len": 16,
        "rope_theta": 10_000.0,
        "rms_norm_eps": 1e-6,
        "initializer_std": 0.02,
    }


def test_nonzero_rank_skips_save_but_still_enters_barrier(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        checkpointing.torch,
        "save",
        lambda *args, **kwargs: events.append("save"),
    )
    monkeypatch.setattr(checkpointing.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpointing.dist, "barrier", lambda: events.append("barrier"))

    checkpointing.save_checkpoint(  # type: ignore[arg-type]
        Path("ignored.pt"),
        FakeModel(),
        [FakeOptimizer(1)],
        step=3,
        args=Namespace(),
        rank=1,
    )

    assert events == ["barrier"]

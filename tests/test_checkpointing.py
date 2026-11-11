from __future__ import annotations

import errno
import logging
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

import hilbert_rownorm.checkpointing as checkpointing
from hilbert_rownorm.checkpointing import (
    capture_rng_state,
    checkpoint_path,
    list_checkpoints,
    load_checkpoint,
    restore_checkpoint,
    restore_rng_state,
    save_checkpoint,
    select_resume_checkpoint,
)
from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.data import FINEWEB_MAGIC, FINEWEB_VERSION, HEADER_INTS, FineWebBatchLoader
from hilbert_rownorm.model import TransformerLM
from hilbert_rownorm.optimizer_factory import build_optimizers, set_lr_scale
from hilbert_rownorm.optimizer_recipes import resolve_optimizer_recipe

CPU = torch.device("cpu")


def run_state(tmp_path: Path, head: str = "rownorm", seed: int = 0):
    torch.manual_seed(seed)
    config = ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=8)
    model = TransformerLM(config)
    recipe = resolve_optimizer_recipe("190m", "muon", head).config
    optimizers = build_optimizers(model, "muon", recipe)
    data = tmp_path / "data"
    if not data.exists():
        data.mkdir()
        header = np.zeros(HEADER_INTS, dtype="<i4")
        header[:3] = FINEWEB_MAGIC, FINEWEB_VERSION, 129
        with (data / "train.bin").open("wb") as handle:
            header.tofile(handle)
            (np.arange(129) % 31).astype("<u2").tofile(handle)
    loader = FineWebBatchLoader(data, "*.bin", batch_size=2, sequence_length=8, seed=1)
    return model, optimizers, loader


def take_step(model, optimizers, loader) -> None:
    inputs, targets = loader.next_batch()
    loss = torch.nn.functional.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
    loss.backward()
    set_lr_scale(optimizers, 1.0)
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def save(tmp_path: Path, state, update: int, **changes):
    model, optimizers, loader = state
    values = dict(
        update=update,
        model=model,
        optimizers=optimizers,
        train_loader=loader,
        fingerprint={"seed": 1},
        tracker_state={"wandb_run_id": "abc", "elapsed_seconds": 2.5},
        args=Namespace(output=tmp_path / "run", lr=None),
        keep=2,
        rank=0,
        world_size=1,
        device=CPU,
    )
    values.update(changes)
    return save_checkpoint(tmp_path / "run", **values)


def restore(tmp_path: Path, path: Path, state, **changes):
    model, optimizers, loader = state
    values = dict(
        path=path,
        model=model,
        optimizers=optimizers,
        train_loader=loader,
        fingerprint={"seed": 1},
        run_steps=10,
        rank=0,
        world_size=1,
        device=CPU,
    )
    values.update(changes)
    return restore_checkpoint(load_checkpoint(path), **values)


def test_saved_checkpoint_restores_model_optimizers_and_data_position(tmp_path: Path) -> None:
    state = run_state(tmp_path)
    take_step(*state)
    path = save(tmp_path, state, 1)
    assert path == checkpoint_path(tmp_path / "run", 1)
    assert path.name == "update-00000001.pt"
    take_step(*state)
    expected = {name: value.clone() for name, value in state[0].state_dict().items()}

    fresh = run_state(tmp_path, seed=5)
    point = restore(tmp_path, path, fresh)
    take_step(*fresh)

    assert point.update == 1
    assert point.tracker == {"wandb_run_id": "abc", "elapsed_seconds": 2.5}
    for name, value in fresh[0].state_dict().items():
        assert torch.equal(value, expected[name]), name
    payload = load_checkpoint(path)
    assert payload["args"] == {"lr": None, "output": str(tmp_path / "run")}
    assert payload["world_size"] == 1 and len(payload["rng"]) == 1


def test_only_the_newest_checkpoints_are_kept(tmp_path: Path) -> None:
    state = run_state(tmp_path)
    for update in (2, 4, 6, 8):
        save(tmp_path, state, update, keep=3)
    assert [update for update, _ in list_checkpoints(tmp_path / "run")] == [8, 6, 4]


def test_a_failed_write_keeps_earlier_checkpoints_and_lets_training_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    state = run_state(tmp_path)
    first = save(tmp_path, state, 2)
    before = first.read_bytes()

    def full_disk(payload, handle) -> None:
        handle.write(b"partial")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(checkpointing.torch, "save", full_disk)
    with caplog.at_level(logging.WARNING):
        assert save(tmp_path, state, 4) is None

    assert "checkpoint at update 4 was not saved" in caplog.text
    assert "No space left on device" in caplog.text
    assert [path.name for path in (tmp_path / "run" / "checkpoints").iterdir()] == [first.name]
    assert first.read_bytes() == before


def test_nonzero_ranks_write_nothing_and_return_rank_zero_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = run_state(tmp_path)
    expected = tmp_path / "run" / "checkpoints" / "update-00000003.pt"
    monkeypatch.setattr(
        checkpointing, "_all_gather", lambda value, world_size: [value, value]
    )
    monkeypatch.setattr(
        checkpointing,
        "_broadcast",
        lambda value, rank, world_size, device: {"path": str(expected), "error": None},
    )

    assert save(tmp_path, state, 3, rank=1, world_size=2) == expected
    assert not (tmp_path / "run").exists()


def test_ranks_with_different_data_positions_are_a_bug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = run_state(tmp_path)
    shifted = {**state[2].state_dict(), "block_position": 4}
    monkeypatch.setattr(
        checkpointing,
        "_all_gather",
        lambda value, world_size: [value, {**value, "loader": shifted}],
    )
    with pytest.raises(RuntimeError, match="training-data positions differ"):
        save(tmp_path, state, 3, world_size=2)


def test_selection_skips_unreadable_files_and_removes_interrupted_writes(
    tmp_path: Path,
) -> None:
    state = run_state(tmp_path)
    good = save(tmp_path, state, 2)
    newest = save(tmp_path, state, 4)
    newest.write_bytes(newest.read_bytes()[:500])
    renamed = save(tmp_path, state, 6, keep=3)
    renamed.rename(renamed.with_name("update-00000009.pt"))
    leftover = good.with_name(".update-00000010.pt.tmp-1234")
    leftover.write_bytes(b"interrupted")

    assert select_resume_checkpoint(tmp_path / "run") == good

    names = sorted(path.name for path in good.parent.iterdir())
    assert names == [
        "update-00000002.pt",
        "update-00000004.pt.corrupt",
        "update-00000009.pt.corrupt",
    ]
    assert select_resume_checkpoint(tmp_path / "empty") is None


def test_restore_rejects_a_changed_configuration(tmp_path: Path) -> None:
    state = run_state(tmp_path)
    path = save(tmp_path, state, 2)
    with pytest.raises(ValueError, match=r"run configuration changed\n  seed: checkpoint=1, run=2"):
        restore(tmp_path, path, run_state(tmp_path), fingerprint={"seed": 2})


def test_restore_rejects_a_different_optimizer_setup(tmp_path: Path) -> None:
    state = run_state(tmp_path)
    path = save(tmp_path, state, 2)
    with pytest.raises(ValueError, match="optimizers differ"):
        restore(tmp_path, path, run_state(tmp_path, head="adamw"))

    changed = run_state(tmp_path)
    changed[1][-1].param_groups[0]["beta"] = 0.5
    with pytest.raises(ValueError, match=r"RowNorm group 0 hyperparameters changed.*beta"):
        restore(tmp_path, path, changed)


def test_restore_rejects_an_update_past_the_end_of_the_run(tmp_path: Path) -> None:
    state = run_state(tmp_path)
    path = save(tmp_path, state, 12)
    with pytest.raises(ValueError, match="outside this run's 10 updates"):
        restore(tmp_path, path, run_state(tmp_path))


def test_load_rejects_other_files(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="not a Hilbert-RowNorm training checkpoint"):
        load_checkpoint(path)
    torch.save({"format": checkpointing.CHECKPOINT_FORMAT, "version": 99}, path)
    with pytest.raises(ValueError, match="checkpoint version 99"):
        load_checkpoint(path)


def test_random_state_round_trips_through_a_checkpoint_file(tmp_path: Path) -> None:
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    path = tmp_path / "rng.pt"
    torch.save(capture_rng_state(CPU), path)
    expected = (random.random(), np.random.rand(), torch.rand(2))

    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    restore_rng_state(torch.load(path, weights_only=True), CPU)

    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])

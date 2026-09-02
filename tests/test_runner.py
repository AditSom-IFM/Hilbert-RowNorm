from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import hilbert_rownorm.runner as runner
from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.optimizer_recipes import OptimizerRecipe
from hilbert_rownorm.run_plan import RunPlan


def _config() -> ModelConfig:
    return ModelConfig(
        dim=4,
        n_layers=1,
        head_dim=2,
        vocab_size=8,
        max_seq_len=16,
    )


def _plan(run_steps: int = 1) -> RunPlan:
    return RunPlan(
        accumulation_steps=1,
        tokens_per_step=16,
        target_tokens=160,
        schedule_steps=10,
        run_steps=run_steps,
        warmup_steps=2,
    )


def test_prepare_run_preserves_deterministic_construction_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []
    config = _config()
    plan = _plan(run_steps=10)
    base_model = SimpleNamespace(config=config)
    wrapped_model = object()
    optimizers = [object()]
    geometry = object()
    args = Namespace(
        model="tiny",
        seed=17,
        gradient_checkpointing=True,
        optimizer="adamw",
        lm_head_optimizer="adamw",
        micro_batch=2,
        data=tmp_path / "data",
        eval_tokens=100,
        output=tmp_path / "run",
        exact_head_diameter_every_log=True,
        exact_head_diameter_block_rows=64,
        hilbert_step_probe_tokens=128,
        log_every=10,
    )

    monkeypatch.setattr(
        runner,
        "validate_arguments",
        lambda received, world_size: events.append(("validate", world_size)),
    )
    monkeypatch.setattr(
        runner,
        "model_preset",
        lambda name: events.append(("preset", name)) or config,
    )
    monkeypatch.setattr(
        runner,
        "build_run_plan",
        lambda received, received_config, world_size: (
            events.append(("plan", world_size)) or plan
        ),
    )
    monkeypatch.setattr(
        runner.torch,
        "manual_seed",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        runner,
        "TransformerLM",
        lambda received_config, **kwargs: (
            events.append(("model", received_config, kwargs)) or base_model
        ),
    )
    optimizer_config = {"recipe": "test"}
    optimizer_recipe = OptimizerRecipe("adamw-backbone:tiny:adamw:adamw", optimizer_config)
    monkeypatch.setattr(
        runner,
        "resolve_optimizer_recipe",
        lambda model, backbone, head: (
            events.append(("optimizer_recipe", model, backbone, head))
            or optimizer_recipe
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_optimizers",
        lambda model, name, recipe: (
            events.append(("optimizers", model, name, recipe)) or optimizers
        ),
    )
    monkeypatch.setattr(
        runner,
        "HeadGeometry",
        lambda received, **kwargs: (
            events.append(("geometry", received, kwargs)) or geometry
        ),
    )
    monkeypatch.setattr(
        runner,
        "DDP",
        lambda model, **kwargs: events.append(("ddp", model, kwargs)) or wrapped_model,
    )

    class FakeLoader:
        shards = ("shard",)
        global_tokens_per_batch = 32

        def __init__(self, root, pattern, **kwargs) -> None:
            events.append(("loader", pattern, kwargs))

    monkeypatch.setattr(runner, "FineWebBatchLoader", FakeLoader)
    monkeypatch.setattr(
        runner,
        "evaluation_batches",
        lambda requested, size: (
            events.append(("evaluation_batches", requested, size)) or (3, 96)
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_run_config",
        lambda *values: events.append("run_config") or {"purpose": "test"},
    )

    prepared = runner.prepare_run(
        args,
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )

    assert [event[0] if isinstance(event, tuple) else event for event in events] == [
        "validate",
        "preset",
        "plan",
        "seed",
        "model",
        "optimizer_recipe",
        "optimizers",
        "geometry",
        "ddp",
        "loader",
        "loader",
        "evaluation_batches",
        "run_config",
    ]
    assert events[3] == ("seed", 17)
    assert events[4] == (
        "model",
        config,
        {
            "device": torch.device("cpu"),
            "compute_dtype": None,
            "gradient_checkpointing": True,
        },
    )
    assert events[5] == ("optimizer_recipe", "tiny", "adamw", "adamw")
    assert events[6] == ("optimizers", base_model, "adamw", optimizer_config)
    assert events[7] == (
        "geometry",
        optimizers,
        {
            "rank": 0,
            "total_updates": 10,
            "exact_diameter_every_log": True,
            "log_every": 10,
            "diameter_block_rows": 64,
            "hilbert_probe_tokens": 128,
        },
    )
    assert events[8] == (
        "ddp",
        base_model,
        {"device_ids": None, "broadcast_buffers": False},
    )
    assert events[9] == (
        "loader",
        "fineweb_train_*.bin",
        {
            "gradient_accumulation_steps": 1,
            "seed": 17,
            "batch_size": 2,
            "sequence_length": 16,
            "rank": 0,
            "world_size": 2,
        },
    )
    assert events[10] == (
        "loader",
        "fineweb_val_*.bin",
        {
            "shuffle": False,
            "batch_size": 2,
            "sequence_length": 16,
            "rank": 0,
            "world_size": 2,
        },
    )
    assert events[11] == ("evaluation_batches", 100, 32)
    assert prepared.model is base_model
    assert prepared.train_model is wrapped_model
    assert prepared.optimizers is optimizers
    assert prepared.head_geometry is geometry
    assert prepared.run_config == {"purpose": "test"}


def test_final_update_collects_all_retained_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []
    head_step = torch.ones(8, 4)
    base_model = object()
    wrapped_model = object()
    train_loader = object()
    val_loader = object()

    class FakeGeometry:
        def exact_diameter_due(self, update: int) -> bool:
            events.append(("diameter_due", update))
            return True

        def exact_diameter(self, enabled: bool) -> float | None:
            events.append(("diameter", enabled))
            return 0.25 if enabled else None

        def optimizer_step(self, enabled: bool) -> torch.Tensor | None:
            events.append(("head_step", enabled))
            return head_step if enabled else None

    class FakeTracker:
        active = True

        def add(self, loss: torch.Tensor, norm: torch.Tensor) -> None:
            events.append(("tracker_add", loss.item(), norm.item()))

        def report(self, update: int, scale: float, **kwargs) -> None:
            events.append(("tracker_report", update, scale, kwargs))

    args = Namespace(
        log_every=100,
        min_lr_ratio=0.1,
        max_gradient_norm=1.0,
        no_gradient_clipping=False,
        eval_every=100,
        hilbert_step_probe_tokens=8,
        save_every=100,
        output=tmp_path / "run",
    )
    run = runner.PreparedRun(
        args=args,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        config=_config(),
        plan=_plan(),
        model=base_model,  # type: ignore[arg-type]
        train_model=wrapped_model,  # type: ignore[arg-type]
        optimizers=[],
        head_geometry=FakeGeometry(),  # type: ignore[arg-type]
        train_loader=train_loader,  # type: ignore[arg-type]
        val_loader=val_loader,  # type: ignore[arg-type]
        val_batches=3,
        actual_val_tokens=96,
        device_name="CPU",
        run_config={},
    )
    tracker = FakeTracker()

    monkeypatch.setattr(
        runner,
        "schedule_scale",
        lambda *values: events.append(("schedule", values)) or 0.5,
    )
    def train_step(model, optimizers, loader, **kwargs):
        assert model is wrapped_model
        assert optimizers is run.optimizers
        assert loader is train_loader
        assert kwargs == {
            "accumulation_steps": 1,
            "lr_scale": 0.5,
            "max_gradient_norm": 1.0,
            "device": torch.device("cpu"),
            "autocast_dtype": None,
            "gradient_clipping": True,
        }
        events.append("train_step")
        return torch.tensor(2.0), torch.tensor(3.0)

    monkeypatch.setattr(runner, "train_step", train_step)

    def evaluate(model, loader, *values, **kwargs):
        assert model is wrapped_model
        assert loader is val_loader
        assert values == (3, torch.device("cpu"), None)
        assert kwargs["head_step"] is head_step
        assert kwargs["hilbert_probe_tokens"] == 8
        events.append("evaluate")
        return 4.0, "hilbert-rms"

    monkeypatch.setattr(runner, "evaluate", evaluate)
    def save_checkpoint(path, model, optimizers, step, received_args, rank):
        assert model is base_model
        assert optimizers is run.optimizers
        assert received_args is args
        events.append(("checkpoint", path.name, step, rank))

    monkeypatch.setattr(runner, "save_checkpoint", save_checkpoint)
    clock = iter((10.0, 12.0, 20.0, 23.0))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))

    runner.execute_updates(run, tracker)  # type: ignore[arg-type]

    names = [event[0] if isinstance(event, tuple) else event for event in events]
    assert names == [
        "schedule",
        "train_step",
        "tracker_add",
        "diameter_due",
        "diameter",
        "head_step",
        "evaluate",
        "checkpoint",
        "tracker_report",
    ]
    report = events[-1]
    assert isinstance(report, tuple)
    assert report[3]["head_diameter"] == 0.25
    assert report[3]["validation_hilbert"] == "hilbert-rms"
    assert report[3]["validation_seconds"] == 2.0
    assert report[3]["checkpoint_seconds"] == 3.0
    assert report[3]["checkpoint_path"] == tmp_path / "run" / "final.pt"


def test_run_pretraining_finishes_success_before_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    prepared = SimpleNamespace(plan=SimpleNamespace(run_steps=3))

    class FakeTracker:
        def finish(self, updates: int, exit_code: int) -> None:
            events.append(("finish", updates, exit_code))

    monkeypatch.setattr(
        runner,
        "prepare_run",
        lambda *values: events.append("prepare") or prepared,
    )
    monkeypatch.setattr(
        runner,
        "start_tracker",
        lambda run: events.append("start_tracker") or FakeTracker(),
    )
    monkeypatch.setattr(
        runner,
        "print_run_summary",
        lambda run: events.append("summary"),
    )
    monkeypatch.setattr(
        runner,
        "execute_updates",
        lambda run, tracker: events.append("execute"),
    )
    monkeypatch.setattr(runner.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        runner.dist,
        "destroy_process_group",
        lambda: events.append("destroy"),
    )

    runner.run_pretraining(Namespace(), 0, 1, torch.device("cpu"))

    assert events == [
        "prepare",
        "start_tracker",
        "summary",
        "execute",
        ("finish", 3, 0),
        "destroy",
    ]


def test_run_pretraining_reports_failure_before_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    prepared = SimpleNamespace(plan=SimpleNamespace(run_steps=3))

    class FakeTracker:
        def finish(self, updates: int, exit_code: int) -> None:
            events.append(("finish", updates, exit_code))

    monkeypatch.setattr(runner, "prepare_run", lambda *values: prepared)
    monkeypatch.setattr(runner, "start_tracker", lambda run: FakeTracker())
    monkeypatch.setattr(runner, "print_run_summary", lambda run: None)

    def fail(run, tracker) -> None:
        events.append("execute")
        raise RuntimeError("sentinel failure")

    monkeypatch.setattr(runner, "execute_updates", fail)
    monkeypatch.setattr(runner.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        runner.dist,
        "destroy_process_group",
        lambda: events.append("destroy"),
    )

    with pytest.raises(RuntimeError, match="sentinel failure"):
        runner.run_pretraining(Namespace(), 0, 1, torch.device("cpu"))

    assert events == ["execute", ("finish", 0, 1), "destroy"]

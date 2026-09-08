import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import hilbert_rownorm.tracking as tracking_module
from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.hilbert_diagnostics import HilbertStepRms
from hilbert_rownorm.model import TransformerLM
from hilbert_rownorm.optimizer_factory import build_optimizers, set_lr_scale
from hilbert_rownorm.tracking import Tracker, optimizer_configuration


def optimizer_config():
    return {
        "betas": [0.7, 0.8],
        "groups": {
            "backbone": {"lr": 0.01, "eps": 1e-8, "weight_decay": 0.05},
            "input_embedding": {
                "lr": 0.02,
                "eps": 1e-8,
                "weight_decay": 0.05,
            },
            "lm_head": {
                "algorithm": "adamw",
                "lr": 0.03,
                "eps": 1e-8,
                "weight_decay": 0.05,
            },
        },
    }


class FakeRun:
    def __init__(self) -> None:
        self.definitions = []
        self.records = []
        self.summary = {}
        self.exit_code = None

    def define_metric(self, *args, **kwargs) -> None:
        self.definitions.append((args, kwargs))

    def log(self, record) -> None:
        self.records.append(record)

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


def test_tracker_emits_only_retained_paper_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = FakeRun()
    init_calls = []

    def init(**kwargs):
        init_calls.append(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=init,
            Settings=lambda **kwargs: kwargs,
        ),
    )
    config = ModelConfig(dim=32, n_layers=1, head_dim=8, vocab_size=101, max_seq_len=16)
    model = TransformerLM(config)
    optimizers = build_optimizers(model, "adamw", optimizer_config())
    set_lr_scale(optimizers, 0.25)
    clock = iter((10.0, 20.0, 24.0, 30.0))
    monkeypatch.setattr(tracking_module.time, "perf_counter", lambda: next(clock))
    tracker = Tracker(
        rank=0,
        device=torch.device("cpu"),
        optimizers=optimizers,
        parameter_count=100,
        tokens_per_step=200,
        validation_tokens=400,
        max_gradient_norm=1.0,
        gradient_clipping=True,
        project="project",
        entity=None,
        name="run",
        group="190m-adamw",
        mode="online",
        directory=tmp_path,
        config={"purpose": "test"},
    )
    tracker.add(torch.tensor(2.0), torch.tensor(0.5))
    tracker.add(torch.tensor(4.0), torch.tensor(2.0))

    tracker.report(1, 0.25, log_train=False)
    assert run.records == []

    tracker.report(
        2,
        0.25,
        log_train=True,
        head_diameter=1.25,
        validation_loss=3.5,
        validation_hilbert=HilbertStepRms(rms=0.5, tokens=128),
        validation_seconds=1.5,
        checkpoint_path=tmp_path / "final.pt",
        checkpoint_seconds=0.75,
    )
    tracker.finish(2, 0)

    record = run.records[0]
    assert list(record) == [
        "progress/update",
        "progress/tokens",
        "progress/tpp",
        "optimizer/schedule_scale",
        "optimizer/lr_backbone",
        "optimizer/peak_lr_backbone",
        "optimizer/lr_input_embedding",
        "optimizer/peak_lr_input_embedding",
        "optimizer/lr_lm_head",
        "optimizer/peak_lr_lm_head",
        "diagnostics/lm_head_step/diameter_exact",
        "train/loss",
        "train/perplexity",
        "train/grad_norm",
        "train/grad_norm_max",
        "train/clipped_fraction",
        "train/would_clip_fraction",
        "performance/tokens_per_second",
        "performance/seconds_per_update",
        "performance/elapsed_seconds",
        "memory/peak_allocated_gib",
        "memory/peak_reserved_gib",
        "validation/loss",
        "validation/perplexity",
        "validation/tokens",
        "validation/seconds",
        "events/evaluation",
        "validation/hilbert_step/rms",
        "validation/hilbert_step/tokens",
        "events/checkpoint",
        "performance/checkpoint_seconds",
    ]
    assert record["train/loss"] == pytest.approx(3.0)
    assert record["train/grad_norm"] == pytest.approx(1.25)
    assert record["train/grad_norm_max"] == pytest.approx(2.0)
    assert record["train/clipped_fraction"] == pytest.approx(0.5)
    assert record["train/would_clip_fraction"] == pytest.approx(0.5)
    assert record["performance/tokens_per_second"] == pytest.approx(100.0)
    assert record["performance/seconds_per_update"] == pytest.approx(2.0)
    assert record["performance/elapsed_seconds"] == pytest.approx(4.0)
    assert record["progress/tpp"] == 4
    assert record["optimizer/lr_lm_head"] == pytest.approx(0.25 * 0.03)
    assert record["validation/loss"] == 3.5
    assert record["diagnostics/lm_head_step/diameter_exact"] == 1.25
    assert record["validation/hilbert_step/rms"] == 0.5
    assert record["validation/hilbert_step/tokens"] == 128
    obsolete_fragments = (
        "lm_head_rows",
        "first_order_gain",
        "diameter_lower_bound",
        "diameter_upper_bound",
        "validation/logits",
        "hilbert_step/mean",
        "hilbert_step/max",
    )
    assert all(fragment not in key for key in record for fragment in obsolete_fragments)
    assert run.summary == {
        "checkpoint_file": "final.pt",
        "checkpoint_update": 2,
        "final_update": 2,
        "final_tokens": 400,
        "final_tpp": 4.0,
        "elapsed_seconds": 10.0,
        "end_to_end_tokens_per_second": 40.0,
        "final_validation_loss": 3.5,
        "final_validation_hilbert_step_rms": 0.5,
        "final_validation_hilbert_step_tokens": 128,
    }
    assert run.exit_code == 0
    assert len(init_calls[0].pop("id")) == 32
    assert init_calls[0].pop("resume") == "never"
    assert init_calls == [
        {
            "project": "project",
            "entity": None,
            "name": "run",
            "group": "190m-adamw",
            "job_type": "train",
            "mode": "online",
            "dir": str(tmp_path),
            "config": {"purpose": "test"},
            "force": True,
            "settings": {
                "init_timeout": 60,
                "disable_git": True,
                "disable_code": True,
            },
        }
    ]
    assert run.definitions == [
        (("progress/tokens",), {}),
        (("progress/update",), {}),
        (("train/*",), {"step_metric": "progress/tokens"}),
        (("validation/*",), {"step_metric": "progress/tokens"}),
        (("optimizer/*",), {"step_metric": "progress/tokens"}),
        (("performance/*",), {"step_metric": "progress/tokens"}),
        (("memory/*",), {"step_metric": "progress/tokens"}),
        (("events/*",), {"step_metric": "progress/tokens"}),
        (("diagnostics/*",), {"step_metric": "progress/update"}),
        (
            ("validation/loss",),
            {"step_metric": "progress/tokens", "summary": "min"},
        ),
    ]
    assert capsys.readouterr().out == (
        "step=2 tokens=400 loss=3.00000 grad_norm=1.250 lr=2.500e-03 "
        "interval_tok/s=100 peak_allocated_gib=0.00 peak_reserved_gib=0.00\n"
        "step=2 val_loss=3.50000\n"
    )
    assert json.loads((tmp_path / "metrics.jsonl").read_text()) == record
    assert json.loads((tmp_path / "config.json").read_text()) == {"purpose": "test"}
    assert json.loads((tmp_path / "summary.json").read_text())["final_update"] == 2
    assert tracker._metrics_file.closed


@pytest.mark.parametrize("initialization_error", [None, ValueError("sentinel")])
def test_wandb_initialization_broadcasts_ddp_status(
    initialization_error: Exception | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []

    class OrderedRun(FakeRun):
        def define_metric(self, *args, **kwargs) -> None:
            events.append(("define_metric", args, kwargs))

    run = OrderedRun()

    def settings(**kwargs):
        events.append(("settings", kwargs))
        return kwargs

    def init(**kwargs):
        events.append("init")
        if initialization_error is not None:
            raise initialization_error
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init, Settings=settings),
    )
    monkeypatch.setattr(tracking_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        tracking_module.dist,
        "broadcast",
        lambda status, src: events.append(("broadcast", int(status.item()), src)),
    )

    def create() -> Tracker:
        return Tracker(
            rank=0,
            device=torch.device("cpu"),
            optimizers=[],
            parameter_count=100,
            tokens_per_step=200,
            validation_tokens=400,
            max_gradient_norm=1.0,
            gradient_clipping=True,
            project="project",
            entity=None,
            name="run",
            group="test",
            mode="online",
            directory=tmp_path,
            config={},
        )

    if initialization_error is None:
        assert create().run is run
        expected = [
            (
                "settings",
                {
                    "init_timeout": 60,
                    "disable_git": True,
                    "disable_code": True,
                },
            ),
            "init",
            *(
                ("define_metric", definition[0], definition[1])
                for definition in [
                    (("progress/tokens",), {}),
                    (("progress/update",), {}),
                    (("train/*",), {"step_metric": "progress/tokens"}),
                    (("validation/*",), {"step_metric": "progress/tokens"}),
                    (("optimizer/*",), {"step_metric": "progress/tokens"}),
                    (("performance/*",), {"step_metric": "progress/tokens"}),
                    (("memory/*",), {"step_metric": "progress/tokens"}),
                    (("events/*",), {"step_metric": "progress/tokens"}),
                    (("diagnostics/*",), {"step_metric": "progress/update"}),
                    (
                        ("validation/loss",),
                        {"step_metric": "progress/tokens", "summary": "min"},
                    ),
                ]
            ),
            ("broadcast", 0, 0),
        ]
    else:
        with pytest.raises(RuntimeError, match="ValueError: sentinel"):
            create()
        expected = [
            (
                "settings",
                {
                    "init_timeout": 60,
                    "disable_git": True,
                    "disable_code": True,
                },
            ),
            "init",
            ("broadcast", 1, 0),
        ]

    assert events == expected


def test_nonzero_rank_propagates_wandb_failure_without_initializing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initialization_calls = 0

    def init(**kwargs):
        nonlocal initialization_calls
        initialization_calls += 1
        return FakeRun()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(tracking_module.dist, "is_initialized", lambda: True)

    def broadcast(status: torch.Tensor, *, src: int) -> None:
        assert src == 0
        assert int(status.item()) == 0
        status.fill_(1)

    monkeypatch.setattr(tracking_module.dist, "broadcast", broadcast)

    with pytest.raises(RuntimeError, match="tracking initialization failed on rank zero"):
        Tracker(
            rank=1,
            device=torch.device("cpu"),
            optimizers=[],
            parameter_count=100,
            tokens_per_step=200,
            validation_tokens=400,
            max_gradient_norm=1.0,
            gradient_clipping=True,
            project="project",
            entity=None,
            name="run",
            group="test",
            mode="online",
            directory=tmp_path,
            config={},
        )

    assert initialization_calls == 0


def test_noop_report_preserves_buffered_training_metrics(tmp_path: Path) -> None:
    tracker = Tracker(
        rank=0,
        device=torch.device("cpu"),
        optimizers=[],
        parameter_count=100,
        tokens_per_step=200,
        validation_tokens=400,
        max_gradient_norm=1.0,
        gradient_clipping=True,
        project="project",
        entity=None,
        name="run",
        group="test",
        mode="disabled",
        directory=tmp_path,
        config={},
    )
    tracker.add(torch.tensor(2.0), torch.tensor(0.5))

    tracker.report(1, 1.0, log_train=False)

    assert len(tracker._training_metrics.losses) == 1
    assert len(tracker._training_metrics.grad_norms) == 1


def test_nonzero_rank_collects_training_metrics_before_returning(tmp_path: Path) -> None:
    tracker = Tracker(
        rank=1,
        device=torch.device("cpu"),
        optimizers=[],
        parameter_count=100,
        tokens_per_step=200,
        validation_tokens=400,
        max_gradient_norm=1.0,
        gradient_clipping=True,
        project="project",
        entity=None,
        name="run",
        group="test",
        mode="disabled",
        directory=tmp_path,
        config={},
    )
    tracker.add(torch.tensor(2.0), torch.tensor(0.5))

    tracker.report(1, 1.0, log_train=True)

    assert tracker._training_metrics.losses == []
    assert tracker._training_metrics.grad_norms == []


def test_disabled_clipping_is_distinguished_from_would_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = FakeRun()
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs),
    )
    config = ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=4)
    optimizers = build_optimizers(TransformerLM(config), "adamw", optimizer_config())
    tracker = Tracker(
        rank=0,
        device=torch.device("cpu"),
        optimizers=optimizers,
        parameter_count=100,
        tokens_per_step=200,
        validation_tokens=400,
        max_gradient_norm=1.0,
        gradient_clipping=False,
        project="project",
        entity=None,
        name="run",
        group="test",
        mode="online",
        directory=tmp_path,
        config={},
    )
    tracker.add(torch.tensor(2.0), torch.tensor(0.5))
    tracker.add(torch.tensor(4.0), torch.tensor(2.0))

    tracker.report(2, 1.0, log_train=True)

    assert run.records[0]["train/clipped_fraction"] == 0.0
    assert run.records[0]["train/would_clip_fraction"] == pytest.approx(0.5)


def test_optimizer_configuration_records_all_roles_and_rownorm_beta() -> None:
    config = optimizer_config()
    config["groups"]["lm_head"] = {
        "algorithm": "rownorm",
        "lr": 0.03,
        "beta": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    model_config = ModelConfig(
        dim=16,
        n_layers=1,
        head_dim=8,
        vocab_size=31,
        max_seq_len=4,
    )
    groups = optimizer_configuration(build_optimizers(TransformerLM(model_config), "adamw", config))

    assert [group["role"] for group in groups] == [
        "backbone",
        "input_embedding",
        "lm_head",
    ]
    assert groups[-1]["optimizer"] == "RowNorm"
    assert groups[-1]["beta"] == pytest.approx(0.95)
    assert groups[-1]["bias_correction"] is True
    assert groups[-1]["weight_decay"] == 0.0


def test_optimizer_configuration_records_muon_nesterov() -> None:
    config = {
        "backbone": {
            "lr": 0.008,
            "momentum": 0.95,
            "nesterov": True,
            "weight_decay": 0.1,
            "ns_steps": 5,
            "eps": 1e-5,
        },
        "auxiliary_adamw": {
            "betas": [0.9, 0.999],
            "groups": {
                "input_embedding": {
                    "lr": 0.004,
                    "eps": 1e-10,
                    "weight_decay": 0.1,
                },
                "lm_head": {
                    "algorithm": "adamw",
                    "lr": 0.004,
                    "eps": 1e-10,
                    "weight_decay": 0.1,
                },
            },
        },
    }
    model_config = ModelConfig(
        dim=16,
        n_layers=1,
        head_dim=8,
        vocab_size=31,
        max_seq_len=4,
    )
    backbone = optimizer_configuration(
        build_optimizers(TransformerLM(model_config), "muon", config)
    )[0]

    assert backbone["optimizer"] == "Muon"
    assert backbone["peak_lr"] == pytest.approx(0.008)
    assert backbone["momentum"] == pytest.approx(0.95)
    assert backbone["nesterov"] is True
    assert backbone["weight_decay"] == pytest.approx(0.1)
    assert backbone["ns_steps"] == 5
    assert backbone["eps"] == pytest.approx(1e-5)


@pytest.mark.parametrize("variable", tracking_module.INHERITED_RUN_VARIABLES)
def test_fresh_tracking_rejects_inherited_identity(monkeypatch, variable):
    monkeypatch.setenv(variable, "inherited")
    with pytest.raises(ValueError, match=variable):
        tracking_module.fresh_wandb_identity("new-training")


def test_fresh_identity_is_unique_and_curated_project_is_reserved(monkeypatch):
    for variable in tracking_module.INHERITED_RUN_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    first = tracking_module.fresh_wandb_identity("new-training")
    second = tracking_module.fresh_wandb_identity("new-training")
    assert first["id"] != second["id"]
    assert first["resume"] == second["resume"] == "never"
    with pytest.raises(ValueError, match="curated paper project"):
        tracking_module.fresh_wandb_identity("Hilbert-RowNorm")


def test_fresh_identity_is_accepted_by_real_wandb_settings(monkeypatch):
    # Exercise the installed SDK schema without creating a run or contacting W&B.
    import wandb

    for variable in tracking_module.INHERITED_RUN_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    identity = tracking_module.fresh_wandb_identity("new-training")
    settings = wandb.Settings(run_id=identity["id"], resume=identity["resume"])
    assert settings.run_id == identity["id"]
    assert settings.resume == "never"


def test_disabled_wandb_persists_both_diagnostics_locally(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    tracker = Tracker(
        rank=0,
        device=torch.device("cpu"),
        optimizers=[],
        parameter_count=100,
        tokens_per_step=200,
        validation_tokens=400,
        max_gradient_norm=1.0,
        gradient_clipping=True,
        project="unused",
        entity=None,
        name="run",
        group="test",
        mode="disabled",
        directory=tmp_path,
        config={"seed": 0},
    )
    tracker.report(
        10,
        0.5,
        log_train=False,
        head_diameter=0.02,
        validation_loss=3.1,
        validation_seconds=1.0,
        validation_hilbert=HilbertStepRms(rms=0.01, tokens=8192),
    )
    # The line is readable before finish: a crash does not discard buffered history.
    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert record["diagnostics/lm_head_step/diameter_exact"] == 0.02
    assert record["validation/hilbert_step/rms"] == 0.01
    assert record["progress/update"] == 10
    tracker.finish(10, 0)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["final_validation_hilbert_step_rms"] == 0.01
    assert summary["exit_code"] == 0


def test_local_metric_file_is_closed_when_wandb_finish_fails(tmp_path, monkeypatch):
    class FailingRun(FakeRun):
        def finish(self, **kwargs):
            raise RuntimeError("finish failed")

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: FailingRun(),
            Settings=lambda **kwargs: kwargs,
        ),
    )
    tracker = Tracker(
        rank=0,
        device=torch.device("cpu"),
        optimizers=[],
        parameter_count=100,
        tokens_per_step=200,
        validation_tokens=400,
        max_gradient_norm=1.0,
        gradient_clipping=True,
        project="project",
        entity=None,
        name="run",
        group="test",
        mode="online",
        directory=tmp_path,
        config={},
    )
    with pytest.raises(RuntimeError, match="finish failed"):
        tracker.finish(0, 1)
    assert tracker._metrics_file.closed


def test_local_initialization_failure_is_broadcast_to_peers(tmp_path, monkeypatch):
    (tmp_path / "metrics.jsonl").touch()
    statuses = []
    monkeypatch.setattr(tracking_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        tracking_module.dist, "broadcast", lambda status, src: statuses.append(status.item())
    )
    with pytest.raises(RuntimeError, match="FileExistsError"):
        Tracker(
            rank=0,
            device=torch.device("cpu"),
            optimizers=[],
            parameter_count=100,
            tokens_per_step=200,
            validation_tokens=400,
            max_gradient_norm=1.0,
            gradient_clipping=True,
            project="unused",
            entity=None,
            name="run",
            group="test",
            mode="disabled",
            directory=tmp_path,
            config={},
        )
    assert statuses == [1]


@pytest.fixture
def tracker_factory(tmp_path, monkeypatch):
    def create(run):
        monkeypatch.setitem(
            sys.modules,
            "wandb",
            SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs),
        )
        return Tracker(
            rank=0,
            device=torch.device("cpu"),
            optimizers=[],
            parameter_count=100,
            tokens_per_step=200,
            validation_tokens=400,
            max_gradient_norm=1.0,
            gradient_clipping=True,
            project="project",
            entity=None,
            name="run",
            group="test",
            mode="online",
            directory=tmp_path,
            config={},
        )

    return create


@pytest.mark.parametrize("fail_wandb_finish", [False, True])
def test_local_summary_failure_marks_wandb_failed_and_preserves_local_error(
    tracker_factory, monkeypatch, caplog, fail_wandb_finish
):
    class Run(FakeRun):
        def finish(self, *, exit_code):
            super().finish(exit_code=exit_code)
            if fail_wandb_finish:
                raise RuntimeError("remote shutdown failed")

    run = Run()
    tracker = tracker_factory(run)
    local_error = OSError("disk full while writing summary")
    original_open = Path.open

    def open_path(path, *args, **kwargs):
        if path.name == "summary.json":
            raise local_error
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(OSError) as caught:
        tracker.finish(1, 0)

    assert caught.value is local_error
    assert run.exit_code == 1
    assert tracker._metrics_file.closed
    if fail_wandb_finish:
        assert "W&B shutdown failed while handling a tracking error" in caplog.text


@pytest.mark.parametrize("fail_wandb_finish", [False, True])
def test_partial_wandb_initialization_closes_run_and_sink_before_broadcasting_failure(
    tracker_factory, monkeypatch, caplog, fail_wandb_finish
):
    class Run(FakeRun):
        def define_metric(self, *args, **kwargs):
            raise ValueError("metric definition failed")

        def finish(self, *, exit_code):
            super().finish(exit_code=exit_code)
            if fail_wandb_finish:
                raise RuntimeError("remote shutdown failed")

    run = Run()
    metric_files = []
    original_open = Path.open

    def open_path(path, *args, **kwargs):
        opened = original_open(path, *args, **kwargs)
        if path.name == "metrics.jsonl":
            metric_files.append(opened)
        return opened

    monkeypatch.setattr(Path, "open", open_path)
    statuses = []
    monkeypatch.setattr(tracking_module.dist, "is_initialized", lambda: True)

    def broadcast(status, *, src):
        assert src == 0
        assert run.exit_code == 1
        assert metric_files[0].closed
        statuses.append(status.item())

    monkeypatch.setattr(tracking_module.dist, "broadcast", broadcast)
    with pytest.raises(RuntimeError, match="ValueError: metric definition failed"):
        tracker_factory(run)

    assert statuses == [1]
    if fail_wandb_finish:
        assert "W&B shutdown failed while handling a tracking error" in caplog.text


def test_local_sink_close_failure_marks_wandb_failed(tracker_factory):
    local_error = OSError("metrics close failed")

    class FailingClose(io.StringIO):
        def close(self):
            super().close()
            raise local_error

    run = FakeRun()
    tracker = tracker_factory(run)
    tracker._metrics_file.close()
    tracker._metrics_file = FailingClose()

    with pytest.raises(OSError) as caught:
        tracker.finish(1, 0)

    assert caught.value is local_error
    assert run.exit_code == 1
    assert tracker._metrics_file.closed
    assert not (tracker.directory / "summary.json").exists()


def test_local_metric_write_failure_propagates_and_allows_failed_run_cleanup(tracker_factory):
    local_error = OSError("metrics write failed")

    class FailingWrite(io.StringIO):
        def write(self, value):
            raise local_error

    run = FakeRun()
    tracker = tracker_factory(run)
    tracker._metrics_file.close()
    tracker._metrics_file = FailingWrite()

    with pytest.raises(OSError) as caught:
        tracker.report(1, 1.0, log_train=False, head_diameter=0.02)

    assert caught.value is local_error
    assert run.records == []
    tracker.finish(0, 1)
    assert run.exit_code == 1
    assert tracker._metrics_file.closed

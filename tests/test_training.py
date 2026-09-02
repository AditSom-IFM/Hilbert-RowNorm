from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import hilbert_rownorm.runner as runner_module
import hilbert_rownorm.train as train_module
import hilbert_rownorm.training as training_module
from hilbert_rownorm.config import ModelConfig, model_preset
from hilbert_rownorm.hilbert_diagnostics import HilbertStepRms
from hilbert_rownorm.model import TransformerLM
from hilbert_rownorm.optimizer_factory import build_optimizers
from hilbert_rownorm.train import _parser
from hilbert_rownorm.training import evaluate, schedule_scale, train_step


class Batches:
    def __init__(self, vocab_size: int) -> None:
        self.inputs = torch.randint(vocab_size, (2, 4))
        self.targets = torch.randint(vocab_size, (2, 4))

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs, self.targets

    def reset(self) -> None:
        pass


def optimizer_config(name: str, head_algorithm: str = "adamw"):
    group = {"lr": 0.01, "eps": 1e-8, "weight_decay": 0.02}
    head = {"algorithm": head_algorithm, **group}
    if head_algorithm == "rownorm":
        head = {
            **head,
            "beta": 0.6,
            "weight_decay": 0.0,
        }
    if name == "adamw":
        return {
            "betas": [0.7, 0.8],
            "groups": {
                "backbone": group,
                "input_embedding": group,
                "lm_head": head,
            },
        }
    return {
        "backbone": {
            "lr": 0.01,
            "momentum": 0.5,
            "nesterov": False,
            "weight_decay": 0.02,
            "ns_steps": 1,
            "eps": 1e-8,
        },
        "auxiliary_adamw": {
            "betas": [0.6, 0.7],
            "groups": {"input_embedding": group, "lm_head": head},
        },
    }


REQUIRED_ARGUMENTS = {
    "--model": "190m",
    "--optimizer": "adamw",
    "--lm-head-optimizer": "adamw",
    "--data": "data",
    "--output": "run",
    "--micro-batch": "1",
    "--global-batch": "1",
    "--tokens-per-parameter": "1",
    "--seed": "0",
    "--warmup-fraction": "0",
    "--min-lr-ratio": "0",
    "--max-gradient-norm": "1",
    "--log-every": "1",
    "--eval-every": "0",
    "--eval-tokens": "1",
    "--save-every": "0",
}


def arguments_without(omitted: str) -> list[str]:
    return [
        item
        for option, value in REQUIRED_ARGUMENTS.items()
        if option != omitted
        for item in (option, value)
    ]


def test_global_batch_must_be_explicit() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(arguments_without("--global-batch"))


def test_micro_batch_must_be_explicit() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(arguments_without("--micro-batch"))


def test_lm_head_optimizer_must_be_explicit() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(arguments_without("--lm-head-optimizer"))


def test_head_geometry_probe_controls_are_explicit() -> None:
    defaults = _parser().parse_args(arguments_without("unused"))
    configured = _parser().parse_args(
        arguments_without("unused")
        + [
            "--exact-head-diameter-every-log",
            "--exact-head-diameter-block-rows",
            "64",
            "--hilbert-step-probe-tokens",
            "128",
        ]
    )

    assert defaults.exact_head_diameter_every_log is False
    assert defaults.exact_head_diameter_block_rows == 4096
    assert defaults.hilbert_step_probe_tokens == 0
    assert configured.exact_head_diameter_every_log is True
    assert configured.exact_head_diameter_block_rows == 64
    assert configured.hilbert_step_probe_tokens == 128


@pytest.mark.parametrize(
    "obsolete_arguments",
    [
        ["--optimizer-config", "optimizer.json"],
        ["--head-diagnostics-config", "head.json"],
        ["--exact-head-diameter-probes", "3"],
        ["--logit-diagnostics"],
        ["--raw-head-gradient-snapshots", "raw-gradients"],
    ],
)
def test_obsolete_diagnostic_options_are_rejected(
    obsolete_arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(arguments_without("unused") + obsolete_arguments)


def test_gradient_clipping_can_be_disabled_explicitly() -> None:
    default = _parser().parse_args(arguments_without("unused"))
    disabled = _parser().parse_args(
        arguments_without("unused") + ["--no-gradient-clipping"]
    )

    assert default.no_gradient_clipping is False
    assert disabled.no_gradient_clipping is True


@pytest.mark.parametrize(
    ("optimizer_name", "head_algorithm"),
    [
        ("adamw", "adamw"),
        ("adamw", "rownorm"),
        ("muon", "adamw"),
    ],
)
def test_one_cpu_training_update(optimizer_name: str, head_algorithm: str) -> None:
    torch.manual_seed(5)
    config = ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=4)
    model = TransformerLM(config)
    optimizers = build_optimizers(
        model,
        optimizer_name,
        optimizer_config(optimizer_name, head_algorithm),
    )
    head_before = model.lm_head.weight.detach().clone()

    loss, grad_norm = train_step(
        model,
        optimizers,
        Batches(config.vocab_size),  # type: ignore[arg-type]
        accumulation_steps=2,
        lr_scale=0.5,
        max_gradient_norm=1.0,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(loss).item()
    assert (grad_norm > 0).item()
    assert not torch.equal(head_before, model.lm_head.weight)


def test_disabled_gradient_clipping_measures_without_scaling() -> None:
    class BiasLogits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(3))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.bias.expand(*inputs.shape, -1)

    model = BiasLogits()
    optimizer = torch.optim.SGD(
        [{"params": model.parameters(), "lr": 1.0, "base_lr": 1.0, "role": "backbone"}]
    )
    batches = Batches(vocab_size=3)
    batches.targets.zero_()

    _, grad_norm = train_step(
        model,
        [optimizer],
        batches,  # type: ignore[arg-type]
        accumulation_steps=1,
        lr_scale=1.0,
        max_gradient_norm=0.1,
        device=torch.device("cpu"),
        gradient_clipping=False,
    )

    assert grad_norm.item() > 0.1
    assert model.bias.grad is not None
    assert model.bias.grad.norm().item() == pytest.approx(grad_norm.item())


def test_enabled_gradient_clipping_scales_to_the_configured_norm() -> None:
    class BiasLogits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(3))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.bias.expand(*inputs.shape, -1)

    model = BiasLogits()
    optimizer = torch.optim.SGD(
        [{"params": model.parameters(), "lr": 1.0, "base_lr": 1.0, "role": "backbone"}]
    )
    batches = Batches(vocab_size=3)
    batches.targets.zero_()

    _, grad_norm = train_step(
        model,
        [optimizer],
        batches,  # type: ignore[arg-type]
        accumulation_steps=1,
        lr_scale=1.0,
        max_gradient_norm=0.1,
        device=torch.device("cpu"),
    )

    assert grad_norm.item() > 0.1
    assert model.bias.grad is not None
    assert model.bias.grad.norm().item() == pytest.approx(0.1, abs=2e-7)


def test_evaluate_returns_validation_loss_without_hilbert_probe() -> None:
    config = ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=4)
    model = TransformerLM(config)
    batches = Batches(config.vocab_size)
    with torch.no_grad():
        logits = model(batches.inputs).float()
        expected = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batches.targets.reshape(-1),
        )

    loss, hilbert = evaluate(
        model, batches, batches=1, device=torch.device("cpu"), autocast_dtype=None  # type: ignore[arg-type]
    )

    assert loss == pytest.approx(expected.item())
    assert hilbert is None


def test_evaluate_measures_exact_hilbert_change_on_fixed_hidden_states() -> None:
    torch.manual_seed(11)
    config = ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=4)
    model = TransformerLM(config)
    batches = Batches(config.vocab_size)
    step = torch.randn_like(model.lm_head.weight) * 0.01
    with torch.no_grad():
        _, hidden = model(batches.inputs, return_hidden=True)
        selected = hidden.reshape(-1, hidden.shape[-1])[:5]
        changes = F.linear(selected, step)
        ranges = changes.max(dim=-1).values - changes.min(dim=-1).values

    _, hilbert = evaluate(
        model,
        batches,
        batches=1,
        device=torch.device("cpu"),
        autocast_dtype=None,
        head_step=step,
        hilbert_probe_tokens=5,
    )

    assert hilbert is not None
    assert hilbert.tokens == 5
    assert hilbert.rms == pytest.approx(ranges.square().mean().sqrt().item())


def test_evaluate_requires_the_full_hilbert_panel() -> None:
    config = ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=4)
    model = TransformerLM(config)
    batches = Batches(config.vocab_size)
    step = torch.randn_like(model.lm_head.weight)

    with pytest.raises(RuntimeError, match="requested 9 tokens"):
        evaluate(
            model,
            batches,
            batches=1,
            device=torch.device("cpu"),
            autocast_dtype=None,
            head_step=step,
            hilbert_probe_tokens=9,
        )


def test_evaluate_partitions_the_fixed_panel_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=4)
    model = TransformerLM(config)
    batches = Batches(config.vocab_size)
    step = torch.randn_like(model.lm_head.weight)
    selected_counts: list[int] = []

    class FakeAccumulator:
        def __init__(self, device: torch.device) -> None:
            pass

        def add(self, step_logits: torch.Tensor) -> None:
            selected_counts.append(step_logits.shape[0])

        def compute(self) -> HilbertStepRms:
            return HilbertStepRms(rms=1.0, tokens=8)

    monkeypatch.setattr(training_module, "HilbertRmsAccumulator", FakeAccumulator)
    monkeypatch.setattr(training_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(training_module.dist, "get_world_size", lambda: 3)
    monkeypatch.setattr(training_module.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(training_module.dist, "all_reduce", lambda value: None)

    _, hilbert = evaluate(
        model,
        batches,
        batches=1,
        device=torch.device("cpu"),
        autocast_dtype=None,
        head_step=step,
        hilbert_probe_tokens=8,
    )

    assert hilbert == HilbertStepRms(rms=1.0, tokens=8)
    assert selected_counts == [2]


def test_warmup_and_cosine_schedule() -> None:
    config = model_preset("190m")
    total_steps = config.parameter_count * 20 // (256 * config.max_seq_len)
    warmup_steps = int(0.1 * total_steps)

    assert (total_steps, warmup_steps) == (7_083, 708)
    assert schedule_scale(0, total_steps, warmup_steps, 0.1) == 0
    assert schedule_scale(354, total_steps, warmup_steps, 0.1) == pytest.approx(0.5)
    assert schedule_scale(708, total_steps, warmup_steps, 0.1) == 1
    assert schedule_scale(7_082, total_steps, warmup_steps, 0.1) == pytest.approx(0.1)


def test_max_steps_truncates_zero_based_updates_without_shortening_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = ModelConfig(
        dim=4,
        n_layers=1,
        head_dim=2,
        vocab_size=4,
        max_seq_len=2,
    )
    scales: list[float] = []
    reports: list[tuple[int, float]] = []
    tracker_configs: list[dict[str, object]] = []
    finishes: list[tuple[int, int]] = []

    class FakeLoader:
        shards = (Path("shard.bin"),)
        global_tokens_per_batch = 32

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class FakeHeadGeometry:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def exact_diameter_due(self, update: int) -> bool:
            return False

        def optimizer_step(self, enabled: bool) -> None:
            return None

        def exact_diameter(self, enabled: bool) -> None:
            return None

    class FakeTracker:
        active = False

        def __init__(self, *args: object, **kwargs: object) -> None:
            tracker_configs.append(kwargs["config"])  # type: ignore[arg-type]

        def add(self, loss: torch.Tensor, grad_norm: torch.Tensor) -> None:
            pass

        def report(self, update: int, scale: float, **kwargs: object) -> None:
            reports.append((update, scale))

        def finish(self, updates: int, exit_code: int) -> None:
            finishes.append((updates, exit_code))

    def fake_train_step(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        scales.append(kwargs["lr_scale"])  # type: ignore[arg-type]
        return torch.tensor(1.0), torch.tensor(2.0)

    monkeypatch.setattr(
        train_module,
        "_distributed",
        lambda require_cuda: (0, 1, torch.device("cpu")),
    )
    monkeypatch.setattr(runner_module, "model_preset", lambda name: config)
    monkeypatch.setattr(
        runner_module,
        "resolve_optimizer_recipe",
        lambda model, backbone, head: SimpleNamespace(identifier="test", config={}),
    )
    monkeypatch.setattr(runner_module, "build_optimizers", lambda model, name, recipe: [])
    monkeypatch.setattr(runner_module, "FineWebBatchLoader", FakeLoader)
    monkeypatch.setattr(runner_module, "HeadGeometry", FakeHeadGeometry)
    monkeypatch.setattr(runner_module, "Tracker", FakeTracker)
    monkeypatch.setattr(runner_module, "train_step", fake_train_step)

    arguments = arguments_without("--global-batch") + [
        "--global-batch",
        "16",
        "--max-steps",
        "3",
    ]
    arguments[arguments.index("--warmup-fraction") + 1] = "0.5"
    arguments[arguments.index("--output") + 1] = str(tmp_path / "run")

    train_module.main(arguments)

    assert config.parameter_count == 288
    assert tracker_configs[0]["schedule_steps"] == 9
    assert tracker_configs[0]["run_steps"] == 3
    assert scales == pytest.approx([0.0, 0.25, 0.5])
    assert reports == pytest.approx([(1, 0.0), (2, 0.25), (3, 0.5)])
    assert finishes == [(3, 0)]

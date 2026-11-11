"""Crash a real CPU run, resume it, and compare it with an uninterrupted run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import hilbert_rownorm.runner as runner
from hilbert_rownorm.checkpointing import list_checkpoints, load_checkpoint
from hilbert_rownorm.cli import build_parser
from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.data import FINEWEB_MAGIC, FINEWEB_VERSION, HEADER_INTS

CONFIG = ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=8)
VARIANTS = {
    "adamw": ["--lm-head-optimizer", "adamw"],
    "rownorm": ["--lm-head-optimizer", "rownorm"],
    "rownorm-compositional": [
        "--lm-head-optimizer",
        "rownorm",
        "--no-qk-norm",
        "--attention-qk-optimizer",
        "compositional",
        "--qk-lr",
        "0.004",
        "--cm-mp",
        "1",
    ],
    "adamw-muon-qk": [
        "--lm-head-optimizer",
        "adamw",
        "--lm-head-lr",
        "0.003",
        "--lm-head-betas",
        "0.9",
        "0.99",
        "--no-qk-norm",
        "--qk-lr",
        "0.004",
    ],
}
# Timing and memory fields differ between any two runs.
VOLATILE_PREFIXES = ("performance/", "memory/", "validation/seconds")


class CrashAt(RuntimeError):
    """The injected failure; raised before the given update runs."""


def write_data(root: Path) -> Path:
    """Two train shards of 8 updates each, so 20 updates cross a shard and an epoch."""

    root.mkdir()
    header = np.zeros(HEADER_INTS, dtype="<i4")
    for name, offset in (("train_000000", 0), ("train_000001", 7), ("val_000000", 3)):
        tokens = ((np.arange(257) * 5 + offset) % CONFIG.vocab_size).astype("<u2")
        header[:3] = FINEWEB_MAGIC, FINEWEB_VERSION, len(tokens)
        with (root / f"fineweb_{name}.bin").open("wb") as handle:
            header.tofile(handle)
            tokens.tofile(handle)
    return root


def arguments(data: Path, output: Path, variant: str, *extra: str) -> list[str]:
    return [
        "--model",
        "190m",
        "--optimizer",
        "muon",
        *VARIANTS[variant],
        "--data",
        str(data),
        "--output",
        str(output),
        "--micro-batch",
        "1",
        "--global-batch",
        "4",
        "--tokens-per-parameter",
        "20",
        "--seed",
        "3",
        "--max-steps",
        "20",
        "--warmup-fraction",
        "0.1",
        "--min-lr-ratio",
        "0.1",
        "--max-gradient-norm",
        "1",
        "--log-every",
        "2",
        "--eval-every",
        "4",
        "--eval-tokens",
        "32",
        "--save-every",
        "6",
        "--wandb-mode",
        "disabled",
        "--exact-head-diameter-every-log",
        "--exact-head-diameter-block-rows",
        "8",
        "--hilbert-step-probe-tokens",
        "16",
        *extra,
    ]


def train(argv: list[str], *, crash_at: int | None = None, rank: int = 0, world_size: int = 1):
    """Run the real trainer, optionally failing just before update ``crash_at``."""

    args = build_parser().parse_args(argv)
    checkpoints = list_checkpoints(args.output) if args.resume else []
    update = checkpoints[0][0] if checkpoints else 0
    real_step = runner.train_step

    def step(*values, **kwargs):
        nonlocal update
        update += 1
        if update == crash_at:
            raise CrashAt(f"injected failure before update {crash_at}")
        return real_step(*values, **kwargs)

    threads = torch.get_num_threads()
    runner.train_step = step
    try:
        torch.set_num_threads(1)
        runner.run_pretraining(args, rank, world_size, torch.device("cpu"))
    finally:
        runner.train_step = real_step
        torch.set_num_threads(threads)


@pytest.fixture
def tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "model_preset", lambda name: CONFIG)
    monkeypatch.setitem(sys.modules, "wandb", None)


def records(output: Path) -> list[dict[str, object]]:
    lines = (output / "metrics.jsonl").read_text().splitlines()
    return [
        {
            key: value
            for key, value in json.loads(line).items()
            if not key.startswith(VOLATILE_PREFIXES)
        }
        for line in lines
    ]


def assert_same_state(left: Path, right: Path) -> None:
    """Compare everything a further update would read, bit for bit."""

    first = load_checkpoint(list_checkpoints(left)[0][1])
    second = load_checkpoint(list_checkpoints(right)[0][1])
    assert first["update"] == second["update"]
    assert first["train_loader"] == second["train_loader"]
    assert first["fingerprint"] == second["fingerprint"]
    assert first["model"].keys() == second["model"].keys()
    for name, tensor in first["model"].items():
        assert torch.equal(tensor, second["model"][name]), name
    for old, new in zip(first["optimizers"], second["optimizers"], strict=True):
        assert old["type"] == new["type"]
        assert old["state"]["param_groups"] == new["state"]["param_groups"]
        assert old["state"]["state"].keys() == new["state"]["state"].keys()
        for index, values in old["state"]["state"].items():
            for key, value in values.items():
                other = new["state"]["state"][index][key]
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, other), (old["type"], index, key)
                else:
                    assert value == other, (old["type"], index, key)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_crash_and_resume_matches_an_uninterrupted_run(tmp_path: Path, tiny_model, variant):
    data = write_data(tmp_path / "data")
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    train(arguments(data, reference, variant))

    with pytest.raises(CrashAt):
        train(arguments(data, resumed, variant, "--resume"), crash_at=16)
    assert [update for update, _ in list_checkpoints(resumed)] == [12, 6]
    train(arguments(data, resumed, variant, "--resume"))

    assert_same_state(reference, resumed)
    assert [update for update, _ in list_checkpoints(resumed)] == [20, 18]
    resumed_records = records(resumed)
    resume_events = [record for record in resumed_records if "events/resume" in record]
    assert resume_events == [
        {
            "progress/update": 12,
            "progress/tokens": 12 * 4 * CONFIG.max_seq_len,
            "progress/tpp": 12 * 4 * CONFIG.max_seq_len / CONFIG.parameter_count,
            "events/resume": 1,
        }
    ]
    assert [record for record in resumed_records if "events/resume" not in record] == records(
        reference
    )

    discarded = [
        json.loads(line)["progress/update"]
        for line in (resumed / "metrics.discarded-1.jsonl").read_text().splitlines()
    ]
    assert discarded == [14]
    resumes = [json.loads(line) for line in (resumed / "resumes.jsonl").read_text().splitlines()]
    assert resumes == [
        {
            "resume": 1,
            "from_update": 12,
            "checkpoint": "update-00000012.pt",
            "discarded_records": 1,
            "config_changes": {},
        }
    ]
    failed = json.loads((resumed / "summary.before-resume-1.json").read_text())
    assert failed == {"exit_code": 1}
    reference_summary = json.loads((reference / "summary.json").read_text())
    resumed_summary = json.loads((resumed / "summary.json").read_text())
    for key in ("exit_code", "final_update", "final_tokens", "final_validation_loss"):
        assert resumed_summary[key] == reference_summary[key]


def test_resume_without_a_checkpoint_archives_the_attempt_and_starts_over(
    tmp_path: Path, tiny_model
) -> None:
    data = write_data(tmp_path / "data")
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    train(arguments(data, reference, "rownorm"))
    with pytest.raises(CrashAt):
        train(arguments(data, resumed, "rownorm", "--resume"), crash_at=4)
    assert list_checkpoints(resumed) == []

    train(arguments(data, resumed, "rownorm", "--resume"))

    archive = resumed / "abandoned" / "attempt-1"
    assert sorted(path.name for path in archive.iterdir()) == [
        "config.json",
        "metrics.jsonl",
        "summary.json",
    ]
    assert not (resumed / "resumes.jsonl").exists()
    assert records(resumed) == records(reference)
    assert_same_state(reference, resumed)


def test_resuming_a_finished_run_changes_nothing(tmp_path: Path, tiny_model, capsys) -> None:
    data = write_data(tmp_path / "data")
    output = tmp_path / "run"
    train(arguments(data, output, "adamw", "--resume"))
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}

    train(arguments(data, output, "adamw", "--resume"))

    after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert after == before
    assert "already complete; nothing to do" in capsys.readouterr().out


def test_extending_max_steps_continues_a_finished_run(tmp_path: Path, tiny_model) -> None:
    data = write_data(tmp_path / "data")
    reference, extended = tmp_path / "reference", tmp_path / "extended"
    longer = ["--max-steps", "24"]
    train(arguments(data, reference, "adamw", *longer))
    train(arguments(data, extended, "adamw", "--resume"))

    train(arguments(data, extended, "adamw", "--resume", *longer))

    assert_same_state(reference, extended)
    assert (extended / "summary.before-resume-1.json").exists()
    change = json.loads((extended / "resumes.jsonl").read_text())["config_changes"]
    assert change == {"max_steps": [20, 24], "run_steps": [20, 24]}


@pytest.mark.parametrize(
    ("change", "setting"),
    [
        (["--lm-head-lr", "0.007"], "lm_head_lr"),
        (["--seed", "4"], "seed"),
        (["--global-batch", "8"], "global_batch"),
        (["--eval-tokens", "64"], "eval_tokens"),
    ],
)
def test_resume_refuses_a_changed_configuration(
    tmp_path: Path, tiny_model, change, setting
) -> None:
    data = write_data(tmp_path / "data")
    output = tmp_path / "run"
    with pytest.raises(CrashAt):
        train(arguments(data, output, "adamw", "--resume"), crash_at=8)
    checkpoint = list_checkpoints(output)[0][1]
    before = checkpoint.read_bytes()

    with pytest.raises(ValueError, match=rf"run configuration changed[\s\S]*{setting}"):
        train(arguments(data, output, "adamw", "--resume", *change))

    assert checkpoint.read_bytes() == before
    assert not (output / "resumes.jsonl").exists()


def test_resume_accepts_a_different_microbatch_split(tmp_path: Path, tiny_model) -> None:
    data = write_data(tmp_path / "data")
    output = tmp_path / "run"
    with pytest.raises(CrashAt):
        train(arguments(data, output, "rownorm", "--resume"), crash_at=8)

    split = arguments(data, output, "rownorm", "--resume")
    split[split.index("--micro-batch") + 1] = "2"
    train(split)

    change = json.loads((output / "resumes.jsonl").read_text())["config_changes"]
    assert change == {"micro_batch": [1, 2], "gradient_accumulation_steps": [4, 2]}
    assert json.loads((output / "summary.json").read_text())["final_update"] == 20


def test_unreadable_newest_checkpoint_falls_back_to_the_previous_one(
    tmp_path: Path, tiny_model
) -> None:
    data = write_data(tmp_path / "data")
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    train(arguments(data, reference, "rownorm"))
    with pytest.raises(CrashAt):
        train(arguments(data, resumed, "rownorm", "--resume"), crash_at=16)
    newest = list_checkpoints(resumed)[0][1]
    newest.write_bytes(newest.read_bytes()[:1000])

    train(arguments(data, resumed, "rownorm", "--resume"))

    assert (newest.parent / (newest.name + ".corrupt")).exists()
    resumes = json.loads((resumed / "resumes.jsonl").read_text())
    assert resumes["from_update"] == 6
    assert_same_state(reference, resumed)


class FakeWandbRun:
    def __init__(self, run_id: str, log: list) -> None:
        self.id = run_id
        self.records = log
        self.summary: dict[str, object] = {}
        self.exit_code = None

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def log(self, record) -> None:
        self.records.append(record)

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


def test_resume_continues_the_same_wandb_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "model_preset", lambda name: CONFIG)
    calls: list[dict[str, object]] = []
    logged: list[dict[str, object]] = []
    runs: list[FakeWandbRun] = []

    def init(**kwargs):
        calls.append(kwargs)
        runs.append(FakeWandbRun(kwargs["id"], logged))
        return runs[-1]

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs)
    )
    data = write_data(tmp_path / "data")
    output = tmp_path / "run"
    online = arguments(data, output, "adamw", "--resume")
    online[online.index("--wandb-mode") + 1] = "offline"

    with pytest.raises(CrashAt):
        train(online, crash_at=10)
    train(online)

    assert [call["resume"] for call in calls] == ["never", "allow"]
    assert calls[0]["id"] == calls[1]["id"]
    assert runs[0].exit_code == 1 and runs[1].exit_code == 0
    assert runs[1].summary["resumes"] == 1
    assert runs[1].summary["resumed_from_update"] == 6
    updates = [record["progress/update"] for record in logged if "train/loss" in record]
    # Updates 8 was logged before the crash and again after resuming from 6.
    assert updates == [2, 4, 6, 8, 8, 10, 12, 14, 16, 18, 20]


def _distributed_worker(rank: int, world_size: int, store: str, argv: list[str], crash: int):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=world_size
    )
    runner.model_preset = lambda name: CONFIG
    sys.modules["wandb"] = None  # type: ignore[assignment]
    try:
        train(argv, crash_at=crash or None, rank=rank, world_size=world_size)
    except CrashAt:
        pass


def _spawn(tmp_path: Path, argv: list[str], crash: int = 0) -> None:
    store = tmp_path / f"store-{len(list(tmp_path.glob('store-*')))}"
    mp.spawn(_distributed_worker, args=(2, str(store), argv, crash), nprocs=2, join=True)


def test_two_rank_crash_and_resume_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    data = write_data(tmp_path / "data")
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    _spawn(tmp_path, arguments(data, reference, "rownorm-compositional"))
    _spawn(tmp_path, arguments(data, resumed, "rownorm-compositional", "--resume"), crash=16)
    assert [update for update, _ in list_checkpoints(resumed)] == [12, 6]

    _spawn(tmp_path, arguments(data, resumed, "rownorm-compositional", "--resume"))

    assert_same_state(reference, resumed)
    checkpoint = load_checkpoint(list_checkpoints(resumed)[0][1])
    assert checkpoint["world_size"] == 2 and len(checkpoint["rng"]) == 2
    assert [record for record in records(resumed) if "events/resume" not in record] == records(
        reference
    )


def test_repeated_resumes_from_one_checkpoint_are_numbered(tmp_path: Path, tiny_model) -> None:
    data = write_data(tmp_path / "data")
    output = tmp_path / "run"
    with pytest.raises(CrashAt):
        train(arguments(data, output, "adamw", "--resume"), crash_at=8)
    with pytest.raises(CrashAt):
        train(arguments(data, output, "adamw", "--resume"), crash_at=10)
    train(arguments(data, output, "adamw", "--resume"))

    resumes = [json.loads(line) for line in (output / "resumes.jsonl").read_text().splitlines()]
    assert [(entry["resume"], entry["from_update"]) for entry in resumes] == [(1, 6), (2, 6)]
    assert sorted(path.name for path in output.glob("summary*.json")) == [
        "summary.before-resume-1.json",
        "summary.before-resume-2.json",
        "summary.json",
    ]

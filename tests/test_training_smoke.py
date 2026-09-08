"""Exercise the real runner, optimizers, loader, and local diagnostics together."""

import json
import math
import sys

import numpy as np
import pytest
import torch

import hilbert_rownorm.runner as runner
from hilbert_rownorm.cli import build_parser
from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.data import FINEWEB_MAGIC, FINEWEB_VERSION, HEADER_INTS


@pytest.mark.parametrize("head", ["adamw", "rownorm"])
def test_real_cpu_run_writes_three_curves_without_wandb(tmp_path, monkeypatch, head):
    # Only shrink the architecture; retain the actual selected optimizer recipe.
    config = ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=8)
    monkeypatch.setattr(runner, "model_preset", lambda name: config)
    monkeypatch.setitem(sys.modules, "wandb", None)
    data = tmp_path / "data"
    data.mkdir()
    tokens = (np.arange(513) % config.vocab_size).astype("<u2")
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[:3] = FINEWEB_MAGIC, FINEWEB_VERSION, len(tokens)
    for split in ("train", "val"):
        with (data / f"fineweb_{split}_000000.bin").open("wb") as handle:
            header.tofile(handle)
            tokens.tofile(handle)

    output = tmp_path / "run"
    args = build_parser().parse_args(
        [
            "--model",
            "190m",
            "--optimizer",
            "muon",
            "--lm-head-optimizer",
            head,
            "--data",
            str(data),
            "--output",
            str(output),
            "--micro-batch",
            "2",
            "--global-batch",
            "4",
            "--tokens-per-parameter",
            "20",
            "--seed",
            "0",
            "--max-steps",
            "2",
            "--warmup-fraction",
            "0.1",
            "--min-lr-ratio",
            "0.1",
            "--max-gradient-norm",
            "1",
            "--log-every",
            "1",
            "--eval-every",
            "1",
            "--eval-tokens",
            "32",
            "--save-every",
            "0",
            "--wandb-mode",
            "disabled",
            "--exact-head-diameter-every-log",
            "--exact-head-diameter-block-rows",
            "8",
            "--hilbert-step-probe-tokens",
            "16",
        ]
    )
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        runner.run_pretraining(args, 0, 1, torch.device("cpu"))
    finally:
        torch.set_num_threads(threads)

    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [record["progress/update"] for record in records] == [1, 2]
    for record in records:
        for key in (
            "validation/loss",
            "diagnostics/lm_head_step/diameter_exact",
            "validation/hilbert_step/rms",
        ):
            assert math.isfinite(record[key]) and record[key] >= 0
        assert record["validation/hilbert_step/tokens"] == 16
    # The existing zero-based warmup starts at LR zero; logging still records it.
    for key in ("diagnostics/lm_head_step/diameter_exact", "validation/hilbert_step/rms"):
        assert records[0][key] == 0
        assert records[1][key] > 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["exit_code"] == 0
    assert summary["final_update"] == 2
    assert summary["final_validation_loss"] == records[-1]["validation/loss"]
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["diameter_algorithm"] == "centered_cdist_v2"
    assert metadata["precision_details"]["autocast"] is None
    assert not (output / "wandb").exists()

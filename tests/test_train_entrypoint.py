from __future__ import annotations

from argparse import Namespace

import torch

import hilbert_rownorm.train as train


def test_main_only_parses_initializes_and_delegates(monkeypatch) -> None:
    events: list[object] = []
    args = Namespace(require_cuda=True)

    class FakeParser:
        def parse_args(self, argv):
            events.append(("parse", argv))
            return args

    monkeypatch.setattr(train, "_parser", lambda: FakeParser())
    monkeypatch.setattr(
        train,
        "_distributed",
        lambda require_cuda: (
            events.append(("distributed", require_cuda))
            or (2, 4, torch.device("cpu"))
        ),
    )
    monkeypatch.setattr(
        train,
        "run_pretraining",
        lambda received, rank, world_size, device: events.append(
            ("run", received, rank, world_size, device)
        ),
    )

    train.main(["--sentinel"])

    assert events == [
        ("parse", ["--sentinel"]),
        ("distributed", True),
        ("run", args, 2, 4, torch.device("cpu")),
    ]

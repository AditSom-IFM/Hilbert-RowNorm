from __future__ import annotations

import hilbert_rownorm.train as train_module
from hilbert_rownorm.cli import DEFAULT_WANDB_PROJECT, build_parser


def test_train_reexports_the_cli_parser() -> None:
    assert train_module._parser is build_parser
    assert train_module.DEFAULT_WANDB_PROJECT == DEFAULT_WANDB_PROJECT


def test_cli_exposes_the_final_optimizer_choices_and_tracking_defaults() -> None:
    parser = build_parser()
    backbone_optimizer = next(
        action for action in parser._actions if action.dest == "optimizer"
    )
    lm_head_optimizer = next(
        action for action in parser._actions if action.dest == "lm_head_optimizer"
    )
    wandb_project = next(
        action for action in parser._actions if action.dest == "wandb_project"
    )
    wandb_mode = next(action for action in parser._actions if action.dest == "wandb_mode")

    assert parser.description == (
        "Single-node trainer for the final LM-head optimizer experiments."
    )
    assert backbone_optimizer.required is True
    assert backbone_optimizer.choices == ("adamw", "muon")
    assert lm_head_optimizer.required is True
    assert lm_head_optimizer.choices == ("adamw", "rownorm")
    assert wandb_project.default == DEFAULT_WANDB_PROJECT
    assert wandb_project.default == "Hilbert-RowNorm-training"
    assert parser.allow_abbrev is False
    assert wandb_mode.default == "online"
    assert wandb_mode.choices == ("online", "offline", "disabled")

from __future__ import annotations

from argparse import Namespace

import pytest

from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.head_optimizers import HeadAdamW
from hilbert_rownorm.model import TransformerLM
from hilbert_rownorm.optimizer_factory import build_optimizers
from hilbert_rownorm.optimizer_recipes import resolve_optimizer_recipe
from hilbert_rownorm.run_plan import validate_arguments


def run_args(**changes) -> Namespace:
    values = dict(
        global_batch=4,
        micro_batch=2,
        tokens_per_parameter=1,
        warmup_fraction=0.1,
        min_lr_ratio=0.1,
        max_gradient_norm=1.0,
        eval_tokens=8,
        optimizer="muon",
        lm_head_optimizer="rownorm",
    )
    values.update(changes)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("head", "overrides", "expected"),
    [
        (
            "adamw",
            {"betas": (0.8, 0.95), "weight_decay": 0.0},
            {"betas": [0.8, 0.95], "weight_decay": 0.0},
        ),
        ("rownorm", {"beta": 0.0, "lr": 0.0064}, {"beta": 0.0, "lr": 0.0064}),
    ],
)
def test_head_overrides_change_only_the_lm_head(head, overrides, expected) -> None:
    recipe = resolve_optimizer_recipe("190m", "muon", head, head_overrides=overrides)
    paper = resolve_optimizer_recipe("190m", "muon", head).config

    paper["auxiliary_adamw"]["groups"]["lm_head"].update(expected)
    assert recipe.config == paper
    assert recipe.identifier == f"sweep:190m:muon:{head}"


def test_no_overrides_keeps_the_paper_identifier_and_config() -> None:
    recipe = resolve_optimizer_recipe("190m", "muon", "rownorm", head_overrides={})

    assert recipe.identifier == "paper:190m:muon:rownorm"
    assert recipe.config == resolve_optimizer_recipe("190m", "muon", "rownorm").config


def test_unknown_override_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="adamw LM head cannot override"):
        resolve_optimizer_recipe("190m", "muon", "adamw", head_overrides={"beta": 0.9})


def test_head_betas_override_reaches_only_the_head_adamw() -> None:
    model = TransformerLM(
        ModelConfig(dim=16, n_layers=1, head_dim=8, vocab_size=31, max_seq_len=8)
    )
    config = resolve_optimizer_recipe(
        "190m", "muon", "adamw", head_overrides={"betas": [0.8, 0.95]}
    ).config
    optimizers = build_optimizers(model, "muon", config)

    head = next(o for o in optimizers if isinstance(o, HeadAdamW))
    embedding = optimizers[1]
    assert head.param_groups[0]["betas"] == (0.8, 0.95)
    assert embedding.param_groups[0]["betas"] == (0.9, 0.999)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lm_head_betas": [0.8, 0.9]}, "only to the AdamW LM head"),
        ({"lm_head_optimizer": "adamw", "lm_head_beta": 0.9}, "only to the RowNorm LM head"),
    ],
)
def test_argument_validation_matches_beta_flags_to_the_head(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_arguments(run_args(**changes), world_size=1)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lm_head_lr": 0.0}, "--lm-head-lr must be finite and positive"),
        ({"lm_head_lr": float("nan")}, "--lm-head-lr must be finite and positive"),
        ({"lm_head_beta": 1.0}, r"--lm-head-beta must be in \[0, 1\)"),
        (
            {"lm_head_optimizer": "adamw", "lm_head_betas": [0.9, 1.0]},
            r"--lm-head-betas must be in \[0, 1\)",
        ),
        ({"lm_head_weight_decay": -0.1}, "--lm-head-weight-decay must be finite and non-negative"),
        ({"cm_mp": 0.0}, "--cm-mp must be finite and positive"),
    ],
)
def test_override_values_are_checked_before_the_run_starts(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_arguments(run_args(**changes), world_size=1)


def test_edge_override_values_are_accepted() -> None:
    validate_arguments(
        run_args(lm_head_lr=5e-3, lm_head_beta=0.0, lm_head_weight_decay=0.0),
        world_size=1,
    )
    validate_arguments(
        run_args(lm_head_optimizer="adamw", lm_head_betas=[0.999, 0.9999]), world_size=1
    )


def test_removing_qk_norm_marks_the_recipe_as_a_sweep() -> None:
    recipe = resolve_optimizer_recipe("190m", "muon", "adamw", qk_norm=False)

    assert recipe.identifier == "sweep:190m:muon:adamw"
    assert recipe.config == resolve_optimizer_recipe("190m", "muon", "adamw").config

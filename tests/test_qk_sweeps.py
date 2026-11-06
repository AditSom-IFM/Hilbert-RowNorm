from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from hilbert_rownorm.compositional_qk import CompositionalQK
from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.model import TransformerLM
from hilbert_rownorm.muon import Muon
from hilbert_rownorm.optimizer_factory import build_optimizers, set_lr_scale
from hilbert_rownorm.optimizer_recipes import resolve_optimizer_recipe
from hilbert_rownorm.run_plan import validate_arguments

QK_VALUES = {"lr": 0.004, "momentum": 0.9, "weight_decay": 0.0}


def tiny_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    config = ModelConfig(
        dim=16, n_layers=2, head_dim=8, vocab_size=31, max_seq_len=8, qk_norm=False
    )
    return TransformerLM(config)


def train_steps(model: TransformerLM, optimizers, steps: int = 2) -> None:
    generator = torch.Generator().manual_seed(1)
    set_lr_scale(optimizers, 1.0)
    for _ in range(steps):
        tokens = torch.randint(31, (2, 8), generator=generator)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten())
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()


def test_muon_qk_overrides_give_query_key_their_own_group() -> None:
    recipe = resolve_optimizer_recipe(
        "190m", "muon", "adamw", attention_qk_overrides=QK_VALUES, qk_norm=False
    )
    paper = resolve_optimizer_recipe("190m", "muon", "adamw").config

    assert recipe.config["attention_qk"] == {
        "algorithm": "muon",
        "lr": 0.004,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0,
        "ns_steps": 5,
        "eps": 1e-5,
    }
    assert {k: v for k, v in recipe.config.items() if k != "attention_qk"} == paper
    assert recipe.identifier == "sweep:190m:muon:adamw"


def test_compositional_qk_overrides_replace_the_inherited_values() -> None:
    recipe = resolve_optimizer_recipe(
        "190m",
        "muon",
        "adamw",
        attention_qk="compositional",
        attention_qk_overrides={**QK_VALUES, "mp": 1.0},
        qk_norm=False,
    )

    assert recipe.config["attention_qk"] == {
        "algorithm": "compositional",
        "lr": 0.004,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0,
        "ns_steps": 5,
        "eps": 1e-5,
        "mp": 1.0,
        "damping": 0.01,
        "whitening": "isotropic",
    }
    assert recipe.config["backbone"]["momentum"] == 0.95
    assert recipe.config["backbone"]["weight_decay"] == 0.1


def test_no_overrides_keep_query_key_in_the_backbone_group() -> None:
    recipe = resolve_optimizer_recipe("190m", "muon", "adamw")

    assert "attention_qk" not in recipe.config
    assert recipe.identifier == "paper:190m:muon:adamw"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attention_qk_overrides": {"betas": [0.9, 0.99]}}, "muon attention_qk cannot override"),
        ({"attention_qk_overrides": {"whitening": "full"}}, "muon attention_qk cannot override"),
    ],
)
def test_unknown_query_key_overrides_are_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_optimizer_recipe("190m", "muon", "adamw", **kwargs)


def test_query_key_overrides_require_the_muon_backbone() -> None:
    with pytest.raises(ValueError, match="requires the Muon backbone"):
        resolve_optimizer_recipe(
            "190m", "adamw", "adamw", attention_qk_overrides={"lr": 0.004}
        )


@pytest.mark.parametrize(
    ("attention_qk", "optimizer_type"),
    [("muon", Muon), ("compositional", CompositionalQK)],
)
def test_factory_routes_query_key_to_a_separate_optimizer(attention_qk, optimizer_type) -> None:
    model = tiny_model()
    config = resolve_optimizer_recipe(
        "190m",
        "muon",
        "adamw",
        attention_qk=attention_qk,
        attention_qk_overrides=QK_VALUES,
        qk_norm=False,
    ).config
    optimizers = build_optimizers(model, "muon", config)

    backbone, attention = optimizers[0], optimizers[1]
    assert type(attention) is optimizer_type
    qk_ids = {id(p) for pair in model.qk_pairs() for p in pair}
    assert {id(p) for p in attention.param_groups[0]["params"]} == qk_ids
    assert not qk_ids & {id(p) for p in backbone.param_groups[0]["params"]}
    group = attention.param_groups[0]
    assert (group["base_lr"], group["momentum"], group["weight_decay"]) == (0.004, 0.9, 0.0)
    assert backbone.param_groups[0]["base_lr"] == 0.008
    assert [g["role"] for o in optimizers for g in o.param_groups] == [
        "backbone",
        "attention_qk",
        "input_embedding",
        "lm_head",
    ]


def test_separate_muon_group_with_backbone_values_matches_the_shared_group() -> None:
    shared = tiny_model()
    separate = tiny_model()
    paper = resolve_optimizer_recipe("190m", "muon", "adamw").config
    backbone_values = {
        key: paper["backbone"][key] for key in ("lr", "momentum", "weight_decay")
    }
    split = resolve_optimizer_recipe(
        "190m", "muon", "adamw", attention_qk_overrides=backbone_values
    ).config

    train_steps(shared, build_optimizers(shared, "muon", paper))
    train_steps(separate, build_optimizers(separate, "muon", split))

    for (name, left), right in zip(shared.named_parameters(), separate.parameters(), strict=True):
        assert torch.equal(left, right), name


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
        lm_head_optimizer="adamw",
        attention_qk_optimizer="muon",
        no_qk_norm=True,
    )
    values.update(changes)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"qk_lr": 0.004, "optimizer": "adamw"}, "--qk-\\* options require --optimizer muon"),
        ({"qk_lr": 0.0}, "--qk-lr must be finite and positive"),
        ({"qk_momentum": 1.0}, r"--qk-momentum must be in \[0, 1\)"),
        ({"qk_weight_decay": -0.1}, "--qk-weight-decay must be finite and non-negative"),
    ],
)
def test_query_key_arguments_are_validated(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_arguments(run_args(**changes), world_size=1)


def test_query_key_arguments_accept_both_grids() -> None:
    validate_arguments(run_args(qk_lr=0.032, qk_momentum=0.99, qk_weight_decay=0.0), world_size=1)
    validate_arguments(
        run_args(attention_qk_optimizer="compositional", qk_lr=0.002, qk_momentum=0.9),
        world_size=1,
    )

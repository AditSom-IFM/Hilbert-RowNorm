from __future__ import annotations

import copy
from argparse import Namespace

import pytest
import torch

from hilbert_rownorm.compositional_qk import CompositionalQK
from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.model import TransformerLM
from hilbert_rownorm.muon import Muon, orthogonalize_newton_schulz
from hilbert_rownorm.optimizer_factory import build_optimizers, set_lr_scale
from hilbert_rownorm.optimizer_recipes import resolve_optimizer_recipe
from hilbert_rownorm.run_plan import validate_arguments

HEADS, HEAD_DIM, WIDTH = 3, 4, 10


def pair(seed: int = 0) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    generator = torch.Generator().manual_seed(seed)
    shape = (HEADS * HEAD_DIM, WIDTH)
    query = torch.nn.Parameter(0.3 * torch.randn(shape, generator=generator))
    key = torch.nn.Parameter(0.3 * torch.randn(shape, generator=generator))
    query.grad = torch.randn(shape, generator=generator)
    key.grad = torch.randn(shape, generator=generator)
    return query, key


def optimizer(pairs, **changes) -> CompositionalQK:
    values = dict(
        head_dim=HEAD_DIM,
        lr=0.1,
        momentum=0.9,
        nesterov=False,
        weight_decay=0.0,
        mp=2.0,
        damping=1e-2,
        whitening="full",
        ns_steps=5,
        eps=1e-7,
    )
    values.update(changes)
    return CompositionalQK(pairs, **values)


def heads(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.detach().view(HEADS, HEAD_DIM, WIDTH)


def test_full_whitening_bounds_each_half_of_the_composed_change() -> None:
    query, key = pair()
    before_query, before_key = query.detach().clone(), key.detach().clone()
    optimizer([(query, key)], damping=1e-6).step()
    delta_query, delta_key = query.detach() - before_query, key.detach() - before_key

    # lr * mp / 2 per factor; the 5-step Newton-Schulz overshoots by up to ~15%.
    budget = 0.1 * 2.0 / 2
    query_term = heads(delta_query).mT @ heads(before_key)
    key_term = heads(before_query).mT @ heads(delta_key)
    for term in (query_term, key_term):
        norms = torch.linalg.matrix_norm(term, ord=2)
        assert (norms <= 1.2 * budget).all()
        assert (norms >= 0.8 * budget).all()
    composed = torch.linalg.matrix_norm(query_term + key_term, ord=2)
    assert (composed <= 2.4 * budget).all()


def test_isotropic_step_is_the_partner_scaled_per_head_muon_direction() -> None:
    query, key = pair()
    before_query = query.detach().clone()
    query_gradient = query.grad.clone()
    key_scale = (heads(key).square().sum(dim=(1, 2)) / HEAD_DIM + 1e-2).rsqrt()

    optimizer([(query, key)], whitening="isotropic").step()

    scaled = key_scale[:, None, None] * heads(query_gradient)
    direction = key_scale[:, None, None] * orthogonalize_newton_schulz(scaled, 5, 1e-7)
    expected = before_query - 0.1 * direction.reshape(before_query.shape)
    torch.testing.assert_close(query.detach(), expected)


def test_heads_are_updated_independently() -> None:
    first, second = pair(), pair()
    second[0].grad[HEAD_DIM : 2 * HEAD_DIM] += 1.0
    optimizer([first]).step()
    optimizer([second]).step()

    changed = (first[0] - second[0]).abs().amax(dim=1) > 0
    assert changed[HEAD_DIM : 2 * HEAD_DIM].all()
    assert not changed[:HEAD_DIM].any() and not changed[2 * HEAD_DIM :].any()


def test_momentum_weight_decay_and_skipped_gradients_follow_muon() -> None:
    query, key = pair()
    state = optimizer([(query, key)], nesterov=True, weight_decay=0.2)
    first_gradient = query.grad.clone()
    state.step()
    torch.testing.assert_close(state.state[query]["momentum_buffer"], first_gradient)

    query.grad = torch.zeros_like(query)
    key.grad = None
    before = query.detach().clone()
    state.step()
    # A missing key gradient skips the pair entirely, weight decay included.
    assert torch.equal(query.detach(), before)
    torch.testing.assert_close(state.state[query]["momentum_buffer"], first_gradient)


def test_state_round_trip_continues_identically() -> None:
    query, key = pair()
    state = optimizer([(query, key)], nesterov=True)
    state.step()
    clone_query, clone_key = (copy.deepcopy(p) for p in (query, key))
    resumed = optimizer([(clone_query, clone_key)], nesterov=True)
    resumed.load_state_dict(copy.deepcopy(state.state_dict()))

    for source, target in ((query, clone_query), (key, clone_key)):
        target.grad = source.grad.clone()
    state.step()
    resumed.step()
    assert torch.equal(query.detach(), clone_query.detach())
    assert torch.equal(key.detach(), clone_key.detach())


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"whitening": "diagonal"}, ValueError),
        ({"mp": 0.0}, ValueError),
        ({"momentum": 1.0}, ValueError),
        ({"head_dim": 5}, ValueError),
        ({"nesterov": 1}, TypeError),
    ],
)
def test_invalid_hyperparameters_are_rejected(changes, error) -> None:
    with pytest.raises(error):
        optimizer([pair()], **changes)


def tiny_model() -> TransformerLM:
    config = ModelConfig(
        dim=16, n_layers=2, head_dim=8, vocab_size=31, max_seq_len=8, qk_norm=False
    )
    return TransformerLM(config)


def compositional_config():
    return resolve_optimizer_recipe(
        "190m", "muon", "rownorm", attention_qk="compositional"
    ).config


def test_factory_routes_query_and_key_pairs_away_from_muon() -> None:
    model = tiny_model()
    optimizers = build_optimizers(model, "muon", compositional_config())

    muon, attention = optimizers[0], optimizers[1]
    assert isinstance(muon, Muon) and isinstance(attention, CompositionalQK)
    qk_ids = {id(p) for qk in model.qk_pairs() for p in qk}
    assert qk_ids == {id(p) for p in attention.param_groups[0]["params"]}
    assert not qk_ids & {id(p) for p in muon.param_groups[0]["params"]}
    assert len(muon.param_groups[0]["params"]) == 5 * model.config.n_layers
    assert attention.head_dim == model.config.head_dim
    assert [group["role"] for o in optimizers for group in o.param_groups] == [
        "backbone",
        "attention_qk",
        "input_embedding",
        "lm_head",
    ]
    set_lr_scale(optimizers, 0.5)
    assert attention.param_groups[0]["lr"] == pytest.approx(0.004)


def test_full_training_step_updates_every_parameter() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    optimizers = build_optimizers(model, "muon", compositional_config())
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    tokens = torch.randint(31, (2, 8))
    loss = torch.nn.functional.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten())
    loss.backward()
    set_lr_scale(optimizers, 1.0)
    for each in optimizers:
        each.step()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all()
        assert not torch.equal(parameter.detach(), before[name]), name


def test_compositional_recipe_inherits_the_muon_backbone_settings() -> None:
    recipe = resolve_optimizer_recipe(
        "380m",
        "muon",
        "adamw",
        attention_qk="compositional",
        attention_qk_overrides={"mp": 1.0, "whitening": "full"},
    )

    assert recipe.config["attention_qk"] == {
        "algorithm": "compositional",
        "lr": 0.008,
        "momentum": 0.95,
        "nesterov": True,
        "weight_decay": 0.1,
        "ns_steps": 5,
        "eps": 1e-5,
        "mp": 1.0,
        "damping": 0.01,
        "whitening": "full",
    }
    assert recipe.identifier == "sweep:380m:muon:adamw"
    paper = resolve_optimizer_recipe("380m", "muon", "adamw").config
    assert {k: v for k, v in recipe.config.items() if k != "attention_qk"} == paper


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attention_qk": "lion"}, "attention QK optimizer"),
        ({"attention_qk_overrides": {"mp": 1.0}}, "require attention_qk='compositional'"),
        (
            {"attention_qk": "compositional", "attention_qk_overrides": {"ns": 5}},
            "cannot override",
        ),
    ],
)
def test_recipe_variant_errors_are_clear(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_optimizer_recipe("190m", "muon", "adamw", **kwargs)


def test_compositional_recipe_requires_the_muon_backbone() -> None:
    with pytest.raises(ValueError, match="requires the Muon backbone"):
        resolve_optimizer_recipe("190m", "adamw", "adamw", attention_qk="compositional")


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
        attention_qk_optimizer="compositional",
        no_qk_norm=True,
    )
    values.update(changes)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"attention_qk_optimizer": "muon", "cm_mp": 1.0}, "require --attention-qk-optimizer"),
        ({"no_qk_norm": False}, "requires --no-qk-norm"),
        ({"optimizer": "adamw"}, "requires --optimizer muon"),
    ],
)
def test_argument_validation_rejects_inconsistent_variants(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_arguments(run_args(**changes), world_size=1)


def test_argument_validation_accepts_the_compositional_setup() -> None:
    validate_arguments(run_args(cm_mp=1.0, cm_whitening="full"), world_size=1)

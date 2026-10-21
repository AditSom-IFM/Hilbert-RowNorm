from __future__ import annotations

import copy

import pytest
import torch

from hilbert_rownorm.compositional_qk import CompositionalQK
from hilbert_rownorm.muon import orthogonalize_newton_schulz

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

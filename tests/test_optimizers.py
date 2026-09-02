from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from hilbert_rownorm.head_optimizers import HeadAdamW, RowNorm
from hilbert_rownorm.muon import Muon, orthogonalize_newton_schulz
from hilbert_rownorm.optimizer_factory import build_optimizers, set_lr_scale
from hilbert_rownorm.optimizer_recipes import resolve_optimizer_recipe
from hilbert_rownorm.tracking import optimizer_configuration
from hilbert_rownorm.training import schedule_scale


class TinyModel(nn.Module):
    def __init__(self, *, tied: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(7, 3)
        self.hidden = nn.Linear(3, 3, bias=False)
        self.lm_head = nn.Linear(3, 7, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight

    def hidden_parameters(self):
        yield self.hidden.weight


def adamw_config(head_algorithm: str = "adamw"):
    if head_algorithm == "rownorm":
        head = {
            "algorithm": "rownorm",
            "lr": 0.03,
            "beta": 0.6,
            "eps": 3e-8,
            "weight_decay": 0.1,
        }
    else:
        head = {
            "algorithm": head_algorithm,
            "lr": 0.03,
            "eps": 3e-8,
            "weight_decay": 0.06,
        }
    return {
        "betas": [0.7, 0.8],
        "groups": {
            "backbone": {"lr": 0.01, "eps": 1e-8, "weight_decay": 0.04},
            "input_embedding": {"lr": 0.02, "eps": 2e-8, "weight_decay": 0.05},
            "lm_head": head,
        },
    }


def muon_config():
    return {
        "backbone": {
            "lr": 0.04,
            "momentum": 0.5,
            "nesterov": False,
            "weight_decay": 0.06,
            "ns_steps": 2,
            "eps": 1e-8,
        },
        "auxiliary_adamw": {
            "betas": [0.6, 0.7],
            "groups": {
                "input_embedding": {"lr": 0.02, "eps": 1e-8, "weight_decay": 0.03},
                "lm_head": {
                    "algorithm": "adamw",
                    "lr": 0.01,
                    "eps": 2e-8,
                    "weight_decay": 0.02,
                },
            },
        },
    }


def test_adamw_uses_fully_specified_config() -> None:
    config = adamw_config()
    optimizers = build_optimizers(TinyModel(), "adamw", config)
    groups = [group for optimizer in optimizers for group in optimizer.param_groups]

    assert tuple(type(optimizer) for optimizer in optimizers) == (
        torch.optim.AdamW,
        HeadAdamW,
    )
    assert [
        [group["role"] for group in optimizer.param_groups]
        for optimizer in optimizers
    ] == [["backbone", "input_embedding"], ["lm_head"]]
    assert all(optimizer.defaults["decoupled_weight_decay"] is True for optimizer in optimizers)
    assert all("momentum" not in group and "nesterov" not in group for group in groups)
    assert [group["base_lr"] for group in groups] == pytest.approx([0.01, 0.02, 0.03])
    assert [group["role"] for group in groups] == [
        "backbone",
        "input_embedding",
        "lm_head",
    ]
    assert [group["weight_decay"] for group in groups] == pytest.approx([0.04, 0.05, 0.06])
    assert [group["eps"] for group in groups] == pytest.approx([1e-8, 2e-8, 3e-8])
    assert tuple(groups[0]["betas"]) == pytest.approx((0.7, 0.8))


@pytest.mark.parametrize(
    ("algorithm", "expected_type"),
    [
        ("adamw", HeadAdamW),
        ("rownorm", RowNorm),
    ],
)
def test_lm_head_algorithm_is_selected_by_config(algorithm, expected_type) -> None:
    optimizers = build_optimizers(TinyModel(), "adamw", adamw_config(algorithm))

    assert type(optimizers[0]) is torch.optim.AdamW
    assert type(optimizers[1]) is expected_type
    assert optimizers[0].param_groups[0]["role"] == "backbone"
    assert optimizers[0].param_groups[1]["role"] == "input_embedding"
    assert optimizers[1].param_groups[0]["role"] == "lm_head"


@pytest.mark.parametrize("algorithm", ["adamw", "rownorm"])
def test_lm_head_optimizer_exposes_the_learning_rate_scaled_step(algorithm: str) -> None:
    model = TinyModel()
    optimizer = build_optimizers(model, "adamw", adamw_config(algorithm))[-1]
    gradient = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [3.0, -2.0, 4.0],
            [-1.0, 5.0, 2.0],
            [2.0, 1.0, -3.0],
            [4.0, -1.0, 1.0],
            [-3.0, 2.0, 5.0],
            [1.0, -4.0, 2.0],
        ]
    )
    model.lm_head.weight.grad = gradient

    optimizer.step()
    step = optimizer.lm_head_step()  # type: ignore[attr-defined]

    assert step.shape == model.lm_head.weight.shape
    assert torch.isfinite(step).all()


def test_incomplete_optimizer_config_is_rejected() -> None:
    config = adamw_config()
    del config["groups"]["lm_head"]

    with pytest.raises(ValueError, match="exactly"):
        build_optimizers(TinyModel(), "adamw", config)


def test_unknown_head_algorithm_is_rejected() -> None:
    config = adamw_config("mystery")

    with pytest.raises(ValueError, match="algorithm"):
        build_optimizers(TinyModel(), "adamw", config)


@pytest.mark.parametrize(
    ("weight_decay", "message"),
    [(-0.1, "weight_decay >= 0"), (float("nan"), "finite"), (float("inf"), "finite")],
)
def test_rownorm_rejects_invalid_weight_decay(
    weight_decay: float, message: str
) -> None:
    parameter = nn.Parameter(torch.zeros(4, 2))

    with pytest.raises(ValueError, match=message):
        RowNorm(
            [parameter],
            lr=0.2,
            beta=0.6,
            eps=0.1,
            weight_decay=weight_decay,
        )


def test_rownorm_rejects_removed_bias_correction_option() -> None:
    config = adamw_config("rownorm")
    config["groups"]["lm_head"]["bias_correction"] = True

    with pytest.raises(ValueError, match="must contain exactly"):
        build_optimizers(TinyModel(), "adamw", config)


def test_muon_uses_fully_specified_config() -> None:
    optimizers = build_optimizers(TinyModel(), "muon", muon_config())
    groups = [group for optimizer in optimizers for group in optimizer.param_groups]

    assert [group["role"] for group in groups] == [
        "backbone",
        "input_embedding",
        "lm_head",
    ]
    assert [group["base_lr"] for group in groups] == pytest.approx([0.04, 0.02, 0.01])
    assert [group["weight_decay"] for group in groups] == pytest.approx([0.06, 0.03, 0.02])
    assert groups[0]["nesterov"] is False
    assert optimizers[0].state_dict()["param_groups"][0]["nesterov"] is False
    assert tuple(groups[-1]["betas"]) == pytest.approx((0.6, 0.7))
    assert type(optimizers[-1]) is HeadAdamW


def test_muon_with_rownorm_preserves_optimizer_and_semantic_group_order() -> None:
    model = TinyModel()
    config = muon_config()
    config["auxiliary_adamw"]["groups"]["lm_head"] = {
        "algorithm": "rownorm",
        "lr": 0.03,
        "beta": 0.6,
        "eps": 3e-8,
        "weight_decay": 0.1,
    }

    optimizers = build_optimizers(model, "muon", config)

    assert tuple(type(optimizer) for optimizer in optimizers) == (
        Muon,
        torch.optim.AdamW,
        RowNorm,
    )
    assert [
        [group["role"] for group in optimizer.param_groups]
        for optimizer in optimizers
    ] == [["backbone"], ["input_embedding"], ["lm_head"]]
    assert [
        [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        for optimizer in optimizers
    ] == [[id(model.hidden.weight)], [id(model.embed.weight)], [id(model.lm_head.weight)]]


def test_muon_config_requires_boolean_nesterov() -> None:
    missing = muon_config()
    del missing["backbone"]["nesterov"]
    with pytest.raises(ValueError, match="exactly"):
        build_optimizers(TinyModel(), "muon", missing)

    non_boolean = muon_config()
    non_boolean["backbone"]["nesterov"] = 1
    with pytest.raises(ValueError, match="must be a bool"):
        build_optimizers(TinyModel(), "muon", non_boolean)


def test_schedule_scales_standard_adamw_decay() -> None:
    model = TinyModel()
    config = adamw_config()
    optimizers = build_optimizers(model, "adamw", config)
    embed_before = model.embed.weight.detach().clone()
    hidden_before = model.hidden.weight.detach().clone()
    head_before = model.lm_head.weight.detach().clone()
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    set_lr_scale(optimizers, 0.25)

    for optimizer in optimizers:
        optimizer.step()

    torch.testing.assert_close(model.embed.weight, embed_before * (1 - 0.25 * 0.02 * 0.05))
    torch.testing.assert_close(model.hidden.weight, hidden_before * (1 - 0.25 * 0.01 * 0.04))
    torch.testing.assert_close(model.lm_head.weight, head_before * (1 - 0.25 * 0.03 * 0.06))
    assert [
        group["lr"] for optimizer in optimizers for group in optimizer.param_groups
    ] == pytest.approx([0.0025, 0.005, 0.0075])


def test_schedule_scales_rownorm_head_optimizer() -> None:
    optimizers = build_optimizers(TinyModel(), "adamw", adamw_config("rownorm"))

    set_lr_scale(optimizers, 0.25)

    assert optimizers[-1].param_groups[0]["lr"] == pytest.approx(0.0075)


def test_native_head_adamw_matches_stock_adamw() -> None:
    initial = torch.tensor(
        [[1.0, 3.0], [2.0, -1.0], [-4.0, 2.0], [5.0, -2.0]],
        dtype=torch.float64,
    )
    head_parameter = nn.Parameter(initial.clone())
    stock_parameter = nn.Parameter(initial.clone())
    head = HeadAdamW(
        [head_parameter],
        lr=0.2,
        betas=(0.4, 0.7),
        eps=1e-4,
        weight_decay=0.3,
    )
    stock = torch.optim.AdamW(
        [stock_parameter],
        lr=0.2,
        betas=(0.4, 0.7),
        eps=1e-4,
        weight_decay=0.3,
    )
    gradients = (
        torch.tensor([[1.0, 2.0], [2.0, -3.0], [-1.0, 4.0], [3.0, 1.0]]),
        torch.tensor([[0.5, -2.0], [3.0, 1.0], [2.0, -1.0], [-4.0, 2.0]]),
    )

    for gradient in gradients:
        head_parameter.grad = gradient.to(torch.float64)
        stock_parameter.grad = gradient.to(torch.float64)
        head.step()
        stock.step()

        torch.testing.assert_close(head_parameter, stock_parameter)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                head.state[head_parameter][key], stock.state[stock_parameter][key]
            )


@pytest.mark.parametrize("algorithm", ["adamw", "rownorm"])
def test_reported_head_step_excludes_decoupled_weight_decay(algorithm: str) -> None:
    parameter = nn.Parameter(torch.ones(4, 2))
    if algorithm == "adamw":
        optimizer = HeadAdamW(
            [parameter],
            lr=0.2,
            betas=(0.4, 0.7),
            eps=1e-4,
            weight_decay=0.3,
        )
    else:
        optimizer = RowNorm(
            [parameter],
            lr=0.2,
            beta=0.6,
            eps=0.1,
            weight_decay=0.3,
        )
    before = parameter.detach().clone()
    parameter.grad = torch.zeros_like(parameter)

    optimizer.step()

    assert not torch.equal(parameter, before)
    assert torch.equal(optimizer.lm_head_step(), torch.zeros_like(parameter))


def test_final_190m_adamw_head_matches_stock_adamw_bitwise() -> None:
    config = resolve_optimizer_recipe("190m", "muon", "adamw").config
    head_config = config["auxiliary_adamw"]["groups"]["lm_head"]
    betas = tuple(config["auxiliary_adamw"]["betas"])
    generator = torch.Generator().manual_seed(1_904)
    initial = torch.randn((257, 32), generator=generator)
    gradients = [torch.randn(initial.shape, generator=generator) for _ in range(32)]
    actual_parameter = nn.Parameter(initial.clone())
    reference_parameter = nn.Parameter(initial.clone())
    actual = HeadAdamW(
        [actual_parameter],
        lr=head_config["lr"],
        betas=betas,
        eps=head_config["eps"],
        weight_decay=head_config["weight_decay"],
    )
    reference = torch.optim.AdamW(
        [reference_parameter],
        lr=head_config["lr"],
        betas=betas,
        eps=head_config["eps"],
        weight_decay=head_config["weight_decay"],
    )

    for index, gradient in enumerate(gradients):
        scale = schedule_scale(index, 7_083, 708, 0.1)
        actual.param_groups[0]["lr"] = scale * head_config["lr"]
        reference.param_groups[0]["lr"] = scale * head_config["lr"]
        actual_parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        actual.step()
        reference.step()

        assert torch.equal(actual_parameter, reference_parameter)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(
                actual.state[actual_parameter][key],
                reference.state[reference_parameter][key],
            )
        state = actual.state[actual_parameter]
        step = int(state["step"].item())
        expected_head_step = -float(actual.param_groups[0]["lr"]) * (
            (state["exp_avg"] / (1.0 - betas[0] ** step))
            / (
                (state["exp_avg_sq"] / (1.0 - betas[1] ** step)).sqrt()
                + head_config["eps"]
            )
        )
        assert torch.equal(actual.lm_head_step(), expected_head_step)


def test_final_190m_rownorm_head_matches_manuscript_update_bitwise() -> None:
    config = resolve_optimizer_recipe("190m", "muon", "rownorm").config
    head_config = config["auxiliary_adamw"]["groups"]["lm_head"]
    generator = torch.Generator().manual_seed(1_905)
    initial = torch.randn((257, 32), generator=generator)
    gradients = [torch.randn(initial.shape, generator=generator) for _ in range(32)]
    parameter = nn.Parameter(initial.clone())
    optimizer = RowNorm(
        [parameter],
        lr=head_config["lr"],
        beta=head_config["beta"],
        eps=head_config["eps"],
        weight_decay=head_config["weight_decay"],
    )
    expected_parameter = initial.clone()
    expected_moment = torch.zeros_like(initial)

    for step, gradient in enumerate(gradients, start=1):
        scale = schedule_scale(step - 1, 7_083, 708, 0.1)
        learning_rate = scale * head_config["lr"]
        optimizer.param_groups[0]["lr"] = learning_rate
        parameter.grad = gradient.clone()
        expected_moment.mul_(head_config["beta"]).add_(
            gradient, alpha=1.0 - head_config["beta"]
        )
        correction = -math.expm1(step * math.log(head_config["beta"]))
        expected_update = expected_moment / (
            torch.linalg.vector_norm(expected_moment, dim=1, keepdim=True)
            + head_config["eps"] * correction
        )
        expected_parameter.mul_(
            1.0 - learning_rate * head_config["weight_decay"]
        ).add_(expected_update, alpha=-learning_rate)

        optimizer.step()

        assert torch.equal(parameter, expected_parameter)
        assert torch.equal(
            optimizer.state[parameter]["momentum_buffer"], expected_moment
        )
        assert optimizer.state[parameter]["step"] == step
        assert torch.equal(
            optimizer.lm_head_step(),
            -learning_rate * expected_update,
        )


@pytest.mark.parametrize("algorithm", ["adamw", "rownorm"])
def test_head_optimizer_state_round_trip_continues_identically(algorithm: str) -> None:
    def make(parameter: nn.Parameter):
        if algorithm == "rownorm":
            return RowNorm(
                [parameter],
                lr=0.2,
                beta=0.6,
                eps=0.1,
                weight_decay=0.0,
            )
        return HeadAdamW(
            [parameter],
            lr=0.2,
            betas=(0.4, 0.7),
            eps=1e-4,
            weight_decay=0.3,
        )

    parameter = nn.Parameter(torch.arange(8, dtype=torch.float64).reshape(4, 2))
    optimizer = make(parameter)
    parameter.grad = torch.tensor(
        [[1.0, 2.0], [2.0, -3.0], [-1.0, 4.0], [3.0, 1.0]],
        dtype=torch.float64,
    )
    optimizer.step()

    restored_parameter = nn.Parameter(parameter.detach().clone())
    restored = make(restored_parameter)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    next_gradient = torch.tensor(
        [[0.5, -2.0], [3.0, 1.0], [2.0, -1.0], [-4.0, 2.0]],
        dtype=torch.float64,
    )
    parameter.grad = next_gradient
    restored_parameter.grad = next_gradient.clone()

    optimizer.step()
    restored.step()

    torch.testing.assert_close(restored_parameter, parameter)
    original_state = optimizer.state[parameter]
    restored_state = restored.state[restored_parameter]
    assert set(restored_state) == set(original_state)
    for key in original_state:
        torch.testing.assert_close(restored_state[key], original_state[key])


def test_rownorm_two_steps_match_ema_row_normalization_reference() -> None:
    initial = torch.tensor(
        [[1.0, 3.0], [2.0, -1.0], [-4.0, 2.0], [5.0, -2.0]],
        dtype=torch.float64,
    )
    parameter = nn.Parameter(initial.clone())
    optimizer = RowNorm(
        [parameter],
        lr=0.2,
        beta=0.6,
        eps=0.1,
        weight_decay=0.3,
    )
    gradients = (
        torch.tensor(
            [[1.0, 2.0], [2.0, -1.0], [-3.0, 4.0], [5.0, -2.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[-2.0, 3.0], [4.0, 2.0], [1.0, -5.0], [3.0, 1.0]],
            dtype=torch.float64,
        ),
    )
    momentum = torch.zeros_like(parameter)
    expected_parameter = initial.clone()

    for step, gradient in enumerate(gradients, start=1):
        parameter.grad = gradient
        momentum = 0.6 * momentum + 0.4 * gradient
        correction = 1.0 - 0.6**step
        signal = momentum / correction
        update = signal / (
            torch.linalg.vector_norm(signal, dim=1, keepdim=True) + 0.1
        )
        expected_parameter = expected_parameter * (1.0 - 0.2 * 0.3) - 0.2 * update

        optimizer.step()

        torch.testing.assert_close(parameter, expected_parameter)
        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], momentum)
        assert optimizer.state[parameter]["step"] == step
        torch.testing.assert_close(optimizer.lm_head_step(), -0.2 * update)

    state = optimizer.state[parameter]
    assert set(state) == {"momentum_buffer", "step"}
    assert optimizer.param_groups[0]["bias_correction"] is True
    assert "nesterov" not in optimizer.param_groups[0]


def test_rownorm_bias_correction_matches_explicit_corrected_ema() -> None:
    buffer = torch.tensor(
        [[3.0, 4.0], [1e-10, -2e-10], [0.0, 0.0]], dtype=torch.float64
    )
    beta = 0.95
    step = 3
    eps = 1e-8
    correction = 1.0 - beta**step
    corrected = buffer / correction
    expected = corrected / (
        torch.linalg.vector_norm(corrected, dim=1, keepdim=True) + eps
    )

    actual = RowNorm._normalized_ema(
        buffer,
        beta=beta,
        step=step,
        eps=eps,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_rownorm_step_advances_only_when_gradient_is_present() -> None:
    parameter = nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    optimizer = RowNorm(
        [parameter],
        lr=0.2,
        beta=0.6,
        eps=0.1,
        weight_decay=0.0,
    )

    optimizer.step()
    assert parameter not in optimizer.state

    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert optimizer.state[parameter]["step"] == 1

    parameter.grad = None
    optimizer.step()
    assert optimizer.state[parameter]["step"] == 1


def test_rownorm_rejects_legacy_state_without_step() -> None:
    parameter = nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    optimizer = RowNorm(
        [parameter],
        lr=0.2,
        beta=0.6,
        eps=0.1,
        weight_decay=0.0,
    )
    optimizer.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter)
    parameter.grad = torch.ones_like(parameter)

    with pytest.raises(RuntimeError, match="legacy RowNorm optimizer state"):
        optimizer.step()


@pytest.mark.parametrize("nesterov", [False, True])
def test_muon_momentum_update_matches_reference(nesterov: bool) -> None:
    parameter = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    optimizer = Muon(
        [parameter],
        lr=0.1,
        momentum=0.5,
        nesterov=nesterov,
        weight_decay=0.0,
        ns_steps=0,
        eps=0.0,
    )
    first_gradient = torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float64))
    second_gradient = torch.tensor([[0.0, 2.0], [1.0, 0.0]], dtype=torch.float64)
    parameter.grad = first_gradient

    optimizer.step()

    parameter.grad = second_gradient
    optimizer.step()

    first_buffer = first_gradient
    first_direction = first_gradient + 0.5 * first_buffer if nesterov else first_buffer
    second_buffer = 0.5 * first_buffer + second_gradient
    second_direction = (
        second_gradient + 0.5 * second_buffer if nesterov else second_buffer
    )
    expected = -0.1 * (
        first_direction / torch.linalg.vector_norm(first_direction)
        + second_direction / torch.linalg.vector_norm(second_direction)
    )

    torch.testing.assert_close(parameter, expected)
    torch.testing.assert_close(
        optimizer.state[parameter]["momentum_buffer"], second_buffer
    )
    assert optimizer.param_groups[0]["nesterov"] is nesterov
    assert optimizer.state_dict()["param_groups"][0]["nesterov"] is nesterov


def test_muon_rectangular_orientation() -> None:
    parameter = nn.Parameter(torch.eye(2))
    optimizer = Muon(
        [parameter],
        lr=0.1,
        momentum=0.5,
        nesterov=False,
        weight_decay=0.0,
        ns_steps=2,
        eps=1e-6,
    )
    matrix = torch.arange(15, dtype=torch.float32).reshape(5, 3)

    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.5)
    assert optimizer.param_groups[0]["nesterov"] is False
    assert optimizer.param_groups[0]["ns_steps"] == 2
    assert orthogonalize_newton_schulz(matrix, steps=2, eps=1e-6).shape == matrix.shape


def test_muon_applies_algorithm_8_shape_adjustment() -> None:
    parameter = nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))
    gradient = torch.arange(1, 9, dtype=torch.float64).reshape_as(parameter)
    parameter.grad = gradient
    optimizer = Muon(
        [parameter],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        ns_steps=0,
        eps=0.0,
    )

    optimizer.step()

    expected = -gradient / torch.linalg.vector_norm(gradient) * (2**0.5)
    torch.testing.assert_close(parameter, expected)


def test_tied_embedding_and_head_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        build_optimizers(TinyModel(tied=True), "adamw", adamw_config())


def test_factory_preserves_exact_config_error_messages() -> None:
    with pytest.raises(ValueError) as unknown_optimizer:
        build_optimizers(TinyModel(), "lion", adamw_config())
    assert str(unknown_optimizer.value) == "optimizer must be 'adamw' or 'muon', got 'lion'"

    incomplete = adamw_config()
    del incomplete["groups"]["lm_head"]
    with pytest.raises(ValueError) as incomplete_groups:
        build_optimizers(TinyModel(), "adamw", incomplete)
    assert str(incomplete_groups.value) == (
        "AdamW groups must contain exactly "
        "['backbone', 'input_embedding', 'lm_head']; "
        "got ['backbone', 'input_embedding']"
    )

    unknown_head = adamw_config("mystery")
    with pytest.raises(ValueError) as head_algorithm:
        build_optimizers(TinyModel(), "adamw", unknown_head)
    assert str(head_algorithm.value) == (
        "lm_head.algorithm must be 'adamw' or 'rownorm'; "
        "got 'mystery'"
    )

    invalid_nesterov = muon_config()
    invalid_nesterov["backbone"]["nesterov"] = 1
    with pytest.raises(ValueError) as nesterov:
        build_optimizers(TinyModel(), "muon", invalid_nesterov)
    assert str(nesterov.value) == "Muon backbone.nesterov must be a bool"


def test_optimizer_configuration_preserves_exact_factory_metadata() -> None:
    adamw = optimizer_configuration(
        build_optimizers(TinyModel(), "adamw", adamw_config())
    )
    assert adamw == [
        {
            "optimizer": "AdamW",
            "role": "backbone",
            "peak_lr": 0.01,
            "weight_decay": 0.04,
            "betas": [0.7, 0.8],
            "eps": 1e-8,
        },
        {
            "optimizer": "AdamW",
            "role": "input_embedding",
            "peak_lr": 0.02,
            "weight_decay": 0.05,
            "betas": [0.7, 0.8],
            "eps": 2e-8,
        },
        {
            "optimizer": "HeadAdamW",
            "role": "lm_head",
            "peak_lr": 0.03,
            "weight_decay": 0.06,
            "betas": [0.7, 0.8],
            "eps": 3e-8,
        },
    ]

    muon = optimizer_configuration(
        build_optimizers(TinyModel(), "muon", muon_config())
    )
    assert muon == [
        {
            "optimizer": "Muon",
            "role": "backbone",
            "peak_lr": 0.04,
            "weight_decay": 0.06,
            "eps": 1e-8,
            "momentum": 0.5,
            "nesterov": False,
            "ns_steps": 2,
        },
        {
            "optimizer": "AdamW",
            "role": "input_embedding",
            "peak_lr": 0.02,
            "weight_decay": 0.03,
            "betas": [0.6, 0.7],
            "eps": 1e-8,
        },
        {
            "optimizer": "HeadAdamW",
            "role": "lm_head",
            "peak_lr": 0.01,
            "weight_decay": 0.02,
            "betas": [0.6, 0.7],
            "eps": 2e-8,
        },
    ]


@pytest.mark.parametrize("nesterov", [False, True])
def test_muon_state_round_trip_continues_identically(nesterov: bool) -> None:
    parameter = nn.Parameter(torch.arange(8, dtype=torch.float64).reshape(4, 2))
    optimizer = Muon(
        [parameter],
        lr=0.1,
        momentum=0.5,
        nesterov=nesterov,
        weight_decay=0.2,
        ns_steps=2,
        eps=1e-6,
    )
    parameter.grad = torch.tensor(
        [[1.0, 2.0], [2.0, -3.0], [-1.0, 4.0], [3.0, 1.0]],
        dtype=torch.float64,
    )
    optimizer.step()

    restored_parameter = nn.Parameter(parameter.detach().clone())
    restored = Muon(
        [restored_parameter],
        lr=0.1,
        momentum=0.5,
        nesterov=nesterov,
        weight_decay=0.2,
        ns_steps=2,
        eps=1e-6,
    )
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    next_gradient = torch.tensor(
        [[0.5, -2.0], [3.0, 1.0], [2.0, -1.0], [-4.0, 2.0]],
        dtype=torch.float64,
    )
    parameter.grad = next_gradient
    restored_parameter.grad = next_gradient.clone()

    optimizer.step()
    restored.step()

    torch.testing.assert_close(restored_parameter, parameter)
    torch.testing.assert_close(
        restored.state[restored_parameter]["momentum_buffer"],
        optimizer.state[parameter]["momentum_buffer"],
    )
    assert restored.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_optimizer_list_state_round_trip_preserves_positional_contract(
    optimizer_name: str,
) -> None:
    config = adamw_config() if optimizer_name == "adamw" else muon_config()
    model = TinyModel()
    optimizers = build_optimizers(model, optimizer_name, config)
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = (
            torch.arange(1, parameter.numel() + 1, dtype=parameter.dtype).reshape_as(parameter)
            * (index + 1)
        )
    for optimizer in optimizers:
        optimizer.step()

    model_state = copy.deepcopy(model.state_dict())
    optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
    restored_model = TinyModel()
    restored_model.load_state_dict(model_state)
    restored_optimizers = build_optimizers(restored_model, optimizer_name, config)
    assert tuple(type(optimizer) for optimizer in restored_optimizers) == tuple(
        type(optimizer) for optimizer in optimizers
    )
    assert [
        [group["role"] for group in optimizer.param_groups]
        for optimizer in restored_optimizers
    ] == [
        [group["role"] for group in optimizer.param_groups]
        for optimizer in optimizers
    ]
    for optimizer, state in zip(restored_optimizers, optimizer_states, strict=True):
        optimizer.load_state_dict(state)

    for index, (parameter, restored_parameter) in enumerate(
        zip(model.parameters(), restored_model.parameters(), strict=True)
    ):
        gradient = (
            torch.arange(1, parameter.numel() + 1, dtype=parameter.dtype).reshape_as(parameter)
            * (index + 2)
        )
        parameter.grad = gradient
        restored_parameter.grad = gradient.clone()
    for optimizer, restored_optimizer in zip(
        optimizers, restored_optimizers, strict=True
    ):
        optimizer.step()
        restored_optimizer.step()

    for parameter, restored_parameter in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        torch.testing.assert_close(restored_parameter, parameter)
    for optimizer, restored_optimizer in zip(
        optimizers, restored_optimizers, strict=True
    ):
        original_state = optimizer.state_dict()
        restored_state = restored_optimizer.state_dict()
        assert restored_state["param_groups"] == original_state["param_groups"]
        assert set(restored_state["state"]) == set(original_state["state"])
        for parameter_index, values in original_state["state"].items():
            restored_values = restored_state["state"][parameter_index]
            assert set(restored_values) == set(values)
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(restored_values[key], value)
                else:
                    assert restored_values[key] == value

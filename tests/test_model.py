import pytest
import torch

from hilbert_rownorm.config import ModelConfig, model_preset
from hilbert_rownorm.model import FixedRMSNorm, TransformerLM, apply_rotary


@pytest.mark.parametrize(
    ("name", "dim", "parameters"),
    [
        ("190m", 512, 185_680_896),
        ("380m", 768, 379_184_640),
        ("640m", 1_024, 639_797_248),
    ],
)
def test_presets(name: str, dim: int, parameters: int) -> None:
    config = model_preset(name)
    assert (config.dim, config.n_layers, config.head_dim, config.mlp_dim) == (
        dim,
        32,
        64,
        4 * dim,
    )
    assert config.max_seq_len == 2_048
    assert config.initializer_std == 0.02
    assert config.parameter_count == parameters


def test_removed_1p4b_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model preset"):
        model_preset("1p4b")


@pytest.mark.parametrize("initializer_std", [0.0, -0.02, float("nan"), float("inf")])
def test_initializer_std_must_be_finite_and_positive(initializer_std: float) -> None:
    with pytest.raises(ValueError, match="initializer_std"):
        ModelConfig(dim=64, initializer_std=initializer_std)


def tiny_config(**changes: object) -> ModelConfig:
    values = dict(dim=32, n_layers=2, head_dim=8, vocab_size=101, max_seq_len=16)
    values.update(changes)
    return ModelConfig(**values)  # type: ignore[arg-type]


def test_forward_init_and_parameter_ownership() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = TransformerLM(config)
    tokens = torch.randint(config.vocab_size, (2, 7))

    assert model(tokens).shape == (2, 7, config.vocab_size)
    logits = model(tokens)
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(logits) > 0
    assert model.embed.weight is not model.lm_head.weight
    assert model.lm_head.weight.shape == (config.vocab_size, config.dim)
    assert model.num_parameters() == config.parameter_count

    hidden = list(model.hidden_parameters())
    owned = hidden + [model.embed.weight, model.lm_head.weight]
    assert len(hidden) == 7 * config.n_layers
    assert len({id(parameter) for parameter in owned}) == len(owned)
    assert {id(parameter) for parameter in owned} == {id(p) for p in model.parameters()}


def test_tiny_model_state_dict_schema_and_nonpersistent_rope() -> None:
    model = TransformerLM(tiny_config())

    assert tuple(model.state_dict()) == (
        "embed.weight",
        "blocks.0.attn.q.weight",
        "blocks.0.attn.k.weight",
        "blocks.0.attn.v.weight",
        "blocks.0.attn.out.weight",
        "blocks.0.gate.weight",
        "blocks.0.up.weight",
        "blocks.0.down.weight",
        "blocks.1.attn.q.weight",
        "blocks.1.attn.k.weight",
        "blocks.1.attn.v.weight",
        "blocks.1.attn.out.weight",
        "blocks.1.gate.weight",
        "blocks.1.up.weight",
        "blocks.1.down.weight",
        "lm_head.weight",
    )
    assert tuple(dict(model.named_buffers())) == ("rope.cos", "rope.sin")
    assert not model.rope.state_dict()


def test_architecture_and_initialization() -> None:
    torch.manual_seed(11)
    config = tiny_config(dim=64)
    model = TransformerLM(config)
    assert list(FixedRMSNorm(4).parameters()) == []
    matrices = [
        module.weight
        for module in model.modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
    ]
    assert len(matrices) == 7 * config.n_layers + 2
    assert torch.cat([matrix.flatten() for matrix in matrices]).std().item() == pytest.approx(
        config.initializer_std, rel=0.03
    )
    assert all(torch.count_nonzero(matrix) == matrix.numel() for matrix in matrices)
    assert not torch.equal(model.embed.weight, model.lm_head.weight)
    for block in model.blocks:
        assert block.attn.q_norm.normalized_shape == config.head_dim
        assert block.attn.k_norm.normalized_shape == config.head_dim
    linears = (module for module in model.modules() if isinstance(module, torch.nn.Linear))
    assert all(module.bias is None for module in linears)


def test_attention_is_causal() -> None:
    torch.manual_seed(19)
    model = TransformerLM(tiny_config()).eval()
    left = torch.tensor([[1, 2, 3, 4, 5]])
    right = torch.tensor([[1, 2, 3, 8, 9]])
    torch.testing.assert_close(
        model.forward_features(left)[:, :3], model.forward_features(right)[:, :3]
    )


def test_rope_is_half_split_and_sequence_is_bounded() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    rotated = apply_rotary(x, torch.zeros(1, 1, 1, 2), torch.ones(1, 1, 1, 2))
    torch.testing.assert_close(rotated, torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))
    with pytest.raises(ValueError, match="length"):
        TransformerLM(tiny_config(max_seq_len=4))(torch.zeros(1, 5, dtype=torch.long))


def test_fp32_parameters_with_bf16_compute() -> None:
    model = TransformerLM(tiny_config(), compute_dtype=torch.bfloat16)
    tokens = torch.randint(101, (2, 7))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert model.forward_features(tokens).dtype == torch.bfloat16
        assert model(tokens).dtype == torch.bfloat16
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}

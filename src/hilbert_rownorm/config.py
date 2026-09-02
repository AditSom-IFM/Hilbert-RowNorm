"""Model presets for the optimizer comparison experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

VOCAB_SIZE = 50_257
CONTEXT_LENGTH = 2_048

MODEL_WIDTHS = {
    "190m": 512,
    "380m": 768,
    "640m": 1_024,
}


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    n_layers: int = 32
    head_dim: int = 64
    mlp_expansion: int = 4
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = CONTEXT_LENGTH
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    initializer_std: float = 0.02

    def __post_init__(self) -> None:
        sizes = (
            self.dim,
            self.n_layers,
            self.head_dim,
            self.mlp_expansion,
            self.vocab_size,
            self.max_seq_len,
        )
        if any(not isinstance(value, int) or value <= 0 for value in sizes):
            raise ValueError("model sizes must be positive integers")
        if self.dim % self.head_dim:
            raise ValueError("dim must be divisible by head_dim")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if not math.isfinite(self.initializer_std) or self.initializer_std <= 0:
            raise ValueError("initializer_std must be finite and positive")

    @property
    def n_heads(self) -> int:
        return self.dim // self.head_dim

    @property
    def mlp_dim(self) -> int:
        return self.mlp_expansion * self.dim

    @property
    def parameter_count(self) -> int:
        block = (4 + 3 * self.mlp_expansion) * self.dim**2
        return self.n_layers * block + 2 * self.vocab_size * self.dim


def model_preset(name: str) -> ModelConfig:
    key = name.lower().replace(".", "p").replace("-", "").replace("_", "")
    try:
        return ModelConfig(dim=MODEL_WIDTHS[key])
    except KeyError as error:
        raise ValueError(f"unknown model preset {name!r}: choose {tuple(MODEL_WIDTHS)}") from error

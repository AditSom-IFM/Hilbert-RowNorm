"""Llama-style language model used for optimizer comparisons."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class FixedRMSNorm(nn.Module):
    """RMSNorm without a learned gain."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = width
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str | None) -> None:
        super().__init__()
        frequency = config.rope_theta ** (
            -2 * torch.arange(config.head_dim // 2, device=device).float() / config.head_dim
        )
        angles = torch.arange(config.max_seq_len, device=device).float()[:, None] * frequency
        self.register_buffer("cos", angles.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, :, None, :], persistent=False)

    def forward(self, length: int) -> tuple[Tensor, Tensor]:
        return self.cos[:, :length], self.sin[:, :length]


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply half-split (not interleaved) RoPE."""

    left, right = x.chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), -1).to(x.dtype)


class Attention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.n_heads, self.head_dim = config.n_heads, config.head_dim
        self.q = nn.Linear(config.dim, config.dim, bias=False, **factory)
        self.k = nn.Linear(config.dim, config.dim, bias=False, **factory)
        self.v = nn.Linear(config.dim, config.dim, bias=False, **factory)
        self.out = nn.Linear(config.dim, config.dim, bias=False, **factory)
        self.q_norm = FixedRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = FixedRMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, length, width = x.shape
        shape = (batch, length, self.n_heads, self.head_dim)
        q = apply_rotary(self.q_norm(self.q(x).view(shape)), cos, sin).transpose(1, 2)
        k = apply_rotary(self.k_norm(self.k(x).view(shape)), cos, sin).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.attn_norm = FixedRMSNorm(config.dim, config.rms_norm_eps)
        self.attn = Attention(config, device, dtype)
        self.mlp_norm = FixedRMSNorm(config.dim, config.rms_norm_eps)
        self.gate = nn.Linear(config.dim, config.mlp_dim, bias=False, **factory)
        self.up = nn.Linear(config.dim, config.mlp_dim, bias=False, **factory)
        self.down = nn.Linear(config.mlp_dim, config.dim, bias=False, **factory)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        y = self.mlp_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class TransformerLM(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        param_dtype: torch.dtype | None = torch.float32,
        compute_dtype: torch.dtype | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.gradient_checkpointing = gradient_checkpointing
        factory = {"device": device, "dtype": param_dtype}
        self.embed = nn.Embedding(config.vocab_size, config.dim, **factory)
        self.rope = RotaryEmbedding(config, device)
        self.blocks = nn.ModuleList(
            Block(config, device, param_dtype) for _ in range(config.n_layers)
        )
        self.out_norm = FixedRMSNorm(config.dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False, **factory)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)

    def forward_features(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(f"expected [batch, length <= {self.config.max_seq_len}] tokens")
        x = self.embed(tokens)
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
        cos, sin = self.rope(tokens.shape[1])
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        return x

    def forward(
        self, tokens: Tensor, *, return_hidden: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        hidden = self.out_norm(self.forward_features(tokens))
        logits = self.lm_head(hidden)
        return (logits, hidden) if return_hidden else logits

    def hidden_parameters(self) -> Iterator[nn.Parameter]:
        """Matrices owned by Muon in a Muon-backbone recipe."""

        yield from self.blocks.parameters()

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

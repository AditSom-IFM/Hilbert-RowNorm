"""Strict optimizer configuration and semantic parameter routing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn
from torch.optim import AdamW, Optimizer

from .head_optimizers import HeadAdamW, RowNorm
from .muon import Muon


def _name(name: str) -> str:
    value = name.lower().replace("-", "").replace("_", "")
    if value in {"adam", "adamw"}:
        return "adamw"
    if value == "muon":
        return "muon"
    raise ValueError(f"optimizer must be 'adamw' or 'muon', got {name!r}")


def _split_parameters(model: nn.Module) -> tuple[tuple[nn.Parameter, ...], ...]:
    try:
        hidden = tuple(model.hidden_parameters())  # type: ignore[attr-defined]
        groups = (hidden, (model.embed.weight,), (model.lm_head.weight,))  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError("model must expose hidden_parameters(), embed, and lm_head") from error

    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    role_ids = [id(parameter) for group in groups for parameter in group]
    trainable_ids = [id(parameter) for parameter in trainable]
    if len(role_ids) != len(set(role_ids)):
        raise ValueError("optimizer parameter roles overlap (is the LM head tied?)")
    if set(role_ids) != set(trainable_ids):
        raise ValueError("optimizer parameter roles must cover every trainable parameter")
    return groups


def _require_keys(config: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(config) != expected:
        raise ValueError(
            f"{label} must contain exactly {sorted(expected)}; got {sorted(config)}"
        )


def _number(
    config: Mapping[str, Any], key: str, label: str, *, positive: bool = False
) -> float:
    value = float(config[key])
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label}.{key} must be finite and {qualifier}")
    return value


def _betas(config: Any, label: str) -> tuple[float, float]:
    if not isinstance(config, Sequence) or isinstance(config, (str, bytes)) or len(config) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    values = tuple(float(value) for value in config)
    if any(not math.isfinite(value) or not 0 <= value < 1 for value in values):
        raise ValueError(f"{label} values must be finite and in [0, 1)")
    return values  # type: ignore[return-value]


def _adamw_group(
    role: str, params: Sequence[nn.Parameter], config: Mapping[str, Any]
) -> dict[str, Any]:
    _require_keys(config, {"lr", "eps", "weight_decay"}, f"{role} config")
    lr = _number(config, "lr", role, positive=True)
    return {
        "params": list(params),
        "role": role,
        "lr": lr,
        "base_lr": lr,
        "eps": _number(config, "eps", role, positive=True),
        "weight_decay": _number(config, "weight_decay", role),
    }


def _head_optimizer(
    params: Sequence[nn.Parameter],
    config: Mapping[str, Any],
    betas: tuple[float, float],
    label: str,
) -> Optimizer:
    if not isinstance(config, Mapping):
        raise ValueError(f"{label} config must be a mapping")
    algorithm = config.get("algorithm")
    if algorithm == "adamw":
        _require_keys(config, {"algorithm", "lr", "eps", "weight_decay"}, label)
        return HeadAdamW(
            params,
            lr=_number(config, "lr", label, positive=True),
            betas=betas,
            eps=_number(config, "eps", label, positive=True),
            weight_decay=_number(config, "weight_decay", label),
        )
    if algorithm == "rownorm":
        _require_keys(
            config,
            {"algorithm", "lr", "beta", "eps", "weight_decay"},
            label,
        )
        beta = _number(config, "beta", label)
        if beta >= 1:
            raise ValueError(f"{label}.beta must be in [0, 1)")
        return RowNorm(
            params,
            lr=_number(config, "lr", label, positive=True),
            beta=beta,
            eps=_number(config, "eps", label, positive=True),
            weight_decay=_number(config, "weight_decay", label),
        )
    raise ValueError(
        f"{label}.algorithm must be 'adamw' or 'rownorm'; "
        f"got {algorithm!r}"
    )


def build_optimizers(
    model: nn.Module,
    optimizer_name: str,
    config: Mapping[str, Any],
) -> list[Optimizer]:
    """Build the backbone, embedding, and LM-head optimizers for one recipe."""

    name = _name(optimizer_name)
    hidden, embedding, head = _split_parameters(model)

    if name == "adamw":
        _require_keys(config, {"betas", "groups"}, "AdamW config")
        groups_config = config["groups"]
        if not isinstance(groups_config, Mapping):
            raise ValueError("AdamW groups must be a mapping")
        _require_keys(
            groups_config, {"backbone", "input_embedding", "lm_head"}, "AdamW groups"
        )
        betas = _betas(config["betas"], "betas")
        groups = [
            _adamw_group("backbone", hidden, groups_config["backbone"]),
            _adamw_group("input_embedding", embedding, groups_config["input_embedding"]),
        ]
        optimizers: list[Optimizer] = [
            AdamW(groups, betas=betas),
            _head_optimizer(head, groups_config["lm_head"], betas, "lm_head"),
        ]
    else:
        _require_keys(config, {"backbone", "auxiliary_adamw"}, "Muon config")
        backbone = config["backbone"]
        auxiliary = config["auxiliary_adamw"]
        if not isinstance(backbone, Mapping) or not isinstance(auxiliary, Mapping):
            raise ValueError("Muon backbone and auxiliary_adamw must be mappings")
        _require_keys(
            backbone,
            {"lr", "momentum", "nesterov", "weight_decay", "ns_steps", "eps"},
            "Muon backbone",
        )
        _require_keys(auxiliary, {"betas", "groups"}, "Muon auxiliary AdamW")
        ns_steps = backbone["ns_steps"]
        if not isinstance(ns_steps, int) or ns_steps < 0:
            raise ValueError("Muon backbone.ns_steps must be a non-negative integer")
        momentum = _number(backbone, "momentum", "Muon backbone")
        if momentum >= 1:
            raise ValueError("Muon backbone.momentum must be in [0, 1)")
        nesterov = backbone["nesterov"]
        if not isinstance(nesterov, bool):
            raise ValueError("Muon backbone.nesterov must be a bool")
        muon = Muon(
            hidden,
            lr=_number(backbone, "lr", "Muon backbone", positive=True),
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=_number(backbone, "weight_decay", "Muon backbone"),
            ns_steps=ns_steps,
            eps=_number(backbone, "eps", "Muon backbone", positive=True),
        )
        groups_config = auxiliary["groups"]
        if not isinstance(groups_config, Mapping):
            raise ValueError("Muon auxiliary AdamW groups must be a mapping")
        _require_keys(
            groups_config,
            {"input_embedding", "lm_head"},
            "Muon auxiliary AdamW groups",
        )
        betas = _betas(auxiliary["betas"], "auxiliary_adamw.betas")
        groups = [_adamw_group("input_embedding", embedding, groups_config["input_embedding"])]
        optimizers = [
            muon,
            AdamW(groups, betas=betas),
            _head_optimizer(
                head,
                groups_config["lm_head"],
                betas,
                "auxiliary_adamw.lm_head",
            ),
        ]
    return optimizers


def set_lr_scale(optimizers: Sequence[Optimizer], scale: float) -> None:
    """Apply the common warmup/decay multiplier to every parameter group."""

    if not math.isfinite(scale) or scale < 0:
        raise ValueError("learning-rate scale must be finite and non-negative")
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * scale

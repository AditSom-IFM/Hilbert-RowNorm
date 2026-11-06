"""Paper recipes, the supported AdamW-backbone comparisons, and sweep variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptimizerRecipe:
    """A stable recipe identifier and its fully resolved optimizer config."""

    identifier: str
    config: dict[str, Any]


_ADAMW_LRS = {
    "190m": {
        "backbone": 0.002,
        "input_embedding": 0.002,
        "lm_head": 0.002,
    },
    "380m": {
        "backbone": 0.0013333333333333333,
        "input_embedding": 0.002,
        "lm_head": 0.0013333333333333333,
    },
}

_MUON_HEAD_LRS = {
    "190m": {"adamw": 0.004, "rownorm": 0.0032},
    "380m": {
        "adamw": 0.0026666666666666666,
        "rownorm": 0.0021333333333333333,
    },
    "640m": {"adamw": 0.002, "rownorm": 0.0016},
}

# Compositional Muon on query/key pairs. Learning rate, momentum, Nesterov,
# weight decay, and Newton-Schulz settings are inherited from the Muon
# backbone; mp=2 cancels the half-split budget at initialization.
COMPOSITIONAL_QK_DEFAULTS = {"mp": 2.0, "damping": 1e-2, "whitening": "isotropic"}

# LM-head keys a sweep may override, per head algorithm.
HEAD_OVERRIDE_KEYS = {
    "adamw": ("lr", "betas", "weight_decay"),
    "rownorm": ("lr", "beta", "weight_decay"),
}


def _apply_head_overrides(
    head: dict[str, Any], overrides: Mapping[str, Any]
) -> None:
    allowed = HEAD_OVERRIDE_KEYS[head["algorithm"]]
    unknown = sorted(set(overrides) - set(allowed))
    if unknown:
        raise ValueError(
            f"{head['algorithm']} LM head cannot override {unknown}; choose from {allowed}"
        )
    for key, value in overrides.items():
        head[key] = [float(v) for v in value] if key == "betas" else float(value)


def _canonical_model_name(name: str) -> str:
    return name.lower().replace(".", "p").replace("-", "").replace("_", "")


def _adamw_recipe(model: str) -> dict[str, Any]:
    try:
        learning_rates = _ADAMW_LRS[model]
    except KeyError as error:
        raise ValueError(
            f"no final AdamW-backbone optimizer recipe exists for model {model!r}"
        ) from error

    def group(role: str) -> dict[str, float | str]:
        values: dict[str, float | str] = {
            "lr": learning_rates[role],
            "eps": 1e-10,
            "weight_decay": 0.1,
        }
        if role == "lm_head":
            values = {"algorithm": "adamw", **values}
        return values

    return {
        "betas": [0.9, 0.999],
        "groups": {
            "backbone": group("backbone"),
            "input_embedding": group("input_embedding"),
            "lm_head": group("lm_head"),
        },
    }


def _muon_recipe(model: str, lm_head_optimizer: str) -> dict[str, Any]:
    try:
        head_lr = _MUON_HEAD_LRS[model][lm_head_optimizer]
    except KeyError as error:
        raise ValueError(
            "no final Muon optimizer recipe exists for "
            f"model {model!r} with LM-head optimizer {lm_head_optimizer!r}"
        ) from error

    if lm_head_optimizer == "adamw":
        head: dict[str, Any] = {
            "algorithm": "adamw",
            "lr": head_lr,
            "eps": 1e-10,
            "weight_decay": 0.1,
        }
    else:
        head = {
            "algorithm": "rownorm",
            "lr": head_lr,
            "beta": 0.95,
            "eps": 1e-8,
            "weight_decay": 0.0,
        }

    return {
        "backbone": {
            "lr": 0.008,
            "momentum": 0.95,
            "nesterov": True,
            "weight_decay": 0.1,
            "ns_steps": 5,
            "eps": 1e-5,
        },
        "auxiliary_adamw": {
            "betas": [0.9, 0.999],
            "groups": {
                "input_embedding": {
                    "lr": 0.004,
                    "eps": 1e-10,
                    "weight_decay": 0.1,
                },
                "lm_head": head,
            },
        },
    }


def resolve_optimizer_recipe(
    model: str,
    backbone_optimizer: str,
    lm_head_optimizer: str,
    *,
    head_overrides: Mapping[str, Any] | None = None,
    attention_qk: str = "muon",
    attention_qk_overrides: Mapping[str, Any] | None = None,
) -> OptimizerRecipe:
    """Resolve a paper recipe or a supported AdamW-backbone comparison.

    ``head_overrides`` replaces LM-head values for hyperparameter sweeps. Any
    override, like ``attention_qk="compositional"`` (which moves query/key
    pairs from Muon to Compositional Muon), turns the identifier family into
    ``sweep``, so paper identifiers
    always denote the exact paper configuration; the resolved values are
    recorded in the run configuration.
    """

    model_name = _canonical_model_name(model)
    backbone = backbone_optimizer.lower()
    head = lm_head_optimizer.lower()

    if backbone not in {"adamw", "muon"}:
        raise ValueError(
            f"backbone optimizer must be 'adamw' or 'muon', got {backbone_optimizer!r}"
        )
    if head not in {"adamw", "rownorm"}:
        raise ValueError(
            f"LM-head optimizer must be 'adamw' or 'rownorm', got {lm_head_optimizer!r}"
        )
    if backbone == "adamw" and head != "adamw":
        raise ValueError("final AdamW-backbone recipes require the AdamW LM head")
    if attention_qk not in ("muon", "compositional"):
        raise ValueError(
            f"attention QK optimizer must be 'muon' or 'compositional', got {attention_qk!r}"
        )
    if attention_qk_overrides and attention_qk != "compositional":
        raise ValueError("attention_qk_overrides require attention_qk='compositional'")
    if attention_qk == "compositional" and backbone != "muon":
        raise ValueError("Compositional Muon on query/key requires the Muon backbone")

    config = (
        _adamw_recipe(model_name)
        if backbone == "adamw"
        else _muon_recipe(model_name, head)
    )
    family = "paper" if backbone == "muon" else "adamw-backbone"
    if head_overrides:
        head_config = (
            config["groups"]["lm_head"]
            if backbone == "adamw"
            else config["auxiliary_adamw"]["groups"]["lm_head"]
        )
        _apply_head_overrides(head_config, head_overrides)
        family = "sweep"
    if attention_qk == "compositional":
        values = {**COMPOSITIONAL_QK_DEFAULTS, **(attention_qk_overrides or {})}
        unknown = sorted(set(values) - set(COMPOSITIONAL_QK_DEFAULTS))
        if unknown:
            raise ValueError(f"compositional attention_qk cannot override {unknown}")
        backbone_config = config["backbone"]
        config["attention_qk"] = {
            "algorithm": "compositional",
            **{
                key: backbone_config[key]
                for key in ("lr", "momentum", "nesterov", "weight_decay", "ns_steps", "eps")
            },
            "mp": float(values["mp"]),
            "damping": float(values["damping"]),
            "whitening": values["whitening"],
        }
        family = "sweep"
    return OptimizerRecipe(
        identifier=f"{family}:{model_name}:{backbone}:{head}", config=config
    )

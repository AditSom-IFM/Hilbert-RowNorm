"""Paper recipes and the supported AdamW-backbone comparisons."""

from __future__ import annotations

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
) -> OptimizerRecipe:
    """Resolve a paper recipe or a supported AdamW-backbone comparison."""

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

    config = (
        _adamw_recipe(model_name)
        if backbone == "adamw"
        else _muon_recipe(model_name, head)
    )
    family = "paper" if backbone == "muon" else "adamw-backbone"
    return OptimizerRecipe(
        identifier=f"{family}:{model_name}:{backbone}:{head}", config=config
    )

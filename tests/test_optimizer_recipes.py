from __future__ import annotations

from pathlib import Path

import pytest

from hilbert_rownorm.optimizer_recipes import resolve_optimizer_recipe

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_RECIPES = {
    ("190m", "adamw", "adamw"): {
        "betas": [0.9, 0.999],
        "groups": {
            "backbone": {"lr": 0.002, "eps": 1e-10, "weight_decay": 0.1},
            "input_embedding": {
                "lr": 0.002,
                "eps": 1e-10,
                "weight_decay": 0.1,
            },
            "lm_head": {
                "algorithm": "adamw",
                "lr": 0.002,
                "eps": 1e-10,
                "weight_decay": 0.1,
            },
        },
    },
    ("380m", "adamw", "adamw"): {
        "betas": [0.9, 0.999],
        "groups": {
            "backbone": {
                "lr": 0.0013333333333333333,
                "eps": 1e-10,
                "weight_decay": 0.1,
            },
            "input_embedding": {
                "lr": 0.002,
                "eps": 1e-10,
                "weight_decay": 0.1,
            },
            "lm_head": {
                "algorithm": "adamw",
                "lr": 0.0013333333333333333,
                "eps": 1e-10,
                "weight_decay": 0.1,
            },
        },
    },
    ("190m", "muon", "adamw"): {
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
                "lm_head": {
                    "algorithm": "adamw",
                    "lr": 0.004,
                    "eps": 1e-10,
                    "weight_decay": 0.1,
                },
            },
        },
    },
    ("190m", "muon", "rownorm"): {
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
                "lm_head": {
                    "algorithm": "rownorm",
                    "lr": 0.0032,
                    "beta": 0.95,
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            },
        },
    },
    ("380m", "muon", "adamw"): {
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
                "lm_head": {
                    "algorithm": "adamw",
                    "lr": 0.0026666666666666666,
                    "eps": 1e-10,
                    "weight_decay": 0.1,
                },
            },
        },
    },
    ("380m", "muon", "rownorm"): {
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
                "lm_head": {
                    "algorithm": "rownorm",
                    "lr": 0.0021333333333333333,
                    "beta": 0.95,
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            },
        },
    },
    ("640m", "muon", "adamw"): {
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
                "lm_head": {
                    "algorithm": "adamw",
                    "lr": 0.002,
                    "eps": 1e-10,
                    "weight_decay": 0.1,
                },
            },
        },
    },
    ("640m", "muon", "rownorm"): {
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
                "lm_head": {
                    "algorithm": "rownorm",
                    "lr": 0.0016,
                    "beta": 0.95,
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            },
        },
    },
}


@pytest.mark.parametrize(("key", "expected"), EXPECTED_RECIPES.items())
def test_final_optimizer_recipes_are_exact(
    key: tuple[str, str, str], expected: dict[str, object]
) -> None:
    model, backbone, head = key

    recipe = resolve_optimizer_recipe(model, backbone, head)

    family = "paper" if backbone == "muon" else "adamw-backbone"
    assert recipe.identifier == f"{family}:{model}:{backbone}:{head}"
    assert recipe.config == expected


def test_recipe_resolution_returns_fresh_nested_configs() -> None:
    first = resolve_optimizer_recipe("190m", "muon", "adamw")
    first.config["backbone"]["lr"] = 1.0

    second = resolve_optimizer_recipe("190m", "muon", "adamw")

    assert second.config == EXPECTED_RECIPES[("190m", "muon", "adamw")]


@pytest.mark.parametrize(
    ("model", "backbone", "head", "message"),
    [
        ("640m", "adamw", "adamw", "no final AdamW-backbone"),
        ("190m", "adamw", "rownorm", "require the AdamW LM head"),
        ("1p4b", "muon", "adamw", "no final Muon optimizer recipe"),
        ("190m", "lion", "adamw", "backbone optimizer"),
        ("190m", "muon", "mystery", "LM-head optimizer"),
    ],
)
def test_unsupported_recipe_combinations_fail_clearly(
    model: str, backbone: str, head: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_optimizer_recipe(model, backbone, head)


def test_model_aliases_share_the_stable_canonical_identifier() -> None:
    recipe = resolve_optimizer_recipe("190-M", "MUON", "ROWNORM")

    assert recipe.identifier == "paper:190m:muon:rownorm"
    assert recipe.config == EXPECTED_RECIPES[("190m", "muon", "rownorm")]


def test_legacy_json_optimizer_recipes_are_absent() -> None:
    assert not list((ROOT / "configs" / "optimizer").glob("*.json"))

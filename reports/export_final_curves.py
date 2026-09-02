"""Export paper curves and verify the sanitized W&B histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path(__file__).with_name("final_runs.json")
DEFAULT_OUTPUT = Path(__file__).with_name("data") / "final_curves.json"

SCHEMA_VERSION = 3
PUBLIC_WANDB_PROJECT = "mbzuai-llm/Hilbert-RowNorm"
TOKENS_PER_UPDATE = 524_288
EVAL_EVERY = 500
DIAMETER_EVERY = 10
HILBERT_TOKENS = 8_192
WORLD_SIZE = 8
EVAL_TOKENS = 5_242_880
SCAN_PAGE_SIZE = 10_000
METHODS = ("adamw", "rownorm")
SEEDS = (0, 1, 2)
DATASET_METADATA_SHA256 = "87edf1844351c445e687ea865215b5fecff3a8b8052dff5b7263ea30712376f4"
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "190m": {
        "dim": 512,
        "parameter_count": 185_680_896,
        "micro_batch": 32,
        "accumulation_steps": 1,
        "total_updates": 7_083,
        "head_lr": {"adamw": 0.004, "rownorm": 0.0032},
    },
    "380m": {
        "dim": 768,
        "parameter_count": 379_184_640,
        "micro_batch": 16,
        "accumulation_steps": 2,
        "total_updates": 14_464,
        "head_lr": {
            "adamw": 0.0026666666666666666,
            "rownorm": 0.0021333333333333333,
        },
    },
    "640m": {
        "dim": 1_024,
        "parameter_count": 639_797_248,
        "micro_batch": 8,
        "accumulation_steps": 4,
        "total_updates": 24_406,
        "head_lr": {"adamw": 0.002, "rownorm": 0.0016},
    },
}

VALIDATION_KEYS = (
    "progress/update",
    "progress/tokens",
    "validation/loss",
    "validation/hilbert_step/rms",
    "validation/hilbert_step/tokens",
)
DIAMETER_KEYS = (
    "progress/update",
    "progress/tokens",
    "diagnostics/lm_head_step/diameter_exact",
)
PAPER_HISTORY_KEYS = set(VALIDATION_KEYS) | set(DIAMETER_KEYS)
PANEL_HISTORY_KEYS = {
    "train/loss",
    "train/perplexity",
    "train/grad_norm",
    "train/grad_norm_max",
    "train/clipped_fraction",
    "train/would_clip_fraction",
    "optimizer/lr_backbone",
    "optimizer/lr_input_embedding",
    "optimizer/lr_lm_head",
    "optimizer/peak_lr_backbone",
    "optimizer/peak_lr_input_embedding",
    "optimizer/peak_lr_lm_head",
    "optimizer/schedule_scale",
}
LIVE_HISTORY_KEYS = PAPER_HISTORY_KEYS | PANEL_HISTORY_KEYS
PAPER_VALUE_KEYS = PAPER_HISTORY_KEYS - {"progress/update", "progress/tokens"}
PANEL_QUERY_KEYS = (
    "progress/update",
    "progress/tokens",
    *sorted(PANEL_HISTORY_KEYS),
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the verified scalar data used by the manuscript plots."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def expected_updates(total: int, interval: int) -> list[int]:
    updates = list(range(interval, total + 1, interval))
    if not updates or updates[-1] != total:
        updates.append(total)
    return updates


def expected_run_id(model: str, method: str, seed: int) -> str:
    return f"hrn{model[:-1]}{method[0]}{seed}"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _mapping_from_items(value: Any, label: str) -> dict[str, Any]:
    """Normalize W&B mapping-like objects without weakening JSON validation."""

    if isinstance(value, Mapping):
        return dict(value)
    items = getattr(value, "items", None)
    if not callable(items):
        raise ValueError(f"{label} must be an object")
    try:
        return dict(items())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an object") from error


def _require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


def _require_positive_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and positive") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _require_nonnegative_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and nonnegative") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def metric_history_sha256(validation: list[dict[str, Any]], diameter: list[dict[str, Any]]) -> str:
    """Hash the canonical merged metric history published to W&B."""

    merged = {int(row["progress/update"]): dict(row) for row in diameter}
    for validation_row in validation:
        update = int(validation_row["progress/update"])
        row = merged.setdefault(update, {})
        if row and row.get("progress/tokens") != validation_row["progress/tokens"]:
            raise ValueError(f"history token mismatch at update {update}")
        row.update(validation_row)
    return _canonical_sha256([merged[update] for update in sorted(merged)])


def validate_manifest(payload: Any) -> dict[str, Any]:
    manifest = dict(_require_mapping(payload, "manifest"))
    if set(manifest) != {"schema_version", "project", "runs"}:
        raise ValueError("manifest contains unexpected fields")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported final-run manifest schema")
    if manifest.get("project") != PUBLIC_WANDB_PROJECT:
        raise ValueError(f"manifest project must be {PUBLIC_WANDB_PROJECT}")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("manifest runs must be a list")

    expected_cells = {
        (model, method, seed) for model in MODEL_SPECS for method in METHODS for seed in SEEDS
    }
    runs: list[dict[str, Any]] = []
    cells: set[tuple[str, str, int]] = set()
    run_ids: set[str] = set()
    for index, raw_entry in enumerate(raw_runs):
        entry = dict(_require_mapping(raw_entry, f"manifest run {index}"))
        expected_fields = {
            "model",
            "method",
            "seed",
            "run_id",
            "metric_history_sha256",
        }
        if set(entry) != expected_fields:
            raise ValueError(f"manifest run {index} contains unexpected fields")
        model = str(entry.get("model"))
        method = str(entry.get("method"))
        seed = _require_integer(entry.get("seed"), f"manifest run {index} seed")
        run_id = entry.get("run_id")
        history_sha256 = entry.get("metric_history_sha256")
        cell = (model, method, seed)
        if cell not in expected_cells:
            raise ValueError(f"manifest run {index} has an unsupported experiment cell")
        if cell in cells:
            raise ValueError(f"duplicate manifest cell: {cell}")
        if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(f"manifest run {index} has an invalid W&B run ID")
        if run_id != expected_run_id(model, method, seed):
            raise ValueError(f"manifest run {index} has a noncanonical W&B run ID")
        if run_id in run_ids:
            raise ValueError(f"duplicate W&B run ID: {run_id}")
        if not isinstance(history_sha256, str) or SHA256_PATTERN.fullmatch(history_sha256) is None:
            raise ValueError(f"manifest run {index} has an invalid history checksum")
        cells.add(cell)
        run_ids.add(run_id)
        runs.append(
            {
                "model": model,
                "method": method,
                "seed": seed,
                "run_id": run_id,
                "metric_history_sha256": history_sha256,
            }
        )

    if cells != expected_cells:
        missing = sorted(expected_cells - cells)
        extra = sorted(cells - expected_cells)
        raise ValueError(
            f"manifest must contain the exact 3 x 2 x 3 matrix; missing={missing}, extra={extra}"
        )
    model_order = {model: index for index, model in enumerate(MODEL_SPECS)}
    method_order = {method: index for index, method in enumerate(METHODS)}
    runs.sort(
        key=lambda run: (
            model_order[run["model"]],
            method_order[run["method"]],
            run["seed"],
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "project": PUBLIC_WANDB_PROJECT,
        "runs": runs,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    return validate_manifest(json.loads(path.read_text()))


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Return a stable digest of a validated final-run manifest."""

    return _canonical_sha256(validate_manifest(manifest))


def _validate_history_grid(
    rows: list[dict[str, Any]],
    *,
    model: str,
    method: str,
    seed: int,
    interval: int,
    value_key: str,
) -> None:
    expected = expected_updates(int(MODEL_SPECS[model]["total_updates"]), interval)
    observed = [int(row["progress/update"]) for row in rows]
    label = f"{model}/{method}/seed{seed}"
    if observed != expected:
        raise ValueError(f"{label}: incomplete {value_key} history")
    for row in rows:
        update = int(row["progress/update"])
        if int(row["progress/tokens"]) != update * TOKENS_PER_UPDATE:
            raise ValueError(f"{label}: token count does not match update")


def _declared_history_key_counts(
    run: Any, model: str, method: str, seed: int
) -> dict[str, int]:
    """Validate W&B's unsampled history manifest and return public key counts."""

    label = f"{model}/{method}/seed{seed}"
    attributes = _require_mapping(getattr(run, "_attrs", None), f"{label} run attributes")
    history_keys = _require_mapping(
        attributes.get("historyKeys"), f"{label} history-key manifest"
    )
    raw_keys = _require_mapping(history_keys.get("keys"), f"{label} history keys")
    public_keys = {str(key) for key in raw_keys if not str(key).startswith("_")}
    if public_keys != LIVE_HISTORY_KEYS:
        missing = sorted(LIVE_HISTORY_KEYS - public_keys)
        unexpected = sorted(public_keys - LIVE_HISTORY_KEYS)
        raise ValueError(
            f"{label}: live history allowlist mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    counts: dict[str, int] = {}
    for key in LIVE_HISTORY_KEYS:
        description = _require_mapping(raw_keys[key], f"{label} history key {key}")
        type_counts = description.get("typeCounts")
        if not isinstance(type_counts, list) or not type_counts:
            raise ValueError(f"{label}: history key {key} has no type counts")
        count = 0
        for index, raw_type_count in enumerate(type_counts):
            type_count = _require_mapping(
                raw_type_count, f"{label} history key {key} type count {index}"
            )
            if type_count.get("type") != "number":
                raise ValueError(f"{label}: history key {key} is not numeric")
            count += _require_integer(type_count.get("count"), f"{label} {key} count")
        counts[key] = count

    diameter_count = len(
        expected_updates(int(MODEL_SPECS[model]["total_updates"]), DIAMETER_EVERY)
    )
    validation_count = len(
        expected_updates(int(MODEL_SPECS[model]["total_updates"]), EVAL_EVERY)
    )
    expected_counts = {
        "progress/update": 2 * diameter_count,
        "progress/tokens": 2 * diameter_count,
        "diagnostics/lm_head_step/diameter_exact": diameter_count,
        "validation/loss": validation_count,
        "validation/hilbert_step/rms": validation_count,
        "validation/hilbert_step/tokens": validation_count,
        **{key: diameter_count for key in PANEL_HISTORY_KEYS},
    }
    if counts != expected_counts:
        changed = {
            key: (counts.get(key), expected_counts[key])
            for key in expected_counts
            if counts.get(key) != expected_counts[key]
        }
        raise ValueError(f"{label}: unexpected live history counts {changed}")

    line_count = _require_integer(
        attributes.get("historyLineCount"), f"{label} history line count"
    )
    if line_count != 2 * diameter_count:
        raise ValueError(f"{label}: unexpected history line count")
    last_step = _require_integer(history_keys.get("lastStep"), f"{label} last history step")
    if last_step != line_count - 1:
        raise ValueError(f"{label}: unexpected last history step")
    return counts


def _history_rows(
    run: Any, model: str, method: str, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    label = f"{model}/{method}/seed{seed}"
    _declared_history_key_counts(run, model, method, seed)
    by_update: dict[int, dict[str, Any]] = {}
    # The unspecialized scan reliably exposes the original paper-metric
    # fragment. Exact duplicate rows can appear while W&B compacts a resumed
    # run, so duplicates are accepted only when every public value agrees.
    for raw_row in run.scan_history(page_size=SCAN_PAGE_SIZE):
        raw = dict(raw_row)
        present = {key: raw[key] for key in PAPER_HISTORY_KEYS if raw.get(key) is not None}
        if not (set(present) & PAPER_VALUE_KEYS):
            continue
        if not {"progress/update", "progress/tokens"} <= set(present):
            raise ValueError(f"{label}: history row is missing its progress coordinate")
        update = _require_integer(present["progress/update"], "history update")
        row: dict[str, Any] = {
            "progress/update": update,
            "progress/tokens": _require_integer(present["progress/tokens"], "history tokens"),
        }
        if "diagnostics/lm_head_step/diameter_exact" in present:
            row["diagnostics/lm_head_step/diameter_exact"] = _require_positive_float(
                present["diagnostics/lm_head_step/diameter_exact"], "exact diameter"
            )
        validation_present = set(present) & set(VALIDATION_KEYS[2:])
        if validation_present:
            if validation_present != set(VALIDATION_KEYS[2:]):
                raise ValueError(f"{label}: incomplete validation metrics at update {update}")
            row["validation/loss"] = _require_positive_float(
                present["validation/loss"], "validation loss"
            )
            row["validation/hilbert_step/rms"] = _require_positive_float(
                present["validation/hilbert_step/rms"], "Hilbert RMS"
            )
            panel_tokens = _require_integer(
                present["validation/hilbert_step/tokens"], "Hilbert panel size"
            )
            if panel_tokens != HILBERT_TOKENS:
                raise ValueError(f"{label}: unexpected Hilbert panel size")
            row["validation/hilbert_step/tokens"] = panel_tokens
        previous_row = by_update.get(update)
        if previous_row is not None:
            if row != previous_row:
                raise ValueError(f"{label}: conflicting history update {update}")
        else:
            by_update[update] = row

    merged = [by_update[update] for update in sorted(by_update)]
    diameter = [
        {key: row[key] for key in DIAMETER_KEYS}
        for row in merged
        if "diagnostics/lm_head_step/diameter_exact" in row
    ]
    validation = [
        {key: row[key] for key in VALIDATION_KEYS} for row in merged if "validation/loss" in row
    ]
    _validate_history_grid(
        diameter,
        model=model,
        method=method,
        seed=seed,
        interval=DIAMETER_EVERY,
        value_key="diagnostics/lm_head_step/diameter_exact",
    )
    _validate_history_grid(
        validation,
        model=model,
        method=method,
        seed=seed,
        interval=EVAL_EVERY,
        value_key="validation/loss",
    )
    expected_panel_count = len(
        expected_updates(int(MODEL_SPECS[model]["total_updates"]), DIAMETER_EVERY)
    )
    raw_panel = run.history(
        samples=expected_panel_count,
        keys=list(PANEL_QUERY_KEYS),
        pandas=False,
    )
    if not isinstance(raw_panel, list):
        raise ValueError(f"{label}: dashboard history must be a list")
    panel_by_update: dict[int, dict[str, Any]] = {}
    for index, raw_panel_row in enumerate(raw_panel):
        raw = dict(_require_mapping(raw_panel_row, f"{label} dashboard row {index}"))
        present = {key: raw[key] for key in PANEL_HISTORY_KEYS if raw.get(key) is not None}
        if set(present) != PANEL_HISTORY_KEYS:
            missing = sorted(PANEL_HISTORY_KEYS - set(present))
            raise ValueError(f"{label}: incomplete dashboard metrics; missing={missing}")
        if raw.get("progress/update") is None or raw.get("progress/tokens") is None:
            raise ValueError(f"{label}: dashboard row is missing its progress coordinate")
        update = _require_integer(raw["progress/update"], "dashboard update")
        row = {
            "progress/update": update,
            "progress/tokens": _require_integer(raw["progress/tokens"], "dashboard tokens"),
            **{
                key: _require_nonnegative_float(value, key)
                for key, value in present.items()
            },
        }
        previous_row = panel_by_update.get(update)
        if previous_row is not None:
            if row != previous_row:
                raise ValueError(f"{label}: conflicting dashboard update {update}")
        else:
            panel_by_update[update] = row
    panel = [panel_by_update[update] for update in sorted(panel_by_update)]
    _validate_history_grid(
        panel,
        model=model,
        method=method,
        seed=seed,
        interval=DIAMETER_EVERY,
        value_key="dashboard metrics",
    )
    return validation, diameter, metric_history_sha256(validation, diameter)


def expected_optimizer_groups(model: str, method: str) -> dict[str, dict[str, Any]]:
    head: dict[str, Any] = {
        "optimizer": "HeadAdamW",
        "role": "lm_head",
        "peak_lr": MODEL_SPECS[model]["head_lr"][method],
        "weight_decay": 0.1,
        "betas": [0.9, 0.999],
        "eps": 1e-10,
    }
    if method == "rownorm":
        head = {
            "optimizer": "RowNorm",
            "role": "lm_head",
            "peak_lr": MODEL_SPECS[model]["head_lr"][method],
            "weight_decay": 0.0,
            "beta": 0.95,
            "bias_correction": True,
            "eps": 1e-8,
        }
    return {
        "backbone": {
            "optimizer": "Muon",
            "role": "backbone",
            "peak_lr": 0.008,
            "weight_decay": 0.1,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "eps": 1e-5,
        },
        "input_embedding": {
            "optimizer": "AdamW",
            "role": "input_embedding",
            "peak_lr": 0.004,
            "weight_decay": 0.1,
            "betas": [0.9, 0.999],
            "eps": 1e-10,
        },
        "lm_head": head,
    }


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15)
    return actual == expected


def validate_optimizer_groups(groups: list[dict[str, Any]], model: str, method: str) -> None:
    expected_groups = expected_optimizer_groups(model, method)
    roles = [group.get("role") for group in groups]
    if len(roles) != len(set(roles)) or set(roles) != set(expected_groups):
        raise ValueError(f"{model}/{method}: optimizer roles do not match the paper")
    for group in groups:
        role = str(group["role"])
        expected = expected_groups[role]
        if set(group) != set(expected):
            raise ValueError(f"{model}/{method}: unexpected fields for {role}")
        for key, expected_value in expected.items():
            if not _values_match(group.get(key), expected_value):
                raise ValueError(f"{model}/{method}: unexpected {role} value for {key}")


def _expected_model_config(model: str) -> dict[str, Any]:
    return {
        "dim": MODEL_SPECS[model]["dim"],
        "n_layers": 32,
        "head_dim": 64,
        "mlp_expansion": 4,
        "vocab_size": 50_257,
        "max_seq_len": 2_048,
        "rope_theta": 10_000.0,
        "rms_norm_eps": 1e-6,
        "initializer_std": 0.02,
    }


def expected_sanitized_config(
    model: str, method: str, seed: int, history_sha256: str
) -> dict[str, Any]:
    spec = MODEL_SPECS[model]
    total_updates = int(spec["total_updates"])
    parameter_count = int(spec["parameter_count"])
    return {
        "record_type": "sanitized_metric_mirror",
        "record_schema_version": 1,
        "model": model,
        "backbone_optimizer": "muon",
        "lm_head_optimizer": method,
        "seed": seed,
        "parameter_count": parameter_count,
        "model_config": _expected_model_config(model),
        "dataset": {
            "name": "FineWeb",
            "sample": "GPT-2-tokenized 100B-token sample",
            "tokenizer": "GPT-2",
            "train_shards": 890,
            "validation_shards": 1,
            "train_tokens_available": 89_000_000_000,
            "validation_tokens_available": 100_000_000,
            "metadata_sha256": DATASET_METADATA_SHA256,
        },
        "training": {
            "micro_batch_per_gpu": spec["micro_batch"],
            "global_batch_sequences": 256,
            "world_size": WORLD_SIZE,
            "gradient_accumulation_steps": spec["accumulation_steps"],
            "tokens_per_update": TOKENS_PER_UPDATE,
            "tokens_per_parameter": 20,
            "target_tokens": parameter_count * 20,
            "scheduled_tokens": total_updates * TOKENS_PER_UPDATE,
            "updates": total_updates,
            "schedule": "linear_warmup_cosine_decay",
            "warmup_fraction": 0.1,
            "warmup_steps": int(total_updates * 0.1),
            "min_lr_ratio": 0.1,
            "max_gradient_norm": 1.0,
            "gradient_clipping": True,
            "gradient_checkpointing": False,
            "precision": "FP32 params / BF16 compute / FP32 logits",
        },
        "optimizer_groups": list(expected_optimizer_groups(model, method).values()),
        "diagnostics": {
            "diameter_every_updates": DIAMETER_EVERY,
            "diameter_block_rows": 4_096,
            "validation_every_updates": EVAL_EVERY,
            "validation_tokens": EVAL_TOKENS,
            "hilbert_panel_tokens": HILBERT_TOKENS,
            "increment_excludes_decoupled_weight_decay": True,
        },
        "environment": {
            "accelerator": "NVIDIA H200",
            "accelerators": WORLD_SIZE,
            "torch": "2.13.0+cu126",
            "cuda": "12.6",
        },
        "metric_history_sha256": history_sha256,
    }


def _validate_run_config(
    config: Mapping[str, Any],
    model: str,
    method: str,
    seed: int,
    history_sha256: str,
) -> list[dict[str, Any]]:
    normalized = json.loads(json.dumps(dict(config), allow_nan=False))
    raw_groups = normalized.get("optimizer_groups")
    if not isinstance(raw_groups, list) or not all(isinstance(group, dict) for group in raw_groups):
        raise ValueError("run config is missing resolved optimizer groups")
    groups = [dict(group) for group in raw_groups]
    validate_optimizer_groups(groups, model, method)
    expected = expected_sanitized_config(model, method, seed, history_sha256)
    if normalized != expected:
        unexpected = sorted(set(normalized) - set(expected))
        missing = sorted(set(expected) - set(normalized))
        changed = sorted(
            key for key in set(normalized) & set(expected) if normalized[key] != expected[key]
        )
        raise ValueError(
            f"{model}/{method}/seed{seed}: sanitized config mismatch; "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    return groups


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    model: str,
    method: str,
    seed: int,
    history_sha256: str,
    validation: list[dict[str, Any]],
    diameter: list[dict[str, Any]],
) -> None:
    label = f"{model}/{method}/seed{seed}"
    total_updates = int(MODEL_SPECS[model]["total_updates"])
    expected = {
        "final_update": total_updates,
        "final_tokens": total_updates * TOKENS_PER_UPDATE,
        "final_tpp": total_updates * TOKENS_PER_UPDATE / int(MODEL_SPECS[model]["parameter_count"]),
        "final_validation_loss": validation[-1]["validation/loss"],
        "final_validation_hilbert_step_rms": validation[-1]["validation/hilbert_step/rms"],
        "final_validation_hilbert_step_tokens": HILBERT_TOKENS,
        "final_lm_head_step_diameter_exact": diameter[-1][
            "diagnostics/lm_head_step/diameter_exact"
        ],
        "validation_point_count": len(validation),
        "diameter_point_count": len(diameter),
        "metric_history_sha256": history_sha256,
    }
    for key, expected_value in expected.items():
        if not _values_match(summary.get(key), expected_value):
            raise ValueError(f"{label}: summary mismatch for {key}")


def export_run(entry: Mapping[str, Any], run: Any) -> dict[str, Any]:
    model = str(entry["model"])
    method = str(entry["method"])
    seed = int(entry["seed"])
    expected_sha256 = str(entry["metric_history_sha256"])
    label = f"{model}/{method}/seed{seed}"
    if run.state != "finished":
        raise ValueError(f"{label}: W&B state is {run.state!r}")
    if str(run.id) != entry["run_id"]:
        raise ValueError(f"{label}: W&B returned the wrong run ID")

    validation, diameter, actual_sha256 = _history_rows(run, model, method, seed)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label}: metric history checksum does not match the manifest")
    config = dict(_require_mapping(run.config, f"{label} config"))
    groups = _validate_run_config(config, model, method, seed, expected_sha256)
    summary = _mapping_from_items(run.summary, f"{label} summary")
    _validate_summary(
        summary,
        model=model,
        method=method,
        seed=seed,
        history_sha256=expected_sha256,
        validation=validation,
        diameter=diameter,
    )

    return {
        "model": model,
        "method": method,
        "seed": seed,
        "provenance": {
            "run_id": str(run.id),
            "run_name": str(run.name),
            "run_url": str(run.url),
            "metric_history_sha256": actual_sha256,
            "optimizer_groups": groups,
        },
        "validation": validation,
        "diameter": diameter,
    }


def export_manifest(manifest: Mapping[str, Any], api: Any) -> dict[str, Any]:
    validated = validate_manifest(manifest)
    project = validated["project"]
    exported = [
        export_run(entry, api.run(f"{project}/{entry['run_id']}")) for entry in validated["runs"]
    ]
    model_specs = {
        model: {
            "parameter_count": values["parameter_count"],
            "total_updates": values["total_updates"],
            "tokens_per_update": TOKENS_PER_UPDATE,
        }
        for model, values in MODEL_SPECS.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_project": project,
        "source_manifest_sha256": manifest_fingerprint(validated),
        "model_specs": model_specs,
        "runs": exported,
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    import wandb

    payload = export_manifest(manifest, wandb.Api(timeout=120))
    write_json_atomic(args.output, payload)
    print(f"exported {len(payload['runs'])} runs to {args.output}", flush=True)


if __name__ == "__main__":
    main()

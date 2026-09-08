"""Render the final manuscript plots from an offline scalar-data export."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from .export_final_curves import (
        DEFAULT_MANIFEST,
        DIAMETER_EVERY,
        EVAL_EVERY,
        HILBERT_TOKENS,
        METHODS,
        MODEL_SPECS,
        RUN_ID_PATTERN,
        SCHEMA_VERSION,
        SHA256_PATTERN,
        TOKENS_PER_UPDATE,
        expected_updates,
        load_manifest,
        manifest_fingerprint,
        metric_history_sha256,
        validate_manifest,
        validate_optimizer_groups,
    )
except ImportError:
    from export_final_curves import (  # type: ignore[no-redef]
        DEFAULT_MANIFEST,
        DIAMETER_EVERY,
        EVAL_EVERY,
        HILBERT_TOKENS,
        METHODS,
        MODEL_SPECS,
        RUN_ID_PATTERN,
        SCHEMA_VERSION,
        SHA256_PATTERN,
        TOKENS_PER_UPDATE,
        expected_updates,
        load_manifest,
        manifest_fingerprint,
        metric_history_sha256,
        validate_manifest,
        validate_optimizer_groups,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(__file__).with_name("data") / "final_curves.json"
DEFAULT_OUTPUT = ROOT / "manuscript" / "Plots"

BLUE = "#4C72B0"
RED = "#C44E52"
RUN_STYLE = {
    "adamw": {"label": "AdamW", "color": RED},
    "rownorm": {"label": "RowNorm", "color": BLUE},
}
LEGEND_METHODS = ("rownorm", "adamw")
FILENAMES = (
    "loss_vs_training_tokens.pdf",
    "diameter_vs_training_tokens.pdf",
    "hilbert_perturbation_vs_training_tokens.pdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final manuscript plots from frozen offline data."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and positive") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


def _validate_rows(
    rows: Any,
    *,
    model: str,
    method: str,
    seed: int,
    interval: int,
    value_key: str,
    required_keys: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{model}/{method}/seed{seed}: history must be a list")
    normalized = []
    for index, raw_row in enumerate(rows):
        row = dict(_require_mapping(raw_row, f"{model}/{method}/seed{seed} row {index}"))
        if set(row) != required_keys:
            raise ValueError(f"{model}/{method}/seed{seed}: unexpected metric fields")
        update = _integer(row["progress/update"], "history update")
        tokens = _integer(row["progress/tokens"], "history tokens")
        if tokens != update * TOKENS_PER_UPDATE:
            raise ValueError(f"{model}/{method}/seed{seed}: token count does not match update")
        row["progress/update"] = update
        row["progress/tokens"] = tokens
        row[value_key] = _positive(row[value_key], value_key)
        if "validation/hilbert_step/rms" in row:
            row["validation/hilbert_step/rms"] = _positive(
                row["validation/hilbert_step/rms"], "Hilbert RMS"
            )
            panel_tokens = _integer(row["validation/hilbert_step/tokens"], "Hilbert panel size")
            if panel_tokens != HILBERT_TOKENS:
                raise ValueError(f"{model}/{method}/seed{seed}: unexpected Hilbert panel size")
            row["validation/hilbert_step/tokens"] = panel_tokens
        normalized.append(row)

    total_updates = int(MODEL_SPECS[model]["total_updates"])
    if [row["progress/update"] for row in normalized] != expected_updates(total_updates, interval):
        raise ValueError(f"{model}/{method}/seed{seed}: incomplete {value_key} history")
    return normalized


def validate_payload(
    raw_payload: Any, manifest_override: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload = dict(_require_mapping(raw_payload, "plot data"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported final-curve data schema")
    manifest = (
        load_manifest(DEFAULT_MANIFEST)
        if manifest_override is None
        else validate_manifest(manifest_override)
    )
    expected_manifest_sha256 = manifest_fingerprint(manifest)
    if payload.get("source_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("plot data does not match the frozen final-run manifest")
    source_project = payload.get("source_project")
    if (
        not isinstance(source_project, str)
        or source_project.count("/") != 1
        or any(not part for part in source_project.split("/"))
    ):
        raise ValueError("plot data is missing its source project")
    if source_project != manifest["project"]:
        raise ValueError("plot data source project does not match the frozen manifest")

    manifest_runs = {(run["model"], run["method"], run["seed"]): run for run in manifest["runs"]}

    raw_specs = _require_mapping(payload.get("model_specs"), "model specs")
    if set(raw_specs) != set(MODEL_SPECS):
        raise ValueError("plot data must describe every final model size")
    for model, expected in MODEL_SPECS.items():
        actual = _require_mapping(raw_specs[model], f"{model} model spec")
        expected_spec = {
            "parameter_count": expected["parameter_count"],
            "total_updates": expected["total_updates"],
            "tokens_per_update": TOKENS_PER_UPDATE,
        }
        if dict(actual) != expected_spec:
            raise ValueError(f"{model} model spec does not match the final campaign")

    raw_runs = payload.get("runs")
    expected_cells = set(manifest_runs)
    if not isinstance(raw_runs, list) or len(raw_runs) != len(expected_cells):
        raise ValueError("plot data must contain every run in the frozen manifest")
    runs: list[dict[str, Any]] = []
    cells: set[tuple[str, str, int]] = set()
    for index, raw_run in enumerate(raw_runs):
        run = dict(_require_mapping(raw_run, f"plot run {index}"))
        model = run.get("model")
        method = run.get("method")
        seed = _integer(run.get("seed"), f"plot run {index} seed")
        if model not in MODEL_SPECS or method not in METHODS:
            raise ValueError(f"plot run {index} has an invalid experiment cell")
        cell = (str(model), str(method), seed)
        if cell not in manifest_runs:
            raise ValueError(f"plot run {index} is not in the frozen manifest")
        if cell in cells:
            raise ValueError(f"duplicate plot cell: {cell}")
        cells.add(cell)
        provenance = dict(_require_mapping(run.get("provenance"), f"{cell} provenance"))
        expected_provenance_fields = {
            "run_id",
            "run_name",
            "run_url",
            "metric_history_sha256",
            "optimizer_groups",
        }
        if set(provenance) != expected_provenance_fields:
            raise ValueError(f"{cell} provenance has unexpected fields")
        run_id = provenance.get("run_id")
        run_name = provenance.get("run_name")
        run_url = provenance.get("run_url")
        history_sha256 = provenance.get("metric_history_sha256")
        if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(f"{cell} provenance has an invalid run ID")
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError(f"{cell} provenance has an invalid run name")
        if not isinstance(run_url, str) or not run_url.startswith("https://"):
            raise ValueError(f"{cell} provenance has an invalid run URL")
        if not isinstance(history_sha256, str) or SHA256_PATTERN.fullmatch(history_sha256) is None:
            raise ValueError(f"{cell} provenance has an invalid history checksum")
        expected_run = manifest_runs[cell]
        if run_id != expected_run["run_id"]:
            raise ValueError(f"{cell} run ID does not match the frozen manifest")
        if history_sha256 != expected_run["metric_history_sha256"]:
            raise ValueError(f"{cell} history checksum does not match the frozen manifest")
        optimizer_groups = provenance.get("optimizer_groups")
        if not isinstance(optimizer_groups, list) or not all(
            isinstance(group, dict) for group in optimizer_groups
        ):
            raise ValueError(f"{cell} provenance has invalid optimizer groups")
        validate_optimizer_groups(optimizer_groups, str(model), str(method))

        validation = _validate_rows(
            run.get("validation"),
            model=str(model),
            method=str(method),
            seed=seed,
            interval=EVAL_EVERY,
            value_key="validation/loss",
            required_keys={
                "progress/update",
                "progress/tokens",
                "validation/loss",
                "validation/hilbert_step/rms",
                "validation/hilbert_step/tokens",
            },
        )
        diameter = _validate_rows(
            run.get("diameter"),
            model=str(model),
            method=str(method),
            seed=seed,
            interval=DIAMETER_EVERY,
            value_key="diagnostics/lm_head_step/diameter_exact",
            required_keys={
                "progress/update",
                "progress/tokens",
                "diagnostics/lm_head_step/diameter_exact",
            },
        )
        actual_history_sha256 = metric_history_sha256(validation, diameter)
        if actual_history_sha256 != history_sha256:
            raise ValueError(f"{cell} exported metric history checksum is invalid")
        runs.append(
            {
                "model": str(model),
                "method": str(method),
                "seed": seed,
                "provenance": provenance,
                "validation": validation,
                "diameter": diameter,
            }
        )

    model_order = {model: index for index, model in enumerate(MODEL_SPECS)}
    method_order = {method: index for index, method in enumerate(METHODS)}
    if cells != expected_cells:
        raise ValueError("plot data must contain every run in the frozen manifest")
    runs.sort(
        key=lambda run: (
            model_order[run["model"]],
            method_order[run["method"]],
            run["seed"],
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": expected_manifest_sha256,
        "source_project": payload["source_project"],
        "model_specs": {model: dict(raw_specs[model]) for model in MODEL_SPECS},
        "runs": runs,
    }


def load_payload(path: Path, manifest_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return validate_payload(json.loads(path.read_text()), manifest_override)


def _series_from_runs(
    runs: list[Mapping[str, Any]], section: str, value_key: str
) -> dict[str, Any]:
    import numpy as np

    if not runs:
        raise ValueError("cannot aggregate an empty run group")
    tokens = np.asarray([row["progress/tokens"] for row in runs[0][section]], dtype=np.float64)
    observations = []
    for run in runs:
        rows = run[section]
        run_tokens = np.asarray([row["progress/tokens"] for row in rows], dtype=np.float64)
        if not np.array_equal(run_tokens, tokens):
            raise ValueError("replicate histories must use the same token grid")
        observations.append(np.asarray([row[value_key] for row in rows], dtype=np.float64))
    values = np.stack(observations, axis=0)
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
    return {
        "tokens": tokens,
        "values": mean,
        "standard_deviation": standard_deviation,
        "lower": mean - standard_deviation,
        "upper": mean + standard_deviation,
        "n": len(runs),
    }


def _configure_matplotlib() -> tuple[Any, Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 11.5,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt, Line2D, FuncFormatter, MaxNLocator


def _format_billions(value: float, _position: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:g}B"


def _prepare_axis(axis: Any, maximum_tokens: float, FuncFormatter: Any, MaxNLocator: Any) -> None:
    axis.set_xlim(0, maximum_tokens * 1.025)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    axis.xaxis.set_major_formatter(FuncFormatter(_format_billions))
    axis.set_xlabel("Training tokens")
    axis.tick_params(width=0.8, length=3.5)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def _draw_series(axis: Any, series: Mapping[str, Any], color: str, *, linewidth: float) -> None:
    x = series["tokens"] / 1e9
    if series["n"] > 1:
        axis.fill_between(
            x,
            series["lower"],
            series["upper"],
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
    axis.plot(x, series["values"], color=color, linewidth=linewidth, zorder=2)


def _legend_handles(Line2D: Any) -> list[Any]:
    return [
        Line2D(
            [0],
            [0],
            color=RUN_STYLE[method]["color"],
            linewidth=1.8,
            label=RUN_STYLE[method]["label"],
        )
        for method in LEGEND_METHODS
    ]


def _plot_loss(
    axis: Any,
    series: Mapping[str, Mapping[str, Any]],
    maximum_tokens: float,
    FuncFormatter: Any,
    MaxNLocator: Any,
) -> None:
    for method in METHODS:
        _draw_series(axis, series[method], RUN_STYLE[method]["color"], linewidth=1.8)
    _prepare_axis(axis, maximum_tokens, FuncFormatter, MaxNLocator)
    axis.set_ylabel("Validation loss")

    late_start = maximum_tokens * 0.68
    inset = axis.inset_axes([0.47, 0.46, 0.50, 0.45])
    late_values = []
    for method in METHODS:
        method_series = series[method]
        x = method_series["tokens"] / 1e9
        mask = x >= late_start
        inset_series = {
            "tokens": method_series["tokens"][mask],
            "values": method_series["values"][mask],
            "standard_deviation": method_series["standard_deviation"][mask],
            "lower": method_series["lower"][mask],
            "upper": method_series["upper"][mask],
            "n": method_series["n"],
        }
        _draw_series(
            inset,
            inset_series,
            RUN_STYLE[method]["color"],
            linewidth=1.05,
        )
        late_values.extend(method_series["lower"][mask].tolist())
        late_values.extend(method_series["upper"][mask].tolist())
    span = max(late_values) - min(late_values)
    padding = max(span * 0.12, 0.004)
    inset.set_xlim(late_start, maximum_tokens * 1.01)
    inset.set_ylim(min(late_values) - padding, max(late_values) + padding)
    inset.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    inset.xaxis.set_major_formatter(FuncFormatter(_format_billions))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    inset.tick_params(labelsize=9.5, width=0.65, length=2.2, pad=1.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
    axis.indicate_inset_zoom(inset, edgecolor="#555555", linewidth=0.75, alpha=0.8)


def _plot_log_metric(
    axis: Any,
    series: Mapping[str, Mapping[str, Any]],
    maximum_tokens: float,
    ylabel: str,
    FuncFormatter: Any,
    MaxNLocator: Any,
) -> None:
    all_values = []
    for method in METHODS:
        if any(series[method]["lower"] <= 0):
            raise ValueError("mean-minus-SD band must remain positive on a logarithmic axis")
        _draw_series(axis, series[method], RUN_STYLE[method]["color"], linewidth=1.45)
        all_values.extend(series[method]["lower"].tolist())
        all_values.extend(series[method]["upper"].tolist())
    _prepare_axis(axis, maximum_tokens, FuncFormatter, MaxNLocator)
    axis.set_yscale("log")
    axis.set_ylim(min(all_values) / 1.35, max(all_values) * 1.35)
    axis.set_ylabel(ylabel)


def _save_pdf(figure: Any, path: Path, *, model: str, title: str) -> None:
    metadata = {
        "Title": title,
        "Author": "Anonymous Authors",
        "Subject": f"AdamW head and RowNorm head curves for {model.upper()}",
        "Keywords": "language model; LM head; AdamW; RowNorm; Hilbert distance",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(path, metadata=metadata)


def generate_plots(
    raw_payload: Any,
    output_dir: Path,
    manifest_override: Mapping[str, Any] | None = None,
) -> list[Path]:
    payload = validate_payload(raw_payload, manifest_override)
    plt, Line2D, FuncFormatter, MaxNLocator = _configure_matplotlib()
    destinations = []
    for model, spec in MODEL_SPECS.items():
        model_runs = [run for run in payload["runs"] if run["model"] == model]
        grouped = {
            method: [run for run in model_runs if run["method"] == method] for method in METHODS
        }
        validation_loss = {
            method: _series_from_runs(runs, "validation", "validation/loss")
            for method, runs in grouped.items()
        }
        diameter = {
            method: _series_from_runs(
                runs,
                "diameter",
                "diagnostics/lm_head_step/diameter_exact",
            )
            for method, runs in grouped.items()
        }
        hilbert = {
            method: _series_from_runs(
                runs,
                "validation",
                "validation/hilbert_step/rms",
            )
            for method, runs in grouped.items()
        }
        maximum_tokens = int(spec["total_updates"]) * TOKENS_PER_UPDATE / 1e9
        definitions = (
            (
                FILENAMES[0],
                "Validation loss",
                "loss",
                validation_loss,
                "Validation loss",
            ),
            (
                FILENAMES[1],
                "LM head step diameter",
                "log",
                diameter,
                r"Row diameter $D(S_t)$",
            ),
            (
                FILENAMES[2],
                "Hilbert RMS perturbation",
                "log",
                hilbert,
                "Hilbert RMS perturbation",
            ),
        )
        model_dir = output_dir / model.upper()
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename, metadata_title, plot_kind, plot_series, ylabel in definitions:
            # The manuscript places three panels across a two-column figure,
            # so use source fonts that remain legible after scaling.
            figure, axis = plt.subplots(figsize=(4.2, 3.75))
            if plot_kind == "loss":
                _plot_loss(
                    axis,
                    plot_series,
                    maximum_tokens,
                    FuncFormatter,
                    MaxNLocator,
                )
            else:
                _plot_log_metric(
                    axis,
                    plot_series,
                    maximum_tokens,
                    ylabel,
                    FuncFormatter,
                    MaxNLocator,
                )
            # Reserve a legend row below the x label and enough right margin
            # for the longest endpoint tick (12.5B) without covering any data.
            figure.subplots_adjust(left=0.19, right=0.93, top=0.97, bottom=0.29)
            figure.legend(
                handles=_legend_handles(Line2D),
                loc="lower center",
                bbox_to_anchor=(0.56, 0.01),
                ncols=2,
                handlelength=1.7,
                columnspacing=1.2,
                borderaxespad=0.35,
            )
            destination = model_dir / filename
            _save_pdf(
                figure,
                destination,
                model=model,
                title=f"{model.upper()} {metadata_title}",
            )
            plt.close(figure)
            destinations.append(destination)
    return destinations


def main() -> None:
    args = parse_args()
    payload = json.loads(args.data.read_text())
    destinations = generate_plots(payload, args.output_dir)
    print(f"generated {len(destinations)} PDFs under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from reports import export_final_curves as exporter
from reports import generate_final_plots as plotter

EXPECTED_HISTORY_HASHES = {
    ("190m", "adamw", 0): "489210dfb9a43baa7811f061d6ef890e58868deaf6bfbcf809e518c36194a35d",
    ("190m", "adamw", 1): "8bfd721740a29403aafde66f7efd729c551195bcba75f788bd00eabaef2c78f2",
    ("190m", "adamw", 2): "bca40930a50d7ff18654c45ff3f495a7f9c0dde5456b83cecd10b3142736de8c",
    ("190m", "rownorm", 0): "fdbad19818f1dd279fd5e72ec16a8b05c019c14e9f3cb8ad2252f0a0e02beb50",
    ("190m", "rownorm", 1): "ef93ebe69aa36df1db93162724456f233c8b928a8d0e3b8edf1668272594f666",
    ("190m", "rownorm", 2): "9f6e2e58ea431aae14aa693c869bac9ee579be3065896e1dc01d3b17ebed2565",
    ("380m", "adamw", 0): "a61d9896518ed7ff6e2ac553e471fd9e7af001e08df746f07ea74941ae6bef89",
    ("380m", "adamw", 1): "a3481d48dec73bbb8ebc94df9a71f0721e1c97599ffd8ab74140da33e31dcb45",
    ("380m", "adamw", 2): "8dcfdf3022a16fdf0a982b864d054fae0e58f2909468dedd08d946ec88e53b26",
    ("380m", "rownorm", 0): "6386880aa69e035dfc85e6d54da0ad7346f9dc4388e12911c7d045b240979290",
    ("380m", "rownorm", 1): "0174727b4a679009cda770072b6ecaa01b2d1698a46c1bf74239309908326bb3",
    ("380m", "rownorm", 2): "4b6561ab9fd250e5c7ba0e7a6c262898df1df3534cb22e70fdf478dc57080263",
    ("640m", "adamw", 0): "7c1394638a7a0f70b7154f7ca107b21024f47d6fc786c3e4b4018d16618b8ce3",
    ("640m", "adamw", 1): "3495ef312edfff6a5aa206e266dccd70ba89f7fd61a2ee54e7746f03227b7590",
    ("640m", "adamw", 2): "0d3d3a59fa581e48ab78c6be1e8eb62ee703cff19141c5c06f679b83e4f9dedc",
    ("640m", "rownorm", 0): "80edec95b39e57a411137b3f942c2a6fde9a846876a199e19ca72e35bec6a8b1",
    ("640m", "rownorm", 1): "e862f14e7423a4852d8ed76ade0bd9eb6c52c07333635eee49c1f802d06a451d",
    ("640m", "rownorm", 2): "ba6f551015d0cbd3f3ceb3a73aa5658b1634461df139623a6e2e14b478343d36",
}


class ItemsOnlySummary:
    """Match W&B's summary object, which is dict-like but not a Mapping."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def items(self):
        return self.payload.items()


class FakeRun:
    def __init__(self, *, model: str, method: str, seed: int) -> None:
        spec = exporter.MODEL_SPECS[model]
        self.id = exporter.expected_run_id(model, method, seed)
        self.name = f"{model}-{method}-head-seed{seed}"
        self.url = f"https://wandb.invalid/runs/{self.id}"
        self.state = "finished"
        method_offset = 0.1 if method == "rownorm" else 0.0
        seed_offset = seed * 0.01

        diameter = [
            {
                "progress/update": update,
                "progress/tokens": update * exporter.TOKENS_PER_UPDATE,
                "diagnostics/lm_head_step/diameter_exact": (1.0 + method_offset + seed_offset)
                / (index + 1),
            }
            for index, update in enumerate(
                exporter.expected_updates(spec["total_updates"], exporter.DIAMETER_EVERY)
            )
        ]
        validation = [
            {
                "progress/update": update,
                "progress/tokens": update * exporter.TOKENS_PER_UPDATE,
                "validation/loss": 5.0
                - index
                / max(
                    1,
                    len(exporter.expected_updates(spec["total_updates"], exporter.EVAL_EVERY)),
                )
                + method_offset
                + seed_offset,
                "validation/hilbert_step/rms": (1.0 + method_offset + seed_offset) / (index + 1),
                "validation/hilbert_step/tokens": exporter.HILBERT_TOKENS,
            }
            for index, update in enumerate(
                exporter.expected_updates(spec["total_updates"], exporter.EVAL_EVERY)
            )
        ]
        merged = {row["progress/update"]: dict(row) for row in diameter}
        for row in validation:
            merged[row["progress/update"]].update(row)
        dashboard = [
            {
                "progress/update": update,
                "progress/tokens": update * exporter.TOKENS_PER_UPDATE,
                **{
                    key: 0.0 if key.endswith("fraction") else 0.1 + index / 10_000
                    for key in exporter.PANEL_HISTORY_KEYS
                },
            }
            for index, update in enumerate(
                exporter.expected_updates(spec["total_updates"], exporter.DIAMETER_EVERY)
            )
        ]
        self.history_rows = [merged[update] for update in sorted(merged)]
        self.dashboard_rows = dashboard
        self.validation = validation
        self.diameter = diameter
        self.history_sha256 = exporter.metric_history_sha256(validation, diameter)
        self.config = exporter.expected_sanitized_config(model, method, seed, self.history_sha256)
        total_updates = int(spec["total_updates"])
        self.summary: Any = {
            "final_update": total_updates,
            "final_tokens": total_updates * exporter.TOKENS_PER_UPDATE,
            "final_tpp": total_updates * exporter.TOKENS_PER_UPDATE / int(spec["parameter_count"]),
            "final_validation_loss": validation[-1]["validation/loss"],
            "final_validation_hilbert_step_rms": validation[-1]["validation/hilbert_step/rms"],
            "final_validation_hilbert_step_tokens": exporter.HILBERT_TOKENS,
            "final_lm_head_step_diameter_exact": diameter[-1][
                "diagnostics/lm_head_step/diameter_exact"
            ],
            "validation_point_count": len(validation),
            "diameter_point_count": len(diameter),
            "metric_history_sha256": self.history_sha256,
        }
        self.scan_calls: list[dict[str, Any]] = []
        self.history_calls: list[dict[str, Any]] = []
        diameter_count = len(diameter)
        validation_count = len(validation)
        line_count = 2 * diameter_count
        counts = {
            "progress/update": line_count,
            "progress/tokens": line_count,
            "diagnostics/lm_head_step/diameter_exact": diameter_count,
            "validation/loss": validation_count,
            "validation/hilbert_step/rms": validation_count,
            "validation/hilbert_step/tokens": validation_count,
            **{key: diameter_count for key in exporter.PANEL_HISTORY_KEYS},
        }
        self._attrs = {
            "historyKeys": {
                "keys": {
                    **{
                        key: {"typeCounts": [{"type": "number", "count": count}]}
                        for key, count in counts.items()
                    },
                    "_step": {"typeCounts": [{"type": "number", "count": line_count}]},
                    "_timestamp": {
                        "typeCounts": [{"type": "number", "count": line_count}]
                    },
                    "_runtime": {"typeCounts": [{"type": "number", "count": line_count}]},
                },
                "lastStep": line_count - 1,
                "sets": [],
            },
            "historyLineCount": line_count,
        }

    @property
    def entry(self) -> dict[str, Any]:
        model = str(self.config["model"])
        method = str(self.config["lm_head_optimizer"])
        seed = int(self.config["seed"])
        return {
            "model": model,
            "method": method,
            "seed": seed,
            "run_id": self.id,
            "metric_history_sha256": self.history_sha256,
        }

    def scan_history(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.scan_calls.append(kwargs)
        return self.history_rows

    def history(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.history_calls.append(kwargs)
        return self.dashboard_rows


class FakeApi:
    def __init__(self, runs: list[FakeRun]) -> None:
        self.runs = {run.id: run for run in runs}

    def run(self, path: str) -> FakeRun:
        project, run_id = path.rsplit("/", 1)
        assert project == exporter.PUBLIC_WANDB_PROJECT
        return self.runs[run_id]


def make_campaign() -> tuple[dict[str, Any], dict[str, Any]]:
    runs = [
        FakeRun(model=model, method=method, seed=seed)
        for model in exporter.MODEL_SPECS
        for method in exporter.METHODS
        for seed in exporter.SEEDS
    ]
    manifest = {
        "schema_version": exporter.SCHEMA_VERSION,
        "project": exporter.PUBLIC_WANDB_PROJECT,
        "runs": [run.entry for run in runs],
    }
    return manifest, exporter.export_manifest(manifest, FakeApi(runs))


def test_final_manifest_freezes_public_sanitized_runs() -> None:
    manifest = exporter.load_manifest(exporter.DEFAULT_MANIFEST)

    assert manifest["project"] == "mbzuai-llm/Hilbert-RowNorm"
    assert len(manifest["runs"]) == 18
    assert {
        (run["model"], run["method"], run["seed"]): run["metric_history_sha256"]
        for run in manifest["runs"]
    } == EXPECTED_HISTORY_HASHES
    assert all(
        run["run_id"] == exporter.expected_run_id(run["model"], run["method"], run["seed"])
        for run in manifest["runs"]
    )
    assert "/mnt/" not in exporter.DEFAULT_MANIFEST.read_text()


def test_manifest_requires_the_exact_matrix_and_fields() -> None:
    manifest = json.loads(exporter.DEFAULT_MANIFEST.read_text())
    manifest["runs"] = manifest["runs"][:-1]
    with pytest.raises(ValueError, match="exact 3 x 2 x 3 matrix"):
        exporter.validate_manifest(manifest)

    manifest = json.loads(exporter.DEFAULT_MANIFEST.read_text())
    manifest["runs"][0]["run_id"] = "noncanonical"
    with pytest.raises(ValueError, match="noncanonical W&B run ID"):
        exporter.validate_manifest(manifest)

    manifest = json.loads(exporter.DEFAULT_MANIFEST.read_text())
    manifest["runs"][0]["extra"] = "not-allowlisted"
    with pytest.raises(ValueError, match="unexpected fields"):
        exporter.validate_manifest(manifest)


def test_export_run_verifies_history_config_and_summary() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)

    exported = exporter.export_run(run.entry, run)

    assert run.scan_calls == [{"page_size": exporter.SCAN_PAGE_SIZE}]
    assert run.history_calls == [
        {
            "samples": len(run.diameter),
            "keys": list(exporter.PANEL_QUERY_KEYS),
            "pandas": False,
        }
    ]
    assert set(exported["validation"][0]) == set(exporter.VALIDATION_KEYS)
    assert set(exported["diameter"][0]) == set(exporter.DIAMETER_KEYS)
    assert exported["provenance"]["metric_history_sha256"] == run.history_sha256
    assert set(exported["provenance"]) == {
        "run_id",
        "run_name",
        "run_url",
        "metric_history_sha256",
        "optimizer_groups",
    }


def test_export_run_accepts_wandb_mapping_like_summary() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)
    run.summary = ItemsOnlySummary(run.summary)

    exported = exporter.export_run(run.entry, run)

    assert exported["provenance"]["run_id"] == run.id


def test_export_run_accepts_complete_dashboard_history() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)

    exported = exporter.export_run(run.entry, run)

    assert exported["provenance"]["metric_history_sha256"] == run.history_sha256


def test_export_run_ignores_identical_backend_duplicates() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)
    run.history_rows.insert(1, copy.deepcopy(run.history_rows[0]))

    exported = exporter.export_run(run.entry, run)

    assert exported["provenance"]["metric_history_sha256"] == run.history_sha256


def test_export_run_rejects_conflicting_backend_duplicates() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)
    conflict = copy.deepcopy(run.history_rows[0])
    conflict["diagnostics/lm_head_step/diameter_exact"] *= 2
    run.history_rows.insert(1, conflict)

    with pytest.raises(ValueError, match="conflicting history update"):
        exporter.export_run(run.entry, run)


def test_export_run_rejects_incomplete_dashboard_history() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)
    del run.dashboard_rows[-1]["train/loss"]

    with pytest.raises(ValueError, match="incomplete dashboard metrics"):
        exporter.export_run(run.entry, run)


def test_export_run_rejects_unrelated_history() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)
    run._attrs["historyKeys"]["keys"]["system/hostname"] = {
        "typeCounts": [{"type": "string", "count": 1}]
    }

    with pytest.raises(ValueError, match="live history allowlist mismatch"):
        exporter.export_run(run.entry, run)


def test_export_run_fails_closed_on_recipe_drift() -> None:
    run = FakeRun(model="190m", method="rownorm", seed=0)
    run.config["optimizer_groups"][0]["momentum"] = 0.9

    with pytest.raises(ValueError, match="unexpected backbone value for momentum"):
        exporter.export_run(run.entry, run)


def test_export_run_rejects_non_allowlisted_config() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)
    run.config["data"] = "/private/data/path"

    with pytest.raises(ValueError, match="sanitized config mismatch"):
        exporter.export_run(run.entry, run)
    assert not hasattr(exporter, "DATA_PATH")


def test_export_run_rejects_metric_history_drift() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)
    run.history_rows[0]["diagnostics/lm_head_step/diameter_exact"] *= 2

    with pytest.raises(ValueError, match="history checksum does not match"):
        exporter.export_run(run.entry, run)


def test_export_run_rejects_summary_drift() -> None:
    run = FakeRun(model="190m", method="adamw", seed=0)
    run.summary["final_tokens"] += 1

    with pytest.raises(ValueError, match="summary mismatch for final_tokens"):
        exporter.export_run(run.entry, run)


def test_plot_payload_is_bound_to_manifest_and_history() -> None:
    manifest, payload = make_campaign()
    plotter.validate_payload(payload, manifest)

    wrong_run = copy.deepcopy(payload)
    wrong_run["runs"][0]["provenance"]["run_id"] = "unrelated-run"
    with pytest.raises(ValueError, match="run ID"):
        plotter.validate_payload(wrong_run, manifest)

    wrong_history = copy.deepcopy(payload)
    wrong_history["runs"][0]["diameter"][0]["diagnostics/lm_head_step/diameter_exact"] *= 2
    with pytest.raises(ValueError, match="exported metric history checksum"):
        plotter.validate_payload(wrong_history, manifest)

    wrong_manifest = copy.deepcopy(payload)
    wrong_manifest["source_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen final-run manifest"):
        plotter.validate_payload(wrong_manifest, manifest)


def test_plot_series_aggregates_replicates_with_sample_standard_deviation() -> None:
    manifest, payload = make_campaign()
    validated = plotter.validate_payload(payload, manifest)
    runs = [run for run in validated["runs"] if run["model"] == "190m" and run["method"] == "adamw"]

    series = plotter._series_from_runs(runs, "validation", "validation/loss")

    assert series["tokens"].tolist() == [row["progress/tokens"] for row in runs[0]["validation"]]
    assert series["n"] == 3
    assert series["values"][0] == pytest.approx(5.01)
    assert series["standard_deviation"][0] == pytest.approx(0.01)
    assert series["lower"][0] == pytest.approx(5.0)
    assert series["upper"][0] == pytest.approx(5.02)


def test_plot_payload_validation_preserves_provenance_and_is_idempotent() -> None:
    manifest, payload = make_campaign()

    validated = plotter.validate_payload(payload, manifest)

    assert validated["source_manifest_sha256"] == payload["source_manifest_sha256"]
    assert validated["runs"] == payload["runs"]
    assert plotter.validate_payload(validated, manifest) == validated


def test_load_then_generate_writes_nine_pdfs_with_unobstructed_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    manifest, payload = make_campaign()
    source = tmp_path / "curves.json"
    source.write_text(json.dumps(payload))
    output_dir = tmp_path / "plots"
    save_pdf = plotter._save_pdf

    def check_layout_and_save(figure, path, **kwargs):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis = figure.axes[0]
        axis_bounds = axis.get_tightbbox(renderer)
        legend_bounds = figure.legends[0].get_window_extent(renderer)
        for bounds in (axis_bounds, legend_bounds):
            assert bounds.x0 >= 0
            assert bounds.y0 >= 0
            assert bounds.x1 <= figure.bbox.x1
            assert bounds.y1 <= figure.bbox.y1
        assert legend_bounds.y1 < axis_bounds.y0
        assert axis.get_title() == ""
        for child_axis in (axis, *axis.child_axes):
            assert all(line.get_marker() == "None" for line in child_axis.lines)
            assert all(line.get_linestyle() == "-" for line in child_axis.lines)
        save_pdf(figure, path, **kwargs)

    monkeypatch.setattr(plotter, "_save_pdf", check_layout_and_save)

    destinations = plotter.generate_plots(
        plotter.load_payload(source, manifest), output_dir, manifest
    )

    expected = {
        Path(model.upper()) / filename
        for model in exporter.MODEL_SPECS
        for filename in plotter.FILENAMES
    }
    assert {path.relative_to(output_dir) for path in destinations} == expected
    generated = {path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()}
    assert generated == expected
    assert all(path.read_bytes().startswith(b"%PDF") for path in destinations)

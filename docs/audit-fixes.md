# Repository audit fixes

This change addresses the 16 actionable findings from the September 2026
repository audit. It does not replace the 18 paper runs or change their
hyperparameters. The checklist below follows the original finding numbers.

## What changes, and what does not

**Changes:** launch isolation, local logging, validation and shutdown handling,
numerical edge cases in diagnostics, figure layout, and reference metadata.

**Unchanged:** model architecture and initialization, parameter ownership,
AdamW/Muon/RowNorm equations, recipes, data ordering, clipping, accumulation,
learning-rate schedule, frozen run selection, history checksums, reported
statistics, and Hilbert token selection and multiplication precision.

The diameter calculation for **future runs** has a numerical revision. The
frozen paper measurements are not retroactively changed; see the version note
below. No cluster jobs or W&B writes were needed for these fixes.

## Review checklist

| Audit finding | Before | After / where to review |
|---|---|---|
| 1. Curated W&B project | Training defaulted to the paper mirror. | Fresh runs use `Hilbert-RowNorm-training`; the trainer reserves `Hilbert-RowNorm` for paper records. See [CLI](../src/hilbert_rownorm/cli.py) and [tracking](../src/hilbert_rownorm/tracking.py). |
| 2. Inherited run identity | Environment variables could resume or combine experiments. | New UUID and `resume="never"`; wrappers clear run/resume/sweep/launch identity, direct invocation rejects it. See [launcher](../scripts/run.sh) and [tracking](../src/hilbert_rownorm/tracking.py). |
| 3. Slurm export precedence | Stale inherited paths or commit pins could override validated values. | Pass validated values in `sbatch`'s own environment. See [submitter](../scripts/submit_head_geometry.sh). |
| 4. Empty GPU constraint | Empty meant the default H200 label. | Empty omits the constraint; unset still defaults to H200. See [submitter](../scripts/submit_head_geometry.sh). |
| 5. Diagnostics without W&B | Requested diameter was skipped; Hilbert values were not saved locally. | Independent execution and flushed `metrics.jsonl`, with `config.json` and final `summary.json`. See [runner](../src/hilbert_rownorm/runner.py) and [tracking](../src/hilbert_rownorm/tracking.py). |
| 6. Plot API composition | `generate_plots(load_payload(...))` lost the manifest fingerprint. | Preserve validated provenance through repeated validation. See [plotter](../reports/generate_final_plots.py). |
| 7. Output reuse | Trailing arguments could bypass the wrapper's path guard. | Reject wrapper-owned overrides and abbreviations; Python atomically reserves a new output directory and shares failures across ranks. See [launcher](../scripts/run.sh), [CLI](../src/hilbert_rownorm/cli.py), and [runner](../src/hilbert_rownorm/runner.py). |
| 8. Shutdown errors | Tracking failure could hide the training error and skip DDP cleanup. | Preserve the training exception and always attempt process-group teardown. See [runner](../src/hilbert_rownorm/runner.py). |
| 9. Evaluation counts | Nonpositive requests silently became one batch. | Reject nonpositive counts; retain the existing rounding convention for positive requests. See [run plan](../src/hilbert_rownorm/run_plan.py). |
| 10. Precision metadata | CPU runs claimed BF16; FP32 casting was ambiguous. | Record compute, matrix-product, cast, loss, and reduction precision separately. See [metadata](../src/hilbert_rownorm/run_metadata.py). |
| 11. Diameter cancellation | Large common offsets could cancel FP32 squared distances. | Translate first, retain FP64 inputs, directly recompute block candidates; record `centered_cdist_v2`. See [geometry](../src/hilbert_rownorm/head_geometry.py). |
| 12. Repeated Hilbert reduction | Reading the distributed accumulator twice changed its value. | Reduce a clone, preserving local totals. See [Hilbert diagnostics](../src/hilbert_rownorm/hilbert_diagnostics.py). |
| 13. Muon citation | A repository/blog entry had invalid journal metadata. | Use the official repository's citation. See [bibliography](../manuscript/sample.bib). |
| 14. Arens–Eells citation | Journal and locator were missing. | Add journal, volume, issue, pages, and DOI from the publisher. See [bibliography](../manuscript/sample.bib). |
| 15. Overlapping legends | Legends obscured diameter curves. | Reserve a legend row outside the axes in all nine PDFs. See [plotter](../reports/generate_final_plots.py). |
| 16. Clipped final ticks | The 640M `12.5B` label crossed the page boundary. | Increase right padding; test label bounds. See [figure tests](../tests/test_final_plot_pipeline.py). |

Regression coverage lives in the corresponding `tests/test_*.py` files.
Launcher tests exercise submitter → mocked Slurm → worker → wrapper → parsed
training arguments, not merely shell string matching. W&B tests use both a
fake logger and the installed SDK settings schema without contacting W&B.

## Validation performed

- All 271 CPU tests, Ruff, shell syntax, diff whitespace, and offline
  lockfile checks pass. Two end-to-end tiny-model tests also exercise the real
  loader, runner, optimizers, evaluation, and local diagnostics with W&B absent.
- An isolated comparison against the PR base matches bitwise over 12 updates
  for Muon + AdamW and Muon + RowNorm under both FP32 and CPU BF16. It checks
  losses, gradient norms, gradients, head increments, parameters, buffers,
  optimizer states, validation loss, and Hilbert RMS. Diagnostic-free controls
  match as well; diagnostics preserve RNG and numerical state. This uses a
  9,184-parameter model and synthetic tokens, not a full GPU pretraining run.
- The intended v1-to-v2 diameter arithmetic change is at most `1.863e-9` in
  those tiny comparisons; offset and FP64 regression cases are tested separately.
- A real two-process CPU/Gloo accumulation check gives identical gradients on
  both ranks, with maximum absolute difference `2.98e-8` from the combined-batch
  reference (a different floating-point reduction order).
- All 18 cached histories validate against the frozen manifest and checksums.
  All nine regenerated figures retain the same means and sample-SD bands.
  Rendered PDFs were inspected; no legend covers a curve and no text crosses
  a page edge. The longest endpoint tick has more than 14 pt of right margin.
- The manuscript compiles with Tectonic, with no missing references/citations
  or overflowing boxes. BibTeX reports no warnings. Existing font/dependency
  warnings remain; this local build is not a claim of identical TeX Live 2026
  layout. The repository's manuscript workflow checks that engine separately.

The cached-history check does not establish current W&B access settings. No
live W&B write or full H200 rerun was performed for this PR.

## Diameter measurement version

The historical paper curves use exhaustive blocked FP32 distances without
translation (v1). New runs record `diameter_algorithm=centered_cdist_v2`:

1. Subtract the first row before the blocked matrix-product distance search.
2. Keep FP64 inputs in FP64; promote lower-precision inputs to FP32.
3. Directly subtract the original rows for each block's maximizing candidate.

This fixes the reproduced common-offset cancellation case. The metric's
compatibility key still contains `exact`: it means all pairs are searched,
not that floating-point arithmetic or candidate selection is exact. This
diagnostic never writes to the parameters or optimizer state.

The nine PDFs are regenerated from the same checksum-verified frozen export.
Their means, sample standard deviations, and underlying observations are
unchanged; only layout differs. No claim is made that the published v1
measurements were affected by the reproduced edge case.

## Reproduction limits still open

The previously completed 190M seed-0 clean-repository checks did **not** match
the historical final losses bit for bit:

| Head | Historical loss | Clean-repository check | Check minus historical |
|---|---:|---:|---:|
| AdamW | 3.1583707333 | 3.1592974663 | +0.0009267330 |
| RowNorm | 3.1709878445 | 3.1705634594 | −0.0004243851 |

Both checks completed 7,083 updates. The available configuration fields agree,
but the cached check evidence contains only final validation values, not enough
history or data/runtime fingerprints to locate the first divergence. These
fixes do not establish its cause or claim to resolve it.

The next step is to recover the existing matching validation histories and
runtime/data evidence. Only if those are insufficient should a short,
controlled H200 comparison be considered. Tiny CPU checks do not establish
full-length, multi-GPU bitwise reproducibility.

Two additional limits remain explicit:

- The data verifier checks shard layout, headers, and counts, not every token
  byte. The prepared corpus is not redistributed.
- The paper's actual hyperparameter-selection process and tuning effort still
  need an author-supplied description. Final recipes are preserved; no search
  history or selection rationale is invented here.

The [README](../README.md) also specifies supported AdamW-backbone combinations,
the scope of the ideal RowNorm theorem, Hilbert panels that differ across model
scales, and checkpoint snapshots that do not implement training resumption.

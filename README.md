# Hilbert-RowNorm

Code for *Toward a First-Principles Update Geometry for the Language-Model
Head*. This repository contains the training code, experiment records, and
plotting tools.

## Key result

We study the language-model head and softmax as one module. Hilbert's
projective distance makes the functional size of a head update equal to the
Euclidean diameter of its token rows. The resulting steepest-descent problem
has an exact minimum-tension dual. Projected RowNorm solves a stronger
covering-radius problem in linear time, satisfies the original diameter bound,
and retains at least $1/\sqrt{2}$ of the optimal instantaneous linear
decrease. This guarantee assumes exact normalization of the same centered
gradient that defines the objective; it does not directly cover the practical
EMA-and-epsilon update used in training.

## Paper experiments

The paper holds a Nesterov Muon backbone and AdamW input-embedding optimizer
fixed, and compares two LM-head optimizers:

- AdamW with $(\beta_1,\beta_2)=(0.9,0.999)$,
  $\epsilon=10^{-10}$, and weight decay $0.1$;
- RowNorm with a bias-corrected EMA, $\beta=0.95$,
  $\epsilon=10^{-8}$, and no head weight decay.

The complete matrix contains 18 runs: 190M, 380M, and 640M models; AdamW and
RowNorm heads; and seeds 0, 1, and 2. All runs use a nominal budget of 20
tokens per parameter, a global batch of 256 sequences, context length 2,048,
10% linear warmup, cosine decay to 10% of the peak learning rate, and global
gradient clipping at 1.0. The paired runs at each scale use the same seed and
data order.

The paper reports three curves:

1. validation cross entropy;
2. exact diameter of the learning-rate-scaled LM-head increment;
3. RMS Hilbert perturbation on a fixed panel of 8,192 validation tokens.

The diameter and Hilbert measurements exclude direct decoupled weight decay.
The panel is fixed within each model scale and shared by its paired runs;
different microbatch sizes select different token positions across scales.
The complete optimizer settings are defined in
[`optimizer_recipes.py`](src/hilbert_rownorm/optimizer_recipes.py).

## Installation and data

Install the package and development dependencies with:

```bash
uv sync --extra dev
```

The trainer expects GPT-2-tokenized FineWeb binary shards named
`fineweb_train_*.bin` and `fineweb_val_*.bin`. The launch worker validates the
shard counts, aggregate token counts, and a digest of shard names, sizes, and
headers before training. That digest checks the shard layout, not every data
byte. The exact prepared corpus used for the paper is not redistributed here,
so the repository does not claim bitwise reconstruction of the training data.
The published W&B records are sufficient to reproduce the paper figures
without access to the corpus.

Inspect the direct training interface with:

```bash
uv run hilbert-rownorm-train --help
```

## Launch the paper matrix on Slurm

The launcher contains no site-specific filesystem paths or W&B account name.
Set these values for your environment before submitting:

```bash
export DATA_ROOT=/path/to/fineweb100b
export HILBERT_STORAGE_ROOT=/path/to/shared/run-storage
export WANDB_ENTITY=your-wandb-entity
export WANDB_PROJECT=Hilbert-RowNorm-training
export TRAIN_PYTHON="$PWD/.venv/bin/python"

scripts/submit_head_geometry.sh 190m
scripts/submit_head_geometry.sh 380m
scripts/submit_head_geometry.sh 640m
```

Each command submits six one-node array tasks: the two head optimizers for
seeds 0, 1, and 2. The submitter verifies a clean Git checkout and pins that
checkout for the workers. New W&B runs deliberately disable Git and source
capture, and their configuration omits local data and output paths.
Fresh runs go to `Hilbert-RowNorm-training` by default, not the curated paper
project. Each receives a new W&B ID with resumption disabled. The shell
launchers clear inherited W&B run, resume, sweep, and launch identity; direct
Python invocation rejects those variables instead of silently reusing them.
An existing output directory is rejected. Use a new directory for each run.

The resource defaults reproduce the paper setup: eight H200 GPUs, 32 CPUs,
64 GiB of host memory, and a 24-hour limit. The following variables adapt the
launcher to another Slurm installation:

```text
SLURM_PARTITION          default: main
SLURM_ACCOUNT            default: unset
SLURM_QOS                default: unset
SLURM_CONSTRAINT         default: nvidia_h200
SLURM_GPUS_PER_NODE      default: 8
SLURM_CPUS_PER_TASK      default: 32
SLURM_MEMORY             default: 64G
SLURM_TIME               default: 24:00:00
SLURM_ARRAY_CONCURRENCY  default: 6
```

Set `SLURM_CONSTRAINT=''` to omit the constraint on clusters without that label.

The paper runs used PyTorch `2.13.0+cu126` with CUDA 12.6. Prepare that CUDA
environment separately before exact replication; the default `uv sync` is
intended for code inspection and CPU validation. `TRAIN_PYTHON`,
`EXPECTED_TORCH_VERSION`, `EXPECTED_CUDA_VERSION`,
`EXPECTED_GPU_COUNT`, and `EXPECTED_GPU_PATTERN` control the worker's runtime
preflight. Changing hardware or software versions is useful for replication,
but no longer reproduces the exact execution environment reported in the
paper.

AdamW backbone selection is retained for 190M and 380M with an AdamW head.
The paper's Muon backbone supports both head optimizers at all three scales.
Unsupported combinations fail explicitly; no untested 640M AdamW backbone
recipe is inferred.

## Sweeps and Compositional Muon (beyond the paper)

These overrides support experiments beyond the paper matrix. Any of them, and
`--no-qk-norm`, changes the recipe identifier family to `sweep`, so paper
identifiers always denote the exact paper configuration; the resolved values
are recorded in `config.json` and W&B.

- LM-head sweeps: `--lm-head-lr`, `--lm-head-weight-decay`, `--lm-head-betas`
  (AdamW head; the input embedding keeps $(0.9, 0.999)$), and `--lm-head-beta`
  (RowNorm EMA). Backbone and input embedding keep their paper settings.
- `--attention-qk-optimizer compositional` moves each layer's query and key
  matrices from Muon to Compositional Muon's half-split rule (Keigwin et al.,
  2026), implemented in
  [`compositional_qk.py`](src/hilbert_rownorm/compositional_qk.py) on this
  repository's Newton–Schulz transform. It requires `--no-qk-norm` (per-head
  QK RMSNorm removes the weight scale that partner whitening measures), and
  comparison Muon runs should pass `--no-qk-norm` too. `--cm-whitening full`
  bounds each half of the first-order query–key product change exactly but
  ignores RoPE; the default `isotropic` commutes with RoPE and meets the
  bound approximately. `--cm-mp` scales the step (default 2, cancelling the
  half-split budget).
- Query/key sweeps: `--qk-lr`, `--qk-momentum`, and `--qk-weight-decay` set the
  query/key matrices' own values. With the default
  `--attention-qk-optimizer muon`, any of them puts the query/key pairs in a
  separate Muon group; with `compositional`, they replace the values
  Compositional Muon otherwise inherits from the backbone. Every other matrix
  keeps the backbone recipe.

## Local diagnostics and numerical conventions

Every run writes `config.json`, `metrics.jsonl`, and `summary.json` under its
output directory, including with `--wandb-mode disabled`. Metric lines are
flushed at each report. Diameter logging still requires
`--exact-head-diameter-every-log`; its cadence is `--log-every` plus the final
update. Hilbert logging requires a positive `--hilbert-step-probe-tokens` and
evaluation enabled. These flags are enabled by the paper matrix launcher.

New runs record `diameter_algorithm=centered_cdist_v2`. This exhaustive blocked
search removes a common row offset before computing distances and directly
recomputes candidate pair distances. It preserves FP64 inputs. The name
`exact` in the metric key means all pairs are searched, not exact arithmetic;
floating-point rounding can still affect which candidate is selected.
The frozen paper histories used uncentered FP32 distances (v1). They remain
unchanged, and regenerated paper PDFs still plot those historical measurements.

CUDA runs use BF16 autocast for the head and Hilbert matrix products, then cast
their outputs to FP32 for cross entropy or the logit range. Hilbert squared
ranges are summed in FP64. CPU runs use FP32 compute. New run metadata records
these distinctions; the old mirror's shorter precision label does not imply
FP32 matrix multiplication.

## Checkpoints and resuming

Long runs can continue after a crash or a Slurm preemption. Pass
`--save-every N` to write a checkpoint every N updates and at the end, and
`--resume` to continue from the newest checkpoint in `--output`:

```bash
# Requeued jobs must reuse the same RUN_ID, so the run keeps its directory.
MICRO_BATCH=8 scripts/run.sh 640m muon my-run --save-every 2000 --resume [TRAIN_ARGS...]
```

With `--resume`, the output directory may already exist. If it has a
checkpoint, training continues from it; if not, earlier records are moved to
`abandoned/attempt-K/` and the run starts over. A run whose final checkpoint
and `summary.json` show it finished exits without training.

A checkpoint (`checkpoints/update-XXXXXXXX.pt`) holds what the next update
depends on: model weights, every optimizer's state, the training-data
position, each rank's random-number state, the W&B run ID, and the elapsed
training time. On CPU, a crashed and resumed run matches an uninterrupted
run bit for bit; [`tests/test_resume.py`](tests/test_resume.py) checks this
for each head and query/key optimizer, and for two ranks.

- Files are written to a temporary name, synced, then renamed, so a crash
  never leaves a partial checkpoint under the final name. `--keep-checkpoints`
  (default 2) keeps the newest ones. A 640M checkpoint is about 5 GiB.
- If a write fails, for example on a full disk, training continues, the
  earlier checkpoints stay, and the metrics record `events/checkpoint_failed`.
- An unreadable checkpoint is renamed `*.corrupt` and the previous one is used.
- A run resumes only if its configuration matches the checkpoint. The
  microbatch size, rank count, `--max-steps`, logging, evaluation and
  checkpoint cadence, and diagnostics may change; the dataset, model,
  optimizer recipe and overrides, batch size, schedule, seed, and evaluation
  tokens may not. Each resume is recorded in `resumes.jsonl`, with any
  changed settings.
- `metrics.jsonl` is cut back to the checkpoint's update; the removed lines go
  to `metrics.discarded-K.jsonl`. W&B continues the same run, so the updates
  between the checkpoint and the crash appear twice there; `events/resume`
  marks where the second copy starts.
- Work done after the last checkpoint is repeated. A Slurm preemption signal
  does not trigger an extra save, so choose `--save-every` for the loss you
  accept; at 640M, 2000 updates are about 55 minutes.

The paper launcher keeps `--save-every 0` and starts each Slurm attempt in a
new directory, as the paper runs did.

## W&B records

The 18 retained records are hosted in the
[`mbzuai-llm/Hilbert-RowNorm`](https://api.wandb.ai/links/mbzuai-llm/jg7sw632)
W&B project. These are sanitized, lossless metric mirrors of the completed
training runs; they were not re-executed when these records were created.
Each record contains the scalar histories used in the paper, the six Training
panel metrics, the seven Optimizer panel metrics, and an allowlisted scientific
configuration. Source files, Git metadata, console logs, host paths,
checkpoints, system telemetry, and unrelated metrics were not copied. The
project remains private during anonymous review and can be made public with
the code release.

W&B stores the appended scalar rows in automatically generated internal
history artifact versions. Their manifests contain only storage digests and
sizes; no user-authored artifact or training output was transferred.

The curated [`Hilbert-RowNorm paper runs`](https://api.wandb.ai/links/mbzuai-llm/jg7sw632)
workspace contains six Training panels and seven Optimizer panels. Every panel
uses `progress/tokens` on the horizontal axis and includes all 18 runs.

The report manifest maps every model, head optimizer, and seed to an immutable
W&B run ID and the SHA-256 checksum of its metric history. The exporter
requires the exact 3-model, 2-method, 3-seed matrix, checks each sanitized
configuration and metric grid, and verifies every history checksum. Git
metadata is neither stored in the manifest nor used to locate results.

## Recreate the figures

Export the selected W&B histories once, then render the PDFs offline:

```bash
uv sync --extra plots
uv run python reports/export_final_curves.py
uv run python reports/generate_final_plots.py
```

The plotter creates the loss, exact-diameter, and Hilbert-RMS figures under
[`manuscript/Plots/`](manuscript/Plots/). This directory contains only the
retained plots; it does not require a LaTeX installation. Each solid curve is
the pointwise mean over three seeds.
The shaded region is the mean plus or minus one sample standard deviation; it
is descriptive run-to-run variation, not a confidence interval.

## Code structure

- [`model.py`](src/hilbert_rownorm/model.py) defines the decoder-only
  Transformer.
- [`optimizer_recipes.py`](src/hilbert_rownorm/optimizer_recipes.py) contains
  the supported recipes.
- [`optimizer_factory.py`](src/hilbert_rownorm/optimizer_factory.py),
  [`muon.py`](src/hilbert_rownorm/muon.py), and
  [`head_optimizers.py`](src/hilbert_rownorm/head_optimizers.py) implement
  parameter routing and optimizer equations.
- [`compositional_qk.py`](src/hilbert_rownorm/compositional_qk.py) applies
  Compositional Muon to query/key pairs.
- [`head_geometry.py`](src/hilbert_rownorm/head_geometry.py) computes the
  exact LM-head increment diameter.
- [`hilbert_diagnostics.py`](src/hilbert_rownorm/hilbert_diagnostics.py)
  computes the fixed-panel Hilbert RMS.
- [`training.py`](src/hilbert_rownorm/training.py) and
  [`runner.py`](src/hilbert_rownorm/runner.py) contain the training loop and
  orchestration.
- [`scripts/`](scripts/) contains the portable Slurm submitter and worker.
- [`tests/`](tests/) locks the equations, recipes, launch matrix, and metric
  semantics.

## Validation

```bash
uv run pytest
uv run ruff check .
bash -n scripts/*.sh
```

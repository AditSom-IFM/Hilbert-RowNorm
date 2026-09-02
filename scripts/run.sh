#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODEL OPTIMIZER RUN_ID [TRAIN_ARGS...]" >&2
  exit 2
fi

model=$1
optimizer=$2
run_id=$3
shift 3

output_root=${OUTPUT_ROOT:-${PWD}/runs}
for value in "${model}" "${optimizer}" "${run_id}"; do
  if [[ ! ${value} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "model, optimizer, and run ID must contain only letters, digits, '.', '_', or '-'" >&2
    exit 2
  fi
done
mkdir -p "${output_root}"
if [[ ! -w ${output_root} ]]; then
  echo "output root is not writable: ${output_root}" >&2
  exit 1
fi
output=${output_root}/${model}/${optimizer}/${run_id}
if [[ -e ${output} ]]; then
  echo "refusing to reuse existing run directory: ${output}" >&2
  exit 1
fi

wandb_args=(
  --wandb-project "${WANDB_PROJECT:-Hilbert-RowNorm}"
  --wandb-run-name "${WANDB_RUN_NAME:-${run_id}}"
)
if [[ -n ${WANDB_ENTITY:-} ]]; then
  wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n ${WANDB_MODE:-} ]]; then
  wandb_args+=(--wandb-mode "${WANDB_MODE}")
fi

nproc=${NPROC:-8}
python=${TRAIN_PYTHON:-python}
if [[ -z ${MICRO_BATCH:-} ]]; then
  echo "MICRO_BATCH must be set explicitly after profiling on the target hardware" >&2
  exit 2
fi
micro_batch=${MICRO_BATCH}

# Keep public runs independent of the local Git history and source tree.
export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true

"${python}" -m torch.distributed.run --standalone --nproc-per-node="${nproc}" \
  -m hilbert_rownorm.train \
  --model "${model}" \
  --optimizer "${optimizer}" \
  --output "${output}" \
  --micro-batch "${micro_batch}" \
  --require-cuda \
  "${wandb_args[@]}" \
  "$@"

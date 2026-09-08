#!/usr/bin/env bash
set -euo pipefail

head_geometry_resolve_submission() {
  if [[ $# -ne 1 ]]; then
    echo "usage: head_geometry_resolve_submission MODEL" >&2
    return 2
  fi

  submit_model=$1
  case ${submit_model} in
    190m|380m|640m) ;;
    *)
      echo "invalid head-geometry model: ${submit_model}" >&2
      return 2
      ;;
  esac
  local concurrency=${SLURM_ARRAY_CONCURRENCY:-6}
  if [[ ! ${concurrency} =~ ^[1-6]$ ]]; then
    echo "SLURM_ARRAY_CONCURRENCY must be an integer from 1 to 6" >&2
    return 2
  fi
  job_name=hilbert-rownorm-${submit_model}
  array=0-5%${concurrency}
  task_range=0-5
}

head_geometry_submit_main() {
  if [[ $# -ne 1 ]]; then
    echo "usage: $0 MODEL" >&2
    return 2
  fi

  head_geometry_resolve_submission "$1"

  local repo_source=${HILBERT_ROWNORM_REPO:-$(dirname "${BASH_SOURCE[0]}")/..}
  local repo
  repo=$(cd "${repo_source}" && pwd -P)
  local worker=${repo}/scripts/slurm_head_geometry.sh
  local storage_root=${HILBERT_STORAGE_ROOT:?HILBERT_STORAGE_ROOT must identify writable shared storage}
  local data_root=${DATA_ROOT:?DATA_ROOT must identify the pre-tokenized FineWeb directory}
  local wandb_project=${WANDB_PROJECT:-Hilbert-RowNorm-training}
  local partition=${SLURM_PARTITION:-main}
  local account=${SLURM_ACCOUNT:-}
  local qos=${SLURM_QOS:-}
  local constraint=${SLURM_CONSTRAINT-nvidia_h200}
  local gpus=${SLURM_GPUS_PER_NODE:-8}
  local cpus=${SLURM_CPUS_PER_TASK:-32}
  local memory=${SLURM_MEMORY:-64G}
  local walltime=${SLURM_TIME:-24:00:00}
  local expected_commit
  local job

  expected_commit=$(git -C "${repo}" rev-parse HEAD)
  "${repo}/scripts/verify_launch_checkout.sh" "${repo}" "${expected_commit}" >/dev/null
  if [[ ! -x ${worker} ]]; then
    echo "head-geometry worker is missing or not executable: ${worker}" >&2
    return 1
  fi
  local storage_directories=(
    "${storage_root}/checkpoints"
    "${storage_root}/slurm-logs"
    "${storage_root}/wandb-artifacts"
    "${storage_root}/wandb-cache"
    "${storage_root}/wandb-data"
  )
  mkdir -p "${storage_directories[@]}"
  local directory
  for directory in "${storage_directories[@]}"; do
    if [[ ! -w ${directory} ]]; then
      echo "output directory is not writable: ${directory}" >&2
      return 1
    fi
  done

  local sbatch_args=(
    --parsable
    --partition="${partition}"
    --nodes=1
    --ntasks-per-node=1
    --gpus-per-node="${gpus}"
    --cpus-per-task="${cpus}"
    --mem="${memory}"
    --time="${walltime}"
    --no-requeue
    --open-mode=append
    --output="${storage_root}/slurm-logs/%x-%A_%a.out"
    --job-name="${job_name}"
    --array="${array}"
  )
  if [[ -n ${account} ]]; then
    sbatch_args+=(--account="${account}")
  fi
  if [[ -n ${qos} ]]; then
    sbatch_args+=(--qos="${qos}")
  fi
  if [[ -n ${constraint} ]]; then
    sbatch_args+=(--constraint="${constraint}")
  fi

  job=$(
    unset WANDB_RUN_ID WANDB_RESUME WANDB_RESUME_FROM WANDB_FORK_FROM WANDB_SWEEP_ID \
      WANDB_LAUNCH WANDB_LAUNCH_CONFIG_PATH
    # With --export=ALL, Slurm prefers the sbatch environment over listed values.
    # Set the validated paths and fresh pin in that environment directly.
    HILBERT_ROWNORM_REPO=${repo} \
      EXPECTED_COMMIT=${expected_commit} \
      HILBERT_STORAGE_ROOT=${storage_root} \
      DATA_ROOT=${data_root} \
      WANDB_PROJECT=${wandb_project} \
      sbatch "${sbatch_args[@]}" --export=ALL "${worker}" "${submit_model}"
  )
  job=${job%%;*}
  echo "submitted ${submit_model} seeds 0, 1, and 2 as ${job}_[${task_range}]"
  echo "even tasks use the AdamW head; odd tasks use the RowNorm head"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  head_geometry_submit_main "$@"
fi

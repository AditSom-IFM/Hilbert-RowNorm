#!/usr/bin/env bash
set -euo pipefail

head_geometry_resolve_plan() {
  if [[ $# -ne 4 ]]; then
    echo "usage: head_geometry_resolve_plan MODEL TASK GRID ATTEMPT" >&2
    return 2
  fi
  if [[ -z ${DATA_ROOT:-} ]]; then
    echo "DATA_ROOT must identify the pre-tokenized FineWeb directory" >&2
    return 2
  fi

  model=$1
  task=$2
  grid=$3
  attempt=$4

  case ${model} in
    190m)
      micro_batch=32
      ;;
    380m)
      micro_batch=16
      ;;
    640m)
      micro_batch=8
      ;;
    *)
      echo "invalid head-geometry model: ${model}" >&2
      return 2
      ;;
  esac

  if [[ ! ${task} =~ ^[0-9]+$ ]] || (( task > 5 )); then
    echo "invalid ${model} task: ${task}" >&2
    return 2
  fi
  seed=$((task / 2))
  arm=$((task % 2))
  case ${arm} in
    0)
      lm_head_optimizer=adamw
      ;;
    1)
      lm_head_optimizer=rownorm
      ;;
  esac

  if [[ ! ${grid} =~ ^[0-9]+$ ]]; then
    echo "invalid Slurm array job ID: ${grid}" >&2
    return 2
  fi
  if [[ ! ${attempt} =~ ^[0-9]+$ ]]; then
    echo "invalid Slurm restart count: ${attempt}" >&2
    return 2
  fi

  run_id=hilbert-rownorm-${model}-${lm_head_optimizer}-20tpp-gb256-mb${micro_batch}-seed${seed}-grid${grid}-task${task}-attempt${attempt}
  train_args=(
    --data "${DATA_ROOT}"
    --lm-head-optimizer "${lm_head_optimizer}"
    --exact-head-diameter-every-log
    --exact-head-diameter-block-rows 4096
    --hilbert-step-probe-tokens 8192
    --global-batch 256
    --tokens-per-parameter 20
    --seed "${seed}"
    --warmup-fraction 0.1
    --min-lr-ratio 0.1
    --max-gradient-norm 1
    --log-every 10
    --eval-every 500
    --eval-tokens 5242880
    --save-every 0
  )
}

head_geometry_main() {
  if [[ $# -ne 1 ]]; then
    echo "usage: $0 MODEL" >&2
    return 2
  fi

  local repo=${HILBERT_ROWNORM_REPO:?HILBERT_ROWNORM_REPO must identify the immutable launch checkout}
  local expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the launch revision}
  local storage_root=${HILBERT_STORAGE_ROOT:?HILBERT_STORAGE_ROOT must identify writable shared storage}
  local output_root=${OUTPUT_ROOT:-${storage_root}/checkpoints}
  local python=${TRAIN_PYTHON:-python}
  local data=${DATA_ROOT:?DATA_ROOT must identify the pre-tokenized FineWeb directory}
  local task=${SLURM_ARRAY_TASK_ID:?this launcher requires a Slurm array task}
  local grid=${SLURM_ARRAY_JOB_ID:?this launcher requires a Slurm array job ID}
  local attempt=${SLURM_RESTART_COUNT:-0}
  local nproc=${NPROC:-${SLURM_GPUS_PER_NODE:-8}}
  local expected_gpus=${EXPECTED_GPU_COUNT:-${nproc}}
  local gpu_pattern=${EXPECTED_GPU_PATTERN:-H200}
  local torch_version=${EXPECTED_TORCH_VERSION:-2.13.0+cu126}
  local cuda_version=${EXPECTED_CUDA_VERSION:-12.6}
  local wandb_mode=${WANDB_MODE:-online}

  if [[ -n ${HEAD_GEOMETRY_MAX_STEPS:-} ]]; then
    echo "the paper launcher does not permit truncated runs" >&2
    return 2
  fi
  head_geometry_resolve_plan "$1" "${task}" "${grid}" "${attempt}"
  unset WANDB_RUN_ID WANDB_RESUME WANDB_RESUME_FROM WANDB_FORK_FROM WANDB_SWEEP_ID \
    WANDB_LAUNCH WANDB_LAUNCH_CONFIG_PATH

  "${repo}/scripts/verify_launch_checkout.sh" \
    "${repo}" "${expected_commit}" >/dev/null
  source "${repo}/scripts/cluster_job_common.sh"
  cd "${repo}"
  activate_python_environment "${repo}" "${python}"

  require_training_hardware \
    "${python}" "${expected_gpus}" "${gpu_pattern}" "${torch_version}" "${cuda_version}"
  require_fineweb_dataset \
    "${python}" \
    "${data}" \
    "${FINEWEB_TRAIN_SHARDS:-890}" \
    "${FINEWEB_VALIDATION_SHARDS:-1}" \
    "${FINEWEB_TRAIN_TOKENS:-89000000000}" \
    "${FINEWEB_VALIDATION_TOKENS:-100000000}" \
    "${FINEWEB_METADATA_SHA256:-87edf1844351c445e687ea865215b5fecff3a8b8052dff5b7263ea30712376f4}"

  local storage_directories=(
    "${output_root}"
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

  echo "run_id=${run_id} task=${task} arm=${arm} seed=${seed} restart=${attempt}"
  NPROC=${nproc} \
    MICRO_BATCH=${micro_batch} \
    OUTPUT_ROOT=${output_root} \
    PYTHONDONTWRITEBYTECODE=1 \
    WANDB_ARTIFACT_DIR=${storage_root}/wandb-artifacts \
    WANDB_CACHE_DIR=${storage_root}/wandb-cache \
    WANDB_DATA_DIR=${storage_root}/wandb-data \
    WANDB_DISABLE_GIT=true \
    WANDB_DISABLE_CODE=true \
    WANDB_PROJECT=${WANDB_PROJECT:-Hilbert-RowNorm-training} \
    WANDB_MODE=${wandb_mode} \
    WANDB_RUN_NAME=${run_id} \
    "${repo}/scripts/run.sh" "${model}" muon "${run_id}" "${train_args[@]}"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  head_geometry_main "$@"
fi

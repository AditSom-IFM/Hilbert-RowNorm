#!/usr/bin/env bash

# Shared preflight checks for the Slurm worker. Source this file only after the
# launch checkout has been verified against its pinned commit.

activate_python_environment() {
  if [[ $# -ne 2 ]]; then
    echo "usage: activate_python_environment REPO PYTHON" >&2
    return 2
  fi

  local repo=$1
  local python=$2

  export PYTHONNOUSERSITE=1
  export PYTHONPATH=${repo}/src${PYTHONPATH:+:${PYTHONPATH}}
  export TRAIN_PYTHON=${python}
}

require_training_hardware() {
  if [[ $# -ne 5 ]]; then
    echo "usage: require_training_hardware PYTHON GPUS GPU_PATTERN TORCH_VERSION CUDA_VERSION" >&2
    return 2
  fi

  local python=$1
  "${python}" -c '
import sys

import torch

expected_gpus = int(sys.argv[1])
gpu_pattern, expected_torch, expected_cuda = sys.argv[2:]
assert torch.__version__ == expected_torch, (
    f"expected torch {expected_torch}, found {torch.__version__}"
)
assert torch.version.cuda == expected_cuda, (
    f"expected CUDA {expected_cuda}, found {torch.version.cuda}"
)
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == expected_gpus, (
    f"expected {expected_gpus} GPUs, found {torch.cuda.device_count()}"
)
assert torch.distributed.is_nccl_available(), "NCCL is unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 is unsupported"
names = [torch.cuda.get_device_name(index) for index in range(expected_gpus)]
if gpu_pattern:
    assert all(gpu_pattern in name for name in names), (
        f"expected GPU names containing {gpu_pattern!r}, found {names}"
    )
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={names}", flush=True)
' "$2" "$3" "$4" "$5"
}

require_fineweb_dataset() {
  if [[ $# -ne 7 ]]; then
    echo "usage: require_fineweb_dataset PYTHON DATA TRAIN_SHARDS VALIDATION_SHARDS TRAIN_TOKENS VALIDATION_TOKENS METADATA_SHA256" >&2
    return 2
  fi

  local python=$1
  local data=$2

  "${python}" -m hilbert_rownorm.data \
    --root "${data}" \
    --train-shards "$3" \
    --validation-shards "$4" \
    --train-tokens "$5" \
    --validation-tokens "$6" \
    --metadata-sha256 "$7"
}

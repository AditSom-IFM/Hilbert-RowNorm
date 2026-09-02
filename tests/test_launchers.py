from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
WORKER = SCRIPTS / "slurm_head_geometry.sh"
SUBMITTER = SCRIPTS / "submit_head_geometry.sh"

DATA_ROOT = "/datasets/fineweb100b"
SCALES = {
    "190m": {"micro_batch": "32"},
    "380m": {"micro_batch": "16"},
    "640m": {"micro_batch": "8"},
}


def _source_and_split(script: Path, command: str) -> list[str]:
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("HEAD_GEOMETRY_", "HILBERT_", "SLURM_", "WANDB_"))
    }
    clean_env["DATA_ROOT"] = DATA_ROOT
    completed = subprocess.run(
        ["bash", "-c", f'source "$1"; {command}', "bash", str(script)],
        check=True,
        capture_output=True,
        env=clean_env,
    )
    return completed.stdout.decode().split("\0")[:-1]


def _worker_plan(scale: str, task: int) -> list[str]:
    return _source_and_split(
        WORKER,
        (
            f"head_geometry_resolve_plan {scale} {task} 731 2; "
            "printf '%s\\0' \"$model\" \"$micro_batch\" \"$seed\" "
            "\"$arm\" \"$lm_head_optimizer\" "
            "\"$run_id\" \"${train_args[@]}\""
        ),
    )


LAUNCH_TASKS = [(scale, task) for scale in SCALES for task in range(6)]


@pytest.mark.parametrize(("scale", "task"), LAUNCH_TASKS)
def test_all_launch_tasks_resolve_to_the_golden_training_plan(
    scale: str, task: int
) -> None:
    recipe = SCALES[scale]
    seed = task // 2
    arm = task % 2
    head_optimizer = "adamw" if arm == 0 else "rownorm"
    run_id = (
        f"hilbert-rownorm-{scale}-{head_optimizer}-20tpp-gb256-"
        f"mb{recipe['micro_batch']}-seed{seed}-grid731-task{task}-attempt2"
    )
    expected_train_args = [
        "--data",
        DATA_ROOT,
        "--lm-head-optimizer",
        head_optimizer,
        "--exact-head-diameter-every-log",
        "--exact-head-diameter-block-rows",
        "4096",
        "--hilbert-step-probe-tokens",
        "8192",
        "--global-batch",
        "256",
        "--tokens-per-parameter",
        "20",
        "--seed",
        str(seed),
        "--warmup-fraction",
        "0.1",
        "--min-lr-ratio",
        "0.1",
        "--max-gradient-norm",
        "1",
        "--log-every",
        "10",
        "--eval-every",
        "500",
        "--eval-tokens",
        "5242880",
        "--save-every",
        "0",
    ]

    assert _worker_plan(scale, task) == [
        scale,
        str(recipe["micro_batch"]),
        str(seed),
        str(arm),
        head_optimizer,
        run_id,
        *expected_train_args,
    ]


@pytest.mark.parametrize("scale", SCALES)
def test_all_submit_plans_cover_three_seeds_and_both_heads(scale: str) -> None:
    plan = _source_and_split(
        SUBMITTER,
        (
            f"head_geometry_resolve_submission {scale}; "
            "printf '%s\\0' \"$submit_model\" \"$job_name\" "
            "\"$array\" \"$task_range\""
        ),
    )
    assert plan == [
        scale,
        f"hilbert-rownorm-{scale}",
        "0-5%6",
        "0-5",
    ]


def test_worker_has_portable_single_node_contract() -> None:
    text = WORKER.read_text()
    common = (SCRIPTS / "cluster_job_common.sh").read_text()
    direct_runner = (SCRIPTS / "run.sh").read_text()

    assert "#SBATCH" not in text
    assert "HILBERT_ROWNORM_REPO" in text
    assert "HILBERT_STORAGE_ROOT" in text
    assert "DATA_ROOT" in text
    assert "EXPECTED_COMMIT" in text
    assert "verify_launch_checkout.sh" in text
    assert "require_training_hardware" in text
    assert "require_fineweb_dataset" in text
    assert "89000000000" in text
    assert "100000000" in text
    assert "87edf1844351c445e687ea865215b5fecff3a8b8052dff5b7263ea30712376f4" in text
    assert "does not permit truncated runs" in text
    assert "WANDB_DISABLE_GIT=true" in text
    assert "WANDB_DISABLE_CODE=true" in text
    assert "WANDB_PROJECT=${WANDB_PROJECT:-Hilbert-RowNorm}" in text
    assert "--max-steps" not in text
    assert "--optimizer-config" not in text
    assert "--head-diagnostics-config" not in text
    assert "--logit-diagnostics" not in text
    assert "--raw-head-gradient-snapshots" not in text
    assert "--exact-head-diameter-probes" not in text
    assert 'expected_torch = sys.argv[2:]' not in common
    assert 'EXPECTED_TORCH_VERSION:-2.13.0+cu126' in text
    assert 'EXPECTED_CUDA_VERSION:-12.6' in text
    assert "output_root=${OUTPUT_ROOT:-${PWD}/runs}" in direct_runner
    assert '--wandb-project "${WANDB_PROJECT:-Hilbert-RowNorm}"' in direct_runner


def test_submitter_uses_environment_configurable_slurm_resources() -> None:
    text = SUBMITTER.read_text()

    assert "verify_launch_checkout.sh" in text
    assert 'git -C "${repo}" rev-parse HEAD' in text
    for setting in (
        "SLURM_PARTITION",
        "SLURM_ACCOUNT",
        "SLURM_QOS",
        "SLURM_CONSTRAINT",
        "SLURM_GPUS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEMORY",
        "SLURM_TIME",
        "SLURM_ARRAY_CONCURRENCY",
    ):
        assert setting in text
    assert "--nodes=1" in text
    assert '--job-name="${job_name}"' in text
    assert '--array="${array}"' in text
    assert "HILBERT_ROWNORM_REPO=${repo}" in text
    assert "HILBERT_STORAGE_ROOT=${storage_root}" in text
    assert "DATA_ROOT=${data_root}" in text
    assert '"${worker}" "${submit_model}"' in text
    assert "! -w ${directory}" in text
    assert "! -x ${worker}" in text


@pytest.mark.parametrize(
    ("script", "function", "arguments", "message"),
    [
        (
            WORKER,
            "head_geometry_resolve_plan",
            "1b 0 1 0",
            "invalid head-geometry model",
        ),
        (
            WORKER,
            "head_geometry_resolve_plan",
            "190m 6 1 0",
            "invalid 190m task",
        ),
        (
            SUBMITTER,
            "head_geometry_resolve_submission",
            "1b",
            "invalid head-geometry model",
        ),
    ],
)
def test_resolvers_reject_unsupported_plans(
    script: Path, function: str, arguments: str, message: str
) -> None:
    env = {**os.environ, "DATA_ROOT": DATA_ROOT}
    completed = subprocess.run(
        ["bash", "-c", f'source "$1"; {function} {arguments}', "bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_only_supported_shell_entry_points_remain() -> None:
    assert {path.name for path in SCRIPTS.glob("*.sh")} == {
        "cluster_job_common.sh",
        "run.sh",
        "slurm_head_geometry.sh",
        "submit_head_geometry.sh",
        "verify_launch_checkout.sh",
    }
    assert os.access(WORKER, os.X_OK)
    assert os.access(SUBMITTER, os.X_OK)

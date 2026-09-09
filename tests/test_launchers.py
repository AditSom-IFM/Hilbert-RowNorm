from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from argparse import Namespace
from pathlib import Path

import pytest

from hilbert_rownorm.cli import build_parser

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


def _clean_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("HEAD_GEOMETRY_", "HILBERT_", "SLURM_", "WANDB_", "EXPECTED_", "FINEWEB_")
        )
        and key not in {"DATA_ROOT", "OUTPUT_ROOT", "MICRO_BATCH", "NPROC", "TRAIN_PYTHON"}
    }


def _source_and_split(script: Path, command: str) -> list[str]:
    clean_env = _clean_environment()
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
            'printf \'%s\\0\' "$model" "$micro_batch" "$seed" '
            '"$arm" "$lm_head_optimizer" '
            '"$run_id" "${train_args[@]}"'
        ),
    )


LAUNCH_TASKS = [(scale, task) for scale in SCALES for task in range(6)]


@pytest.mark.parametrize(("scale", "task"), LAUNCH_TASKS)
def test_all_launch_tasks_resolve_to_the_golden_training_plan(scale: str, task: int) -> None:
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
            'printf \'%s\\0\' "$submit_model" "$job_name" '
            '"$array" "$task_range"'
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
    assert "WANDB_PROJECT=${WANDB_PROJECT:-Hilbert-RowNorm-training}" in text
    assert "--max-steps" not in text
    assert "--optimizer-config" not in text
    assert "--head-diagnostics-config" not in text
    assert "--logit-diagnostics" not in text
    assert "--raw-head-gradient-snapshots" not in text
    assert "--exact-head-diameter-probes" not in text
    assert "expected_torch = sys.argv[2:]" not in common
    assert "EXPECTED_TORCH_VERSION:-2.13.0+cu126" in text
    assert "EXPECTED_CUDA_VERSION:-12.6" in text
    assert "output_root=${OUTPUT_ROOT:-${PWD}/runs}" in direct_runner
    assert '--wandb-project "${WANDB_PROJECT:-Hilbert-RowNorm-training}"' in direct_runner


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


INHERITED_WANDB_IDENTITY = {
    "WANDB_RUN_ID": "shared-run",
    "WANDB_RESUME": "allow",
    "WANDB_RESUME_FROM": "shared-run?_step=10",
    "WANDB_FORK_FROM": "shared-run?_step=10",
    "WANDB_SWEEP_ID": "shared-sweep",
    "WANDB_LAUNCH": "true",
    "WANDB_LAUNCH_CONFIG_PATH": "/unrelated/launch-config.json",
}


def _write_mock_executable(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(source))
    path.chmod(0o755)


@pytest.fixture
def launch_environment(tmp_path: Path) -> dict[str, str]:
    """Use the real checkout verifier, but never contact Slurm or load a GPU."""
    repo = tmp_path / "launch checkout"
    shutil.copytree(SCRIPTS, repo / "scripts")
    (repo / "src" / "hilbert_rownorm").mkdir(parents=True)
    (repo / "src" / "hilbert_rownorm" / "__init__.py").touch()
    for arguments in (
        ["init", "--quiet"],
        ["add", "."],
        [
            "-c",
            "user.name=Launcher Test",
            "-c",
            "user.email=launcher@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "Test checkout",
        ],
    ):
        subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True)

    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    _write_mock_executable(
        mock_bin / "sbatch",
        """
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        arguments = sys.argv[1:]
        if exit_code := os.environ.get("MOCK_SBATCH_EXIT"):
            print("mock sbatch refusal", file=sys.stderr)
            sys.exit(int(exit_code))
        environment = dict(os.environ)
        export = next(value.removeprefix("--export=") for value in arguments
                      if value.startswith("--export="))
        assert export.split(",")[0] == "ALL"
        # Slurm's ALL behavior: inherited environment takes precedence.
        for item in export.split(",")[1:]:
            key, value = item.split("=", 1)
            environment.setdefault(key, value)
        captured = {
            "arguments": arguments,
            "environment": {key: value for key, value in environment.items()
                            if key.startswith(("HILBERT_", "EXPECTED_", "WANDB_"))
                            or key == "DATA_ROOT"},
        }
        Path(os.environ["MOCK_SUBMISSION"]).write_text(json.dumps(captured))
        environment.update(SLURM_ARRAY_TASK_ID="1", SLURM_ARRAY_JOB_ID="731",
                           SLURM_RESTART_COUNT="2")
        result = subprocess.run(["bash", *arguments[-2:]], env=environment,
                                capture_output=True, text=True)
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            sys.exit(result.returncode)
        print("731;test-cluster")
        """,
    )
    _write_mock_executable(
        mock_bin / "training-python",
        """
        import json
        import os
        import sys
        from pathlib import Path

        arguments = sys.argv[1:]
        if arguments[:2] == ["-m", "torch.distributed.run"]:
            captured = {
                "arguments": arguments,
                "environment": {key: value for key, value in os.environ.items()
                                if key.startswith(("WANDB_", "EXPECTED_", "HILBERT_"))},
            }
            Path(os.environ["MOCK_TRAINING"]).write_text(json.dumps(captured))
        elif arguments[0] == "-c" or arguments[:2] == ["-m", "hilbert_rownorm.data"]:
            pass  # Only the hardware and dataset preflights are mocked.
        else:
            raise AssertionError(arguments)
        """,
    )
    return {
        **_clean_environment(),
        "PATH": f"{mock_bin}{os.pathsep}{os.environ['PATH']}",
        "HILBERT_ROWNORM_REPO": str(repo),
        "HILBERT_STORAGE_ROOT": str(tmp_path / "storage, with spaces"),
        "DATA_ROOT": DATA_ROOT,
        "TRAIN_PYTHON": str(mock_bin / "training-python"),
        "MOCK_SUBMISSION": str(tmp_path / "submission.json"),
        "MOCK_TRAINING": str(tmp_path / "training.json"),
    }


def _read_training(env: dict[str, str]) -> tuple[Namespace, dict[str, str]]:
    captured = json.loads(Path(env["MOCK_TRAINING"]).read_text())
    arguments = captured["arguments"]
    training_start = arguments.index("hilbert_rownorm.train") + 1
    return build_parser().parse_args(arguments[training_start:]), captured["environment"]


@pytest.mark.parametrize("repo_spelling", [".", "trailing-slash"])
@pytest.mark.parametrize("constraint", [None, "", "different_gpu"])
def test_full_submission_exports_validated_paths_fresh_pin_and_isolated_tracking(
    launch_environment: dict[str, str], repo_spelling: str, constraint: str | None
) -> None:
    env = launch_environment
    repo = Path(env["HILBERT_ROWNORM_REPO"])
    env.update(INHERITED_WANDB_IDENTITY)
    env["HILBERT_ROWNORM_REPO"] = "." if repo_spelling == "." else f"{repo}/"
    env["EXPECTED_COMMIT"] = "f" * 40
    if constraint is not None:
        env["SLURM_CONSTRAINT"] = constraint

    result = subprocess.run(
        ["bash", str(repo / "scripts" / "submit_head_geometry.sh"), "380m"],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    submission = json.loads(Path(env["MOCK_SUBMISSION"]).read_text())
    submitted_env = submission["environment"]
    expected_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert submitted_env["EXPECTED_COMMIT"] == expected_commit
    assert submitted_env["HILBERT_ROWNORM_REPO"] == str(repo.resolve())
    assert submitted_env["HILBERT_STORAGE_ROOT"] == env["HILBERT_STORAGE_ROOT"]
    assert submitted_env["DATA_ROOT"] == DATA_ROOT
    assert submitted_env["WANDB_PROJECT"] == "Hilbert-RowNorm-training"
    assert not INHERITED_WANDB_IDENTITY.keys() & submitted_env.keys()
    assert "--export=ALL" in submission["arguments"]
    constraints = [arg for arg in submission["arguments"] if arg.startswith("--constraint=")]
    expected_constraint = "nvidia_h200" if constraint is None else constraint
    assert constraints == ([f"--constraint={expected_constraint}"] if expected_constraint else [])
    assert "as 731_[0-5]" in result.stdout

    training, training_env = _read_training(env)
    assert training.model == "380m"
    assert training.optimizer == "muon"
    assert training.lm_head_optimizer == "rownorm"
    assert training.micro_batch == 16
    assert training.seed == 0
    assert training.wandb_project == "Hilbert-RowNorm-training"
    assert training.wandb_run_name.endswith("seed0-grid731-task1-attempt2")
    assert training.output == (
        Path(env["HILBERT_STORAGE_ROOT"])
        / "checkpoints"
        / "380m"
        / "muon"
        / training.wandb_run_name
    )
    assert not INHERITED_WANDB_IDENTITY.keys() & training_env.keys()


def test_submission_failure_propagates_without_reporting_success(
    launch_environment: dict[str, str],
) -> None:
    env = launch_environment
    env["MOCK_SBATCH_EXIT"] = "37"
    result = subprocess.run(
        [
            "bash",
            str(Path(env["HILBERT_ROWNORM_REPO"]) / "scripts" / "submit_head_geometry.sh"),
            "190m",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 37
    assert "mock sbatch refusal" in result.stderr
    assert "submitted" not in result.stdout
    assert not Path(env["MOCK_TRAINING"]).exists()


@pytest.mark.parametrize(
    "setting",
    [
        "--model",
        "--optimizer",
        "--output",
        "--micro-batch",
        "--wandb-project",
        "--wandb-entity",
        "--wandb-run-name",
    ],
)
@pytest.mark.parametrize("equals_form", [False, True])
def test_runner_rejects_trailing_wrapper_owned_arguments(
    launch_environment: dict[str, str], setting: str, equals_form: bool
) -> None:
    env = launch_environment
    env.update(MICRO_BATCH="8", OUTPUT_ROOT=f"{env['HILBERT_STORAGE_ROOT']}/direct")
    override = [f"{setting}=other"] if equals_form else [setting, "other"]
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run.sh"), "640m", "muon", "new-run", *override],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"{setting} is managed by run.sh" in result.stderr
    assert not Path(env["MOCK_TRAINING"]).exists()
    assert not Path(env["OUTPUT_ROOT"]).exists()


@pytest.mark.parametrize(
    "abbreviation",
    ["--mod", "--optim", "--out", "--micro-b", "--wandb-p", "--wandb-e", "--wandb-r"],
)
def test_parser_rejects_abbreviated_wrapper_overrides(
    abbreviation: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--model",
        "190m",
        "--optimizer",
        "muon",
        "--output",
        "new-run",
        "--micro-batch",
        "32",
        *_worker_plan("190m", 0)[6:],
    ]
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([*arguments, abbreviation, "other"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_runner_preserves_supported_training_arguments_and_clears_inherited_identity(
    launch_environment: dict[str, str],
) -> None:
    env = launch_environment
    env.update(INHERITED_WANDB_IDENTITY)
    env.update(
        MICRO_BATCH="8",
        OUTPUT_ROOT=f"{env['HILBERT_STORAGE_ROOT']}/direct",
        WANDB_ENTITY="my-team",
        WANDB_RUN_NAME="my-display-name",
        WANDB_MODE="online",
    )
    train_args = _worker_plan("640m", 0)[6:]
    subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run.sh"),
            "640m",
            "muon",
            "new-run",
            *train_args,
            "--max-steps",
            "5",
            "--gradient-checkpointing",
            "--wandb-mode",
            "disabled",
        ],
        env=env,
        check=True,
        capture_output=True,
    )

    training, training_env = _read_training(env)
    assert training.model == "640m"
    assert training.optimizer == "muon"
    assert training.micro_batch == 8
    assert training.output == Path(env["OUTPUT_ROOT"]) / "640m" / "muon" / "new-run"
    assert training.max_steps == 5
    assert training.gradient_checkpointing
    assert training.wandb_mode == "disabled"
    assert training.wandb_entity == "my-team"
    assert training.wandb_run_name == "my-display-name"
    assert training.wandb_project == "Hilbert-RowNorm-training"
    assert not INHERITED_WANDB_IDENTITY.keys() & training_env.keys()


def test_runner_refuses_an_existing_run_directory(launch_environment: dict[str, str]) -> None:
    env = launch_environment
    env.update(MICRO_BATCH="8", OUTPUT_ROOT=f"{env['HILBERT_STORAGE_ROOT']}/direct")
    output = Path(env["OUTPUT_ROOT"]) / "640m" / "muon" / "existing-run"
    output.mkdir(parents=True)
    checkpoint = output / "checkpoint.pt"
    checkpoint.write_bytes(b"previous checkpoint")

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run.sh"), "640m", "muon", "existing-run"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "refusing to reuse existing run directory" in result.stderr
    assert checkpoint.read_bytes() == b"previous checkpoint"
    assert not Path(env["MOCK_TRAINING"]).exists()

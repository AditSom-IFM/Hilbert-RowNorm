"""Command-line arguments for the pretraining harness."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_WANDB_PROJECT = "Hilbert-RowNorm-training"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without parsing process state."""

    parser = argparse.ArgumentParser(
        description="Single-node trainer for the final LM-head optimizer experiments.",
        allow_abbrev=False,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), required=True)
    parser.add_argument(
        "--lm-head-optimizer", choices=("adamw", "rownorm"), required=True
    )
    parser.add_argument("--exact-head-diameter-every-log", action="store_true")
    parser.add_argument("--exact-head-diameter-block-rows", type=int, default=4_096)
    parser.add_argument("--hilbert-step-probe-tokens", type=int, default=0)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--tokens-per-parameter", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="stop early without changing the full-run learning-rate schedule",
    )
    parser.add_argument("--warmup-fraction", type=float, required=True)
    parser.add_argument("--min-lr-ratio", type=float, required=True)
    parser.add_argument("--max-gradient-norm", type=float, required=True)
    parser.add_argument("--no-gradient-clipping", action="store_true")
    parser.add_argument("--log-every", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--eval-tokens", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    return parser

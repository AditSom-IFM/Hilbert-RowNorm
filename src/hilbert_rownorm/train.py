"""Entry point for the Hilbert-RowNorm pretraining harness."""

from __future__ import annotations

from .cli import DEFAULT_WANDB_PROJECT as DEFAULT_WANDB_PROJECT
from .cli import build_parser as _parser
from .distributed import initialize_distributed as _distributed
from .runner import run_pretraining


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    rank, world_size, device = _distributed(args.require_cuda)
    run_pretraining(args, rank, world_size, device)


if __name__ == "__main__":
    main()

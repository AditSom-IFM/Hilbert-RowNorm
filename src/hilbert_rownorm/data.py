"""Read-only batches from the pre-tokenized FineWeb ``.bin`` shards."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

FINEWEB_MAGIC = 20240520
FINEWEB_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = 1024


@dataclass(frozen=True)
class FineWebShard:
    path: Path
    num_tokens: int


@dataclass(frozen=True)
class FineWebDatasetIdentity:
    train_shards: int
    validation_shards: int
    train_tokens: int
    validation_tokens: int
    metadata_sha256: str


def inspect_shard(path: str | Path) -> FineWebShard:
    """Validate a shard without loading its token payload."""

    path = Path(path)
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<i4", count=HEADER_INTS)
    if len(header) != HEADER_INTS:
        raise ValueError(f"truncated FineWeb header: {path}")
    magic, version, num_tokens = map(int, header[:3])
    if magic != FINEWEB_MAGIC or version != FINEWEB_VERSION:
        raise ValueError(f"invalid FineWeb header: {path}")
    expected_bytes = HEADER_BYTES + 2 * num_tokens
    if num_tokens < 2 or path.stat().st_size != expected_bytes:
        raise ValueError(f"invalid FineWeb payload size: {path}")
    return FineWebShard(path, num_tokens)


def _discover(root: str | Path, pattern: str) -> tuple[FineWebShard, ...]:
    paths = sorted(Path(root).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no FineWeb shards match {pattern!r} under {root}")
    return tuple(inspect_shard(path) for path in paths)


def inspect_dataset(root: str | Path) -> FineWebDatasetIdentity:
    """Identify a FineWeb export from shard names, sizes, and validated headers."""

    train = _discover(root, "fineweb_train_*.bin")
    validation = _discover(root, "fineweb_val_*.bin")
    shards = sorted((*train, *validation), key=lambda shard: shard.path.name)
    metadata = "".join(
        f"{shard.path.name}\t{shard.path.stat().st_size}\t{shard.num_tokens}\n"
        for shard in shards
    )
    return FineWebDatasetIdentity(
        train_shards=len(train),
        validation_shards=len(validation),
        train_tokens=sum(shard.num_tokens for shard in train),
        validation_tokens=sum(shard.num_tokens for shard in validation),
        metadata_sha256=hashlib.sha256(metadata.encode()).hexdigest(),
    )


def _parse_identity_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the metadata identity of pre-tokenized FineWeb shards."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-shards", type=int, required=True)
    parser.add_argument("--validation-shards", type=int, required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--validation-tokens", type=int, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    return parser.parse_args()


def _validate_dataset_main() -> None:
    args = _parse_identity_args()
    actual = inspect_dataset(args.root)
    expected = FineWebDatasetIdentity(
        train_shards=args.train_shards,
        validation_shards=args.validation_shards,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        metadata_sha256=args.metadata_sha256,
    )
    if actual != expected:
        raise SystemExit(f"FineWeb dataset identity mismatch: expected {expected}, found {actual}")
    print(f"FineWeb dataset verified: {actual}", flush=True)


class FineWebBatchLoader:
    """Yield rank-local slices of deterministically shuffled optimizer batches.

    Shuffling happens at the full optimizer-batch boundary, so changing the
    microbatch/accumulation split does not change the examples in an update.
    Set ``shuffle=False`` for a fixed sequential validation prefix.
    """

    def __init__(
        self,
        root: str | Path,
        pattern: str,
        *,
        batch_size: int,
        sequence_length: int,
        rank: int = 0,
        world_size: int = 1,
        gradient_accumulation_steps: int = 1,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if min(batch_size, sequence_length, world_size, gradient_accumulation_steps) < 1:
            raise ValueError("batch, sequence, world size, and accumulation must be positive")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        self.shards = _discover(root, pattern)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.global_batch_size = batch_size * world_size
        self.optimizer_batch_size = self.global_batch_size * gradient_accumulation_steps
        self.global_tokens_per_batch = self.global_batch_size * sequence_length
        self._block_counts = tuple(
            (shard.num_tokens - 1) // sequence_length for shard in self.shards
        )
        if not any(count >= self.optimizer_batch_size for count in self._block_counts):
            raise ValueError("no shard contains one complete optimizer batch")
        self._tokens: np.memmap | None = None
        self.reset()

    def reset(self, epoch: int = 0) -> None:
        """Return to a deterministic epoch (use epoch zero for validation)."""

        self._close()
        self.epoch = epoch
        self._shard_order = np.arange(len(self.shards))
        if self.shuffle:
            np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 0])).shuffle(
                self._shard_order
            )
        self._shard_position = 0
        self._open_shard()

    def next_batch(self) -> tuple[Tensor, Tensor]:
        while self._block_position + self.global_batch_size > len(self._block_order):
            self._advance_shard()
        start = self._block_position + self.rank * self.batch_size
        indices = self._block_order[start : start + self.batch_size]
        offsets = indices[:, None] * self.sequence_length + np.arange(
            self.sequence_length + 1
        )
        assert self._tokens is not None
        batch = torch.from_numpy(np.asarray(self._tokens[offsets], dtype=np.int64))
        self._block_position += self.global_batch_size
        return batch[:, :-1], batch[:, 1:]

    def _open_shard(self) -> None:
        shard_index = int(self._shard_order[self._shard_position])
        shard = self.shards[shard_index]
        groups = self._block_counts[shard_index] // self.optimizer_batch_size
        group_order = np.arange(groups)
        if self.shuffle:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, shard_index, 1])
            )
            rng.shuffle(group_order)
        within_group = np.arange(self.optimizer_batch_size)
        self._block_order = (
            group_order[:, None] * self.optimizer_batch_size + within_group
        ).reshape(-1)
        self._tokens = np.memmap(
            shard.path, dtype="<u2", mode="r", offset=HEADER_BYTES, shape=(shard.num_tokens,)
        )
        self._block_position = 0

    def _advance_shard(self) -> None:
        self._close()
        self._shard_position += 1
        if self._shard_position == len(self.shards):
            self.reset(self.epoch + 1)
        else:
            self._open_shard()

    def _close(self) -> None:
        if self._tokens is not None and self._tokens._mmap is not None:
            self._tokens._mmap.close()
        self._tokens = None


if __name__ == "__main__":
    _validate_dataset_main()

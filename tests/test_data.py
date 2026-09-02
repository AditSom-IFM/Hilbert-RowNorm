from pathlib import Path

import numpy as np
import pytest
import torch

from hilbert_rownorm.data import (
    FINEWEB_MAGIC,
    FINEWEB_VERSION,
    HEADER_INTS,
    FineWebBatchLoader,
    FineWebDatasetIdentity,
    inspect_dataset,
    inspect_shard,
)


def write_shard(path: Path, tokens: np.ndarray, *, magic: int = FINEWEB_MAGIC) -> None:
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[:3] = magic, FINEWEB_VERSION, len(tokens)
    with path.open("wb") as handle:
        header.tofile(handle)
        tokens.astype("<u2").tofile(handle)


def test_header_and_next_token_targets(tmp_path: Path) -> None:
    path = tmp_path / "train.bin"
    write_shard(path, np.arange(21, dtype=np.uint16))
    assert inspect_shard(path).num_tokens == 21
    loader = FineWebBatchLoader(
        tmp_path, "*.bin", batch_size=2, sequence_length=4, shuffle=False
    )
    inputs, targets = loader.next_batch()
    assert torch.equal(inputs, torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
    assert torch.equal(targets, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))

    write_shard(path, np.arange(5, dtype=np.uint16), magic=123)
    with pytest.raises(ValueError, match="header"):
        inspect_shard(path)


def test_dataset_identity_uses_validated_header_metadata(tmp_path: Path) -> None:
    write_shard(tmp_path / "fineweb_train_000001.bin", np.arange(21, dtype=np.uint16))
    write_shard(tmp_path / "fineweb_val_000000.bin", np.arange(9, dtype=np.uint16))

    identity = inspect_dataset(tmp_path)

    assert identity == FineWebDatasetIdentity(
        train_shards=1,
        validation_shards=1,
        train_tokens=21,
        validation_tokens=9,
        metadata_sha256="d184795448e12cb3c4d4d589f96111db45348d1812fb9c9bb9ecf165568d6515",
    )


def test_optimizer_batch_shuffle_is_split_invariant(tmp_path: Path) -> None:
    write_shard(tmp_path / "train.bin", np.arange(129, dtype=np.uint16))
    common = dict(root=tmp_path, pattern="*.bin", sequence_length=4, seed=31)
    accumulated = FineWebBatchLoader(
        batch_size=2, gradient_accumulation_steps=2, **common
    )
    full = FineWebBatchLoader(batch_size=4, **common)
    first = torch.cat([accumulated.next_batch()[0], accumulated.next_batch()[0]])
    full_first = full.next_batch()[0]
    assert torch.equal(first, full_first)

    repeated = FineWebBatchLoader(batch_size=4, **common)
    assert torch.equal(full_first, repeated.next_batch()[0])


def test_ddp_ranks_get_disjoint_slices(tmp_path: Path) -> None:
    write_shard(tmp_path / "train.bin", np.arange(33, dtype=np.uint16))
    common = dict(
        root=tmp_path,
        pattern="*.bin",
        batch_size=1,
        sequence_length=4,
        world_size=2,
        shuffle=False,
    )
    rank_zero = FineWebBatchLoader(rank=0, **common).next_batch()[0]
    rank_one = FineWebBatchLoader(rank=1, **common).next_batch()[0]
    assert torch.equal(rank_zero, torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(rank_one, torch.tensor([[4, 5, 6, 7]]))


def test_validation_is_a_fixed_sequential_prefix(tmp_path: Path) -> None:
    write_shard(tmp_path / "train.bin", np.arange(65, dtype=np.uint16))
    write_shard(tmp_path / "val.bin", np.arange(100, 133, dtype=np.uint16))
    validation = FineWebBatchLoader(
        tmp_path,
        "val.bin",
        batch_size=2,
        sequence_length=4,
        seed=99,
        shuffle=False,
    )
    expected = validation.next_batch()
    validation.next_batch()
    validation.reset()
    actual = validation.next_batch()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))

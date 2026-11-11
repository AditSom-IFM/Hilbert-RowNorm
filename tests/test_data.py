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


def _three_uneven_shards(root: Path) -> None:
    for index, tokens in enumerate((97, 161, 65)):
        start = 1_000 * index
        write_shard(root / f"train_{index}.bin", np.arange(start, start + tokens, dtype=np.uint16))


def _updates(loader: FineWebBatchLoader, count: int, micro_steps: int = 1) -> list[torch.Tensor]:
    return [
        torch.cat([loader.next_batch()[0] for _ in range(micro_steps)])
        for _ in range(count)
    ]


def test_saved_position_continues_the_exact_batch_sequence(tmp_path: Path) -> None:
    _three_uneven_shards(tmp_path)
    common = dict(root=tmp_path, pattern="*.bin", batch_size=8, sequence_length=4, seed=7)
    reference = FineWebBatchLoader(**common)
    states = []
    batches = []
    # 3 + 5 + 2 optimizer batches per epoch: 30 updates cross shard and epoch boundaries.
    for _ in range(30):
        states.append(reference.state_dict())
        batches.append(reference.next_batch()[0])
    assert {state["epoch"] for state in states} == {0, 1, 2}

    for start in range(30):
        resumed = FineWebBatchLoader(**common)
        resumed.load_state_dict(states[start])
        assert resumed.state_dict() == states[start]
        for expected in batches[start:]:
            assert torch.equal(resumed.next_batch()[0], expected)


def test_saved_position_is_portable_across_microbatch_and_rank_splits(tmp_path: Path) -> None:
    _three_uneven_shards(tmp_path)
    common = dict(root=tmp_path, pattern="*.bin", sequence_length=4, seed=7)
    whole = FineWebBatchLoader(batch_size=4, **common)
    _updates(whole, 6)
    state = whole.state_dict()
    expected = _updates(whole, 8)

    accumulated = FineWebBatchLoader(batch_size=2, gradient_accumulation_steps=2, **common)
    accumulated.load_state_dict(state)
    for actual, update in zip(_updates(accumulated, 8, micro_steps=2), expected, strict=True):
        assert torch.equal(actual, update)

    ranks = [
        FineWebBatchLoader(
            batch_size=1, rank=rank, world_size=2, gradient_accumulation_steps=2, **common
        )
        for rank in range(2)
    ]
    for loader in ranks:
        loader.load_state_dict(state)
    for update in expected:
        rows = []
        for _ in range(2):
            rows.extend(loader.next_batch()[0] for loader in ranks)
        assert torch.equal(torch.cat(rows), update)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"seed": 8}, "different data order"),
        ({"optimizer_batch_size": 8}, "different data order"),
        ({"shards": 2}, "different data order"),
        ({"shard_position": 3}, "out of range"),
        ({"block_position": 2}, "optimizer-batch boundary"),
        ({"block_position": 400}, "optimizer-batch boundary"),
        ({"epoch": -1}, "non-negative integers"),
        ({"epoch": 1.0}, "non-negative integers"),
        ({"extra": 0}, "unexpected keys"),
    ],
)
def test_inconsistent_saved_positions_are_rejected(tmp_path: Path, change, message) -> None:
    _three_uneven_shards(tmp_path)
    common = dict(
        root=tmp_path,
        pattern="*.bin",
        batch_size=2,
        sequence_length=4,
        seed=7,
        gradient_accumulation_steps=2,
    )
    state = {**FineWebBatchLoader(**common).state_dict(), **change}
    with pytest.raises(ValueError, match=message):
        FineWebBatchLoader(**common).load_state_dict(state)


def test_validation_loader_cannot_resume_and_identity_ignores_the_directory(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    for root in (first, second):
        root.mkdir()
        _three_uneven_shards(root)
    common = dict(pattern="*.bin", batch_size=2, sequence_length=4)
    validation = FineWebBatchLoader(first, shuffle=False, **common)
    with pytest.raises(ValueError, match="shuffled training loader"):
        validation.load_state_dict(validation.state_dict())
    assert FineWebBatchLoader(first, **common).identity() == FineWebBatchLoader(
        second, **common
    ).identity()
    write_shard(second / "train_2.bin", np.arange(66, dtype=np.uint16))
    assert FineWebBatchLoader(first, **common).identity() != FineWebBatchLoader(
        second, **common
    ).identity()

import numpy as np
import pytest

from morphosml._core import IngestionCursor
from morphosml.data import DataLoader, MMapDataset


def test_cursor_serialization():
    cursor = IngestionCursor()
    cursor.epoch = 3
    cursor.sample_offset = 1280
    cursor.checksum = 9876543210123456789

    token = cursor.to_string()
    assert token == "3:1280:9876543210123456789"

    restored = IngestionCursor.from_string(token)
    assert restored.epoch == 3
    assert restored.sample_offset == 1280
    assert restored.checksum == 9876543210123456789


def test_dataloader_checkpoint_and_resume(tmp_path):
    filepath = str(tmp_path / "fault_tolerance.mldat")
    raw_data = np.arange(400, dtype=np.float64).reshape(100, 4)
    MMapDataset.dump(filepath, raw_data)

    dataset = MMapDataset(filepath)
    batch_size = 10

    # 1. Run full loader to record all batches
    full_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=999,
        epoch=0,
        prefetch_batches=0,
    )
    all_batches = [np.asarray(b) for b in full_loader]
    assert len(all_batches) == 10

    # 2. Run loader partially, stop after 4 batches (40 samples)
    partial_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=999,
        epoch=0,
        prefetch_batches=0,
    )
    resumed_batches = []
    cursor = None
    for i, batch in enumerate(partial_loader):
        resumed_batches.append(np.asarray(batch))
        if i == 3:  # 4th batch (samples 0-39 consumed)
            cursor = partial_loader.get_cursor()
            break

    assert cursor is not None
    assert cursor.epoch == 0
    assert cursor.sample_offset == 40
    assert cursor.checksum == dataset.checksum

    # 3. Simulate process restart: create a fresh loader and resume
    new_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=999,
        epoch=0,
        prefetch_batches=0,
    )
    new_loader.resume_from(cursor)

    for batch in new_loader:
        resumed_batches.append(np.asarray(batch))

    # All batches from partial + resumed must match the full run
    assert len(resumed_batches) == len(all_batches)
    for b_resumed, b_expected in zip(resumed_batches, all_batches):
        np.testing.assert_array_equal(b_resumed, b_expected)


def test_resume_checksum_mismatch(tmp_path):
    filepath = str(tmp_path / "dataset1.mldat")
    MMapDataset.dump(filepath, np.zeros((10, 2), dtype=np.float64))
    dataset = MMapDataset(filepath)

    loader = DataLoader(dataset, batch_size=2)
    bad_cursor = IngestionCursor()
    bad_cursor.epoch = 0
    bad_cursor.sample_offset = 2
    bad_cursor.checksum = 12345  # Mismatched checksum

    with pytest.raises(ValueError, match="Dataset checksum mismatch"):
        loader.resume_from(bad_cursor)

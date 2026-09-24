import numpy as np

from morphosml._core import IdempotentSampler
from morphosml.data import DataLoader, MMapDataset


def test_idempotent_sampler_determinism():
    sampler1 = IdempotentSampler(seed=42)
    sampler2 = IdempotentSampler(seed=42)

    indices1 = sampler1.generate_indices(100, epoch=0)
    indices2 = sampler2.generate_indices(100, epoch=0)

    assert indices1 == indices2
    assert len(indices1) == 100
    assert set(indices1) == set(range(100))

    # Epoch 1 must produce a distinct deterministic permutation
    indices_epoch1 = sampler1.generate_indices(100, epoch=1)
    assert indices_epoch1 != indices1
    assert len(indices_epoch1) == 100
    assert set(indices_epoch1) == set(range(100))

    # Epoch 0 repeated must still match
    assert sampler1.generate_indices(100, epoch=0) == indices1


def test_dataloader_idempotency(tmp_path):
    filepath = str(tmp_path / "idempotent_test.mldat")
    raw_data = np.arange(200, dtype=np.float64).reshape(50, 4)
    MMapDataset.dump(filepath, raw_data)

    dataset = MMapDataset(filepath)

    loader1 = DataLoader(
        dataset, batch_size=8, shuffle=True, seed=123, epoch=0, prefetch_batches=0
    )
    batches1 = [np.asarray(b) for b in loader1]

    loader2 = DataLoader(
        dataset, batch_size=8, shuffle=True, seed=123, epoch=0, prefetch_batches=0
    )
    batches2 = [np.asarray(b) for b in loader2]

    assert len(batches1) == len(batches2)
    for b1, b2 in zip(batches1, batches2):
        np.testing.assert_array_equal(b1, b2)

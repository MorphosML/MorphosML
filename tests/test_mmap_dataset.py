import numpy as np
import pytest

from morphosml.data import MMapDataset, from_numpy


def test_mmap_dataset_creation_and_reading(tmp_path):
    filepath = str(tmp_path / "test_data.mldat")
    raw_data = np.arange(100, dtype=np.float64).reshape(20, 5)

    MMapDataset.dump(filepath, raw_data)

    dataset = MMapDataset(filepath)
    assert len(dataset) == 20
    assert dataset.rows == 20
    assert dataset.cols == 5
    assert dataset.shape == (20, 5)
    assert dataset.checksum != 0
    assert dataset.filepath == filepath

    # Full view zero-copy conversion to numpy
    np_view = dataset.to_numpy()
    np.testing.assert_array_equal(np_view, raw_data)

    # Slice access
    slice_view = dataset.get_slice(5, 10)
    assert slice_view.rows() == 10
    assert slice_view.cols() == 5
    assert slice_view(0, 0) == raw_data[5, 0]
    assert slice_view(9, 4) == raw_data[14, 4]


def test_mmap_dataset_indexing(tmp_path):
    filepath = str(tmp_path / "test_indexing.mldat")
    raw_data = np.linspace(0, 1, 50, dtype=np.float64).reshape(10, 5)

    MMapDataset.dump(filepath, raw_data)
    dataset = MMapDataset(filepath)

    # Single row indexing returns TensorView of 1 row
    row3 = dataset[3]
    assert row3.rows() == 1
    assert row3.cols() == 5
    assert row3(0, 0) == raw_data[3, 0]

    # Slice indexing
    sub_slice = dataset[2:6]
    assert sub_slice.rows() == 4
    assert sub_slice(0, 0) == raw_data[2, 0]

    # Negative index
    last_row = dataset[-1]
    assert last_row(0, 0) == raw_data[9, 0]

    with pytest.raises(IndexError):
        _ = dataset[100]


def test_mmap_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        _ = MMapDataset("non_existent_file_path.mldat")


def test_numpy_dataset_creation():
    X = np.ones((10, 4), dtype=np.float64)
    y = np.arange(10, dtype=np.float64)

    ds = from_numpy(X, y)
    assert len(ds) == 10
    assert ds.shape == (10, 4)
    item_x, item_y = ds[0]
    assert item_x.shape == (4,)
    assert item_y == 0.0

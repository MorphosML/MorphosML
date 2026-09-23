import numpy as np
import pytest

from morphosml import Matrix
from morphosml._core import tensor_view_from_numpy


def test_tensor_view_from_numpy():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    tv = tensor_view_from_numpy(arr)

    assert tv.rows() == 2
    assert tv.cols() == 3
    assert tv.size() == 6
    assert not tv.empty()
    assert tv.stride() == 3

    assert tv(0, 0) == 1.0
    assert tv(0, 2) == 3.0
    assert tv(1, 1) == 5.0
    assert tv[1, 2] == 6.0


def test_tensor_view_bounds():
    arr = np.zeros((3, 3), dtype=np.float64)
    tv = tensor_view_from_numpy(arr)

    with pytest.raises(IndexError):
        _ = tv(3, 0)

    with pytest.raises(IndexError):
        _ = tv[0, 3]


def test_tensor_view_to_matrix():
    arr = np.array([[10.5, 20.5], [30.5, 40.5]], dtype=np.float64)
    tv = tensor_view_from_numpy(arr)

    mat = tv.to_matrix()
    assert isinstance(mat, Matrix)
    assert mat.rows() == 2
    assert mat.cols() == 2
    assert mat(0, 0) == 10.5
    assert mat(1, 1) == 40.5


def test_tensor_view_buffer_protocol_zero_copy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tv = tensor_view_from_numpy(arr)

    # Convert TensorView back to numpy via buffer protocol
    np_view = np.asarray(tv)
    assert np_view.shape == (2, 2)
    np.testing.assert_array_equal(np_view, arr)

    # Zero-copy verification: modifying underlying arr reflects in tv
    arr[0, 0] = 99.0
    assert tv(0, 0) == 99.0
    assert np_view[0, 0] == 99.0

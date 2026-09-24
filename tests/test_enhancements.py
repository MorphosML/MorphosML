import numpy as np
import pytest

from morphosml import (
    KNN,
    DataLoader,
    Dataset,
    IdempotentSampler,
    IngestionCursor,
    LinearRegression,
    LogisticRegression,
    Matrix,
    MMapBuffer,
    MMapDataset,
    NumpyDataset,
    TensorView,
    Vector,
    from_numpy,
    tensor_view_from_numpy,
)


def test_top_level_exports():
    """Verify all HPC ingestion, view, and data classes are accessible directly from morphosml."""
    assert DataLoader is not None
    assert Dataset is not None
    assert MMapDataset is not None
    assert NumpyDataset is not None
    assert from_numpy is not None
    assert IngestionCursor is not None
    assert IdempotentSampler is not None
    assert MMapBuffer is not None
    assert TensorView is not None
    assert tensor_view_from_numpy is not None


def test_tensor_view_constructors_and_shape():
    """Verify TensorView construction from NumPy ndarray and from Matrix."""
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

    # 1. From NumPy array directly via constructor
    tv_np = TensorView(arr)
    assert tv_np.shape == (2, 3)
    assert tv_np.rows() == 2
    assert tv_np.cols() == 3
    assert tv_np.size() == 6
    assert not tv_np.empty()
    assert tv_np[0, 1] == 2.0

    # 2. From Matrix directly via constructor
    mat = Matrix([[10.0, 20.0], [30.0, 40.0]])
    tv_mat = TensorView(mat)
    assert tv_mat.shape == (2, 2)
    assert tv_mat[1, 0] == 30.0

    # 3. Via Matrix.view() method
    tv_view = mat.view()
    assert isinstance(tv_view, TensorView)
    assert tv_view.shape == (2, 2)
    assert tv_view == tv_mat


def test_tensor_view_slicing_and_subview():
    """Verify zero-copy row slicing and 2D subview on TensorView."""
    arr = np.arange(20, dtype=np.float64).reshape(5, 4)
    tv = TensorView(arr)

    # Row slicing via method and Python slice syntax
    row_slice1 = tv.slice(1, 3)
    assert row_slice1.shape == (3, 4)
    assert row_slice1[0, 0] == 4.0
    assert row_slice1[2, 3] == 15.0

    row_slice2 = tv[1:4]
    assert isinstance(row_slice2, TensorView)
    assert row_slice2.shape == (3, 4)
    assert row_slice2 == row_slice1

    # 2D subview: start_row=1, num_rows=2, start_col=1, num_cols=2
    sub = tv.subview(1, 2, 1, 2)
    assert sub.shape == (2, 2)
    assert sub[0, 0] == 5.0
    assert sub[0, 1] == 6.0
    assert sub[1, 0] == 9.0
    assert sub[1, 1] == 10.0


def test_row_indexing_and_vector_extraction():
    """Verify row extraction on Matrix and TensorView returning Vector."""
    mat = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert mat.shape == (2, 3)

    # Matrix.row() and mat[i]
    r0 = mat.row(0)
    assert isinstance(r0, Vector)
    assert r0.size() == 3
    assert r0.to_list() == [1.0, 2.0, 3.0]

    r1 = mat[1]
    assert isinstance(r1, Vector)
    assert r1.to_list() == [4.0, 5.0, 6.0]

    # TensorView.row() and tv[i]
    tv = mat.view()
    tv_r0 = tv.row(0)
    assert isinstance(tv_r0, Vector)
    assert tv_r0 == r0

    tv_r1 = tv[1]
    assert isinstance(tv_r1, Vector)
    assert tv_r1 == r1


def test_tensor_view_to_numpy():
    """Verify materialization from TensorView to NumPy array."""
    arr = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float64)
    tv = TensorView(arr)

    np_out = tv.to_numpy()
    assert isinstance(np_out, np.ndarray)
    assert np_out.shape == (2, 2)
    np.testing.assert_array_equal(np_out, arr)


def test_vector_numpy_constructor_and_buffer():
    """Verify Vector initialization from 1D NumPy array and buffer protocol."""
    np_v = np.array([2.5, 4.5, 6.5], dtype=np.float64)
    v = Vector(np_v)
    assert v.size() == 3
    assert v[0] == 2.5
    assert v[1] == 4.5
    assert v[2] == 6.5

    # Buffer protocol: np.asarray(v)
    v_arr = np.asarray(v)
    assert v_arr.shape == (3,)
    np.testing.assert_array_equal(v_arr, np_v)


def test_equality_operators():
    """Verify __eq__ and != operators across Vector, Matrix, TensorView, IngestionCursor."""
    # Vector
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([1.0, 2.0, 3.0])
    v3 = Vector([1.0, 2.0, 4.0])
    assert v1 == v2
    assert v1 != v3

    # Matrix
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m3 = Matrix([[1.0, 2.0], [3.0, 5.0]])
    assert m1 == m2
    assert m1 != m3

    # TensorView
    tv1 = m1.view()
    tv2 = m2.view()
    tv3 = m3.view()
    assert tv1 == tv2
    assert tv1 != tv3

    # IngestionCursor
    c1 = IngestionCursor.from_string("2:500:99999")
    c2 = IngestionCursor.from_string("2:500:99999")
    c3 = IngestionCursor.from_string("2:501:99999")
    assert c1 == c2
    assert c1 != c3


def test_string_representations():
    """Verify human-readable and clean __repr__ on core objects."""
    v = Vector([1.0, 2.0])
    assert repr(v) == "Vector([1.000000, 2.000000])"
    assert str(v) == "[1.000000, 2.000000]"

    m = Matrix(3, 4)
    assert repr(m) == "<Matrix shape=(3, 4)>"

    tv = m.view()
    assert repr(tv) == "<TensorView shape=(3, 4) stride=4>"

    c = IngestionCursor.from_string("1:100:42")
    assert repr(c) == "<IngestionCursor epoch=1 sample_offset=100 checksum=42>"


def test_cursor_error_handling():
    """Verify IngestionCursor.from_string rejects malformed or corrupted strings."""
    with pytest.raises(ValueError):
        IngestionCursor.from_string("invalid_token")

    with pytest.raises(ValueError):
        IngestionCursor.from_string("1:2")

    with pytest.raises(ValueError):
        IngestionCursor.from_string("1:2:3:extra")


def test_model_is_fitted_lifecycle():
    """Verify is_fitted lifecycle across KNN, LinearRegression, and LogisticRegression."""
    # KNN
    knn = KNN(k=1)
    assert not knn.is_fitted
    knn.fit(np.array([[0.0, 0.0], [1.0, 1.0]]), [0, 1])
    assert knn.is_fitted

    # LinearRegression
    lr = LinearRegression(learning_rate=0.01, epochs=10)
    assert not lr.is_fitted
    lr.fit(np.array([[1.0], [2.0]]), [2.0, 4.0])
    assert lr.is_fitted

    # LogisticRegression
    log_reg = LogisticRegression(learning_rate=0.01, epochs=10)
    assert not log_reg.is_fitted
    log_reg.fit(np.array([[1.0], [2.0]]), [0, 1])
    assert log_reg.is_fitted

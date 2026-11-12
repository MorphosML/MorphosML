import gc
import struct
from pathlib import Path

import numpy as np
import pytest

import morphosml as ml
from morphosml.calculus import derivative, gradient, hessian, integrate, jacobian
from morphosml.data import DataLoader, Dataset
from morphosml.latex import matrix_to_latex, vector_to_latex


def test_mldat_header_size_and_overflow_rejection(tmp_path: Path) -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    valid_path = tmp_path / "valid.mldat"
    ml.MMapBuffer.write_mldat_numpy(str(valid_path), arr)

    # 64-byte FileHeader + 6 * 8 = 112 bytes
    assert valid_path.stat().st_size == 64 + arr.size * 8

    buf = ml.MMapBuffer(str(valid_path))
    assert buf.rows() == 3
    assert buf.cols() == 2

    with pytest.raises(IndexError):
        buf.get_slice(1, (1 << 64) - 1)

    # Craft header whose rows * cols * 8 wraps around 64-bit size_t to 0
    bad_overflow_path = tmp_path / "overflow.mldat"
    overflow_header = struct.pack(
        "<4sIQQQ32s", b"MML\x00", 1, 1 << 62, 4, 0, b"\x00" * 32
    )
    assert len(overflow_header) == 64
    bad_overflow_path.write_bytes(overflow_header)

    with pytest.raises(RuntimeError):
        ml.MMapBuffer(str(bad_overflow_path))

    # Unsupported version
    bad_version_path = tmp_path / "bad_version.mldat"
    bad_version_header = struct.pack(
        "<4sIQQQ32s", b"MML\x00", 99, 0, 0, 0, b"\x00" * 32
    )
    bad_version_path.write_bytes(bad_version_header)

    with pytest.raises(RuntimeError):
        ml.MMapBuffer(str(bad_version_path))

    # Opening a directory instead of a regular file
    with pytest.raises(RuntimeError):
        ml.MMapBuffer(str(tmp_path))


def test_tensor_view_layout_overflow_and_lifetime(tmp_path: Path) -> None:
    arr_f32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        ml.TensorView(arr_f32)
    with pytest.raises(ValueError):
        ml.tensor_view_from_numpy(arr_f32)

    arr_fortran = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    with pytest.raises(ValueError):
        ml.tensor_view_from_numpy(arr_fortran)

    arr_reversed = np.arange(8, dtype=np.float64).reshape(4, 2)[::-1]
    with pytest.raises(ValueError):
        ml.tensor_view_from_numpy(arr_reversed)

    arr_f64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tv = ml.tensor_view_from_numpy(arr_f64)
    with pytest.raises(IndexError):
        tv.slice(1, (1 << 64) - 1)
    with pytest.raises(IndexError):
        tv.subview(1, (1 << 64) - 1, 0, 1)
    with pytest.raises(IndexError):
        tv.subview(0, 1, 1, (1 << 64) - 1)

    # Lifetime: backing array/Matrix/MMapBuffer kept alive via py::keep_alive
    def build_numpy_subview() -> ml.TensorView:
        local_arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        return ml.tensor_view_from_numpy(local_arr).subview(1, 1, 0, 2)

    subview = build_numpy_subview()
    gc.collect()
    assert subview[0, 0] == 30.0
    assert subview[0, 1] == 40.0

    def build_matrix_slice() -> ml.TensorView:
        local_mat = ml.Matrix([[5.0, 6.0], [7.0, 8.0]])
        return local_mat.view().slice(1, 1)

    mat_slice = build_matrix_slice()
    gc.collect()
    assert mat_slice[0, 1] == 8.0

    mldat_path = tmp_path / "life.mldat"
    ml.MMapBuffer.write_mldat_numpy(
        str(mldat_path), np.array([[9.0, 11.0], [13.0, 15.0]], dtype=np.float64)
    )

    def build_mmap_view() -> ml.TensorView:
        local_buf = ml.MMapBuffer(str(mldat_path))
        return local_buf.full_view()

    mmap_view = build_mmap_view()
    gc.collect()
    assert mmap_view[1, 1] == 15.0


def test_matrix_overflow_and_empty_operations() -> None:
    with pytest.raises(OverflowError):
        ml.Matrix(1 << 62, 4)

    empty = ml.Matrix(0, 0)
    assert empty.to_list() == []
    assert (empty * ml.Matrix(0, 3)).shape == (0, 3)
    np_empty = empty.view().to_numpy()
    assert np_empty.shape == (0, 0)


def test_models_input_validation() -> None:
    with pytest.raises(ValueError):
        ml.KNN(0)
    with pytest.raises(ValueError):
        ml.KNN(-2)

    knn = ml.KNN(1)
    with pytest.raises(ValueError):
        knn.fit(np.empty((0, 2), dtype=np.float64), [])
    knn.fit(ml.Matrix([[1.0, 2.0], [3.0, 4.0]]), [0, 1])
    with pytest.raises(ValueError):
        knn.predict(ml.Matrix([[1.0, 2.0, 3.0]]))

    with pytest.raises(ValueError):
        ml.LinearRegression(float("nan"))
    with pytest.raises(ValueError):
        ml.LinearRegression(float("inf"))
    with pytest.raises(ValueError):
        ml.LogisticRegression(float("nan"))

    logreg = ml.LogisticRegression(learning_rate=0.1, epochs=10)
    logreg.fit(ml.Matrix([[0.0], [1.0]]), [0, 1])
    with pytest.raises(ValueError):
        logreg.compute_loss(ml.Matrix([[0.0], [1.0]]), [0])


def test_calculus_validation() -> None:
    with pytest.raises(RuntimeError):
        jacobian(
            lambda x: [x[0], x[1]] if abs(x[0]) < 1e-12 else [x[0]],
            [0.0, 1.0],
        )

    with pytest.raises(ValueError):
        gradient(lambda x: x[0], [1.0], h=0.0)
    with pytest.raises(ValueError):
        gradient(lambda x: x[0], [1.0], h=float("nan"))
    with pytest.raises(ValueError):
        jacobian(lambda x: [x[0]], [1.0], h=-1e-5)
    with pytest.raises(ValueError):
        hessian(lambda x: x[0] ** 2, [1.0], h=float("inf"))
    with pytest.raises(ValueError):
        derivative(lambda x: x**2, 1.0, 1, h=float("nan"))

    with pytest.raises(OverflowError):
        integrate(lambda x: x, 0.0, 1.0, method="simpson", n=(1 << 64) - 1)
    with pytest.raises(ValueError):
        integrate(lambda x: x, 0.0, float("nan"), method="simpson", n=4)


class _FailingDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> np.ndarray:
        if index == 2:
            raise RuntimeError("boom in worker")
        return np.array([float(index)], dtype=np.float64)


def test_python_formatter_and_dataloader_safety() -> None:
    mat_tex = matrix_to_latex(
        np.array([[float("nan"), float("inf")], [-float("inf"), 2.0]])
    )
    assert r"\text{NaN}" in mat_tex
    assert r"\infty" in mat_tex
    assert r"-\infty" in mat_tex

    vec_tex = vector_to_latex(np.array([float("nan"), float("-inf"), 1.0]))
    assert r"\text{NaN}" in vec_tex
    assert r"-\infty" in vec_tex

    loader = DataLoader(
        _FailingDataset(), batch_size=2, shuffle=False, prefetch_batches=1
    )
    with pytest.raises(RuntimeError, match="boom in worker"):
        list(loader)

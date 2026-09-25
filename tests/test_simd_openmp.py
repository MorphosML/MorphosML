import numpy as np
import pytest

import morphosml as ml


def test_simd_introspection():
    """Verify hardware acceleration introspection flags and thread controls."""
    caps = ml.get_simd_capabilities()
    assert isinstance(caps, str)
    assert len(caps) > 0

    orig_threads = ml.get_num_threads()
    assert orig_threads >= 1

    ml.set_num_threads(2)
    assert ml.get_num_threads() == 2

    # Restore original thread count
    ml.set_num_threads(orig_threads)
    assert ml.get_num_threads() == orig_threads


def test_vector_simd_accuracy_large():
    """Test SIMD vector operations (AVX2 unrolled 16 doubles + remainder) against NumPy."""
    n = 1047  # non-multiple of 16 to stress unrolled loops and remainders
    rng = np.random.RandomState(42)
    a_np = rng.randn(n)
    b_np = rng.randn(n)

    a_vec = ml.Vector(a_np.tolist())
    b_vec = ml.Vector(b_np.tolist())

    # Dot product
    expected_dot = float(np.dot(a_np, b_np))
    actual_dot = a_vec.dot(b_vec)
    assert pytest.approx(expected_dot, rel=1e-10) == actual_dot

    # Vector norm
    expected_norm = float(np.linalg.norm(a_np))
    actual_norm = a_vec.norm()
    assert pytest.approx(expected_norm, rel=1e-10) == actual_norm

    # Vector add
    sum_vec = a_vec + b_vec
    assert np.allclose(sum_vec.to_list(), a_np + b_np)

    # Vector sub
    diff_vec = a_vec - b_vec
    assert np.allclose(diff_vec.to_list(), a_np - b_np)

    # Vector scale
    scale_vec = a_vec * 2.5
    assert np.allclose(scale_vec.to_list(), a_np * 2.5)


def test_matrix_parallel_gemm_large():
    """Test OpenMP + AVX2 parallel matrix multiplication against NumPy."""
    m, k, n = 120, 150, 130
    rng = np.random.RandomState(42)
    a_np = rng.randn(m, k)
    b_np = rng.randn(k, n)

    a_mat = ml.Matrix(a_np.tolist())
    b_mat = ml.Matrix(b_np.tolist())

    c_mat = a_mat * b_mat
    c_np = np.dot(a_np, b_np)

    assert np.allclose(c_mat.to_std(), c_np, atol=1e-10)


def test_matrix_vector_simd():
    """Test Matrix-Vector multiplication using SIMD dot products."""
    m, n = 100, 80
    rng = np.random.RandomState(42)
    a_np = rng.randn(m, n)
    v_np = rng.randn(n)

    a_mat = ml.Matrix(a_np.tolist())
    v_vec = ml.Vector(v_np.tolist())

    res_vec = a_mat * v_vec
    expected = np.dot(a_np, v_np)

    assert np.allclose(res_vec.to_list(), expected, atol=1e-10)


def test_knn_parallel_simd():
    """Test KNN multi-threaded SIMD prediction on synthetic clustered dataset."""
    rng = np.random.RandomState(42)
    # 200 training samples with 10 features across 2 clusters
    X_train_c0 = rng.randn(100, 10) - 2.0
    X_train_c1 = rng.randn(100, 10) + 2.0
    X_train = np.vstack([X_train_c0, X_train_c1])
    y_train = [0] * 100 + [1] * 100

    knn = ml.KNN(k=3)
    knn.fit(X_train, y_train)

    # 40 test queries
    X_test_c0 = rng.randn(20, 10) - 2.0
    X_test_c1 = rng.randn(20, 10) + 2.0
    X_test = np.vstack([X_test_c0, X_test_c1])
    y_expected = [0] * 20 + [1] * 20

    preds = knn.predict(X_test)
    assert len(preds) == 40
    acc = sum(p == y for p, y in zip(preds, y_expected)) / 40.0
    assert acc >= 0.95

import numpy as np

from morphosml import Matrix


def test_matrix_creation():
    m = Matrix([[1, 2], [3, 4]])
    assert m.rows() == 2
    assert m.cols() == 2
    assert m(0, 0) == 1
    assert m(0, 1) == 2


def test_matrix_indexing():
    m = Matrix(2, 2)
    m[0, 0] = 10.0
    m[0, 1] = 20.0
    m[1, 0] = 30.0
    m[1, 1] = 40.0
    assert m[0, 0] == 10.0
    assert m[0, 1] == 20.0
    assert m[1, 0] == 30.0
    assert m[1, 1] == 40.0


def test_matrix_buffer_protocol():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    arr = np.asarray(m)
    assert arr.shape == (2, 2)
    assert arr.dtype == np.float64
    assert arr[0, 1] == 2.0


def test_matrix_multiplication():
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    m3 = m1 * m2
    assert m3(0, 0) == 19
    assert m3(0, 1) == 22
    assert m3(1, 0) == 43
    assert m3(1, 1) == 50


def test_matrix_addition_subtraction():
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[10, 20], [30, 40]])
    add = m1 + m2
    sub = m2 - m1
    assert add(0, 0) == 11
    assert add(1, 1) == 44
    assert sub(0, 0) == 9
    assert sub(1, 1) == 36


def test_matrix_transpose():
    m = Matrix([[1, 2], [3, 4]])
    mt = m.transpose()
    assert mt(0, 0) == 1
    assert mt(0, 1) == 3
    assert mt(1, 0) == 2
    assert mt(1, 1) == 4


def test_matrix_random_reproducibility():
    m1 = Matrix.random(4, 4, seed=42)
    m2 = Matrix.random(4, 4, seed=42)
    assert m1.to_list() == m2.to_list()

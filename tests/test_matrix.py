import pytest
from morphosml import Matrix


def test_matrix_creation():
    m = Matrix([[1, 2], [3, 4]])
    assert m.rows() == 2
    assert m.cols() == 2
    assert m(0, 0) == 1
    assert m(0, 1) == 2


def test_matrix_multiplication():
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    m3 = m1 * m2
    assert m3(0, 0) == 19
    assert m3(0, 1) == 22
    assert m3(1, 0) == 43
    assert m3(1, 1) == 50


def test_matrix_transpose():
    m = Matrix([[1, 2], [3, 4]])
    mt = m.transpose()
    assert mt(0, 0) == 1
    assert mt(0, 1) == 3
    assert mt(1, 0) == 2
    assert mt(1, 1) == 4
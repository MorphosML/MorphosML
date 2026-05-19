import pytest
from morphosml import Vector

def test_vector_creation():
    v = Vector([1, 2, 3])
    assert v.size() == 3
    assert v[0] == 1
    assert v[1] == 2
    assert v[2] == 3

def test_vector_addition():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    v3 = v1 + v2
    assert v3.to_list() == [5, 7, 9]

def test_vector_dot():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    assert v1.dot(v2) == 32

def test_vector_norm():
    v = Vector([3, 4])
    assert v.norm() == 5.0
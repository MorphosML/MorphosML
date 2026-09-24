import numpy as np
import pytest

from morphosml import get_seed, set_seed, train_test_split


def test_train_test_split_proportions():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    assert len(X_train) == 40
    assert len(X_test) == 10
    assert len(y_train) == 40
    assert len(y_test) == 10


def test_train_test_split_reproducibility():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    X_train1, X_test1, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(X_test1, X_test2)


def test_train_test_split_invalid_test_size():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=-0.1)


def test_global_seed_system():
    set_seed(9999)
    assert get_seed() == 9999

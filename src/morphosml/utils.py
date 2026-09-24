"""Dataset and numerical utilities for MorphosML."""

from __future__ import annotations

from typing import Any

import numpy as np


def train_test_split(
    X: Any,
    y: Any,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix or array.
    y : array-like of shape (n_samples,)
        Target label or continuous value array.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (strictly between 0.0 and 1.0).
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.
    random_state : int or None, default=None
        Deterministic seed for reproducible pseudorandom permutation.

    Returns
    -------
    X_train : numpy.ndarray
        Training feature split.
    X_test : numpy.ndarray
        Testing feature split.
    y_train : numpy.ndarray
        Training target split.
    y_test : numpy.ndarray
        Testing target split.

    Raises
    ------
    ValueError
        If `test_size` is not in (0.0, 1.0), or if `X` and `y` have mismatched lengths,
        or if the split results in an empty partition.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be strictly between 0.0 and 1.0")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"len(X)={len(X_arr)}, len(y)={len(y_arr)}"
        )

    n_samples = len(X_arr)
    n_test = int(np.round(n_samples * test_size))
    n_train = n_samples - n_test

    if n_test == 0 or n_train == 0:
        raise ValueError(
            f"With n_samples={n_samples} and test_size={test_size}, "
            f"either train ({n_train}) or test ({n_test}) split is empty."
        )

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    return (
        X_arr[train_indices],
        X_arr[test_indices],
        y_arr[train_indices],
        y_arr[test_indices],
    )

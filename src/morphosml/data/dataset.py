"""High-performance dataset abstractions for in-memory and out-of-core data workflows.

Provides:
- `Dataset`: Abstract base dataset interface.
- `NumpyDataset`: In-memory zero-copy wrapper for NumPy arrays using C++ `TensorView`.
- `MMapDataset`: Out-of-core memory-mapped dataset reader for `.mldat` binary files.
- `from_numpy`: Convenience factory function.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from morphosml._core import MMapBuffer, TensorView, tensor_view_from_numpy


class Dataset(ABC):
    """Abstract Base Dataset interface for MorphosML.

    Subclasses must implement `__len__` and `__getitem__` to support batching
    and indexing in `DataLoader`.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int | slice):
        """Retrieve a sample or a slice of samples."""
        pass


class NumpyDataset(Dataset):
    """In-memory zero-copy dataset wrapping NumPy array buffers via `TensorView`.

    Parameters
    ----------
    X : numpy.ndarray
        2D feature array of shape `(n_samples, n_features)`. Must be convertible
        to C-contiguous float64.
    y : numpy.ndarray, optional
        Target labels or regression values of length `n_samples`.

    Raises
    ------
    TypeError
        If `X` or `y` is not an instance of `numpy.ndarray`.
    ValueError
        If `X` is not 2-dimensional or if lengths of `X` and `y` do not match.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim}D")

        # Ensure C-contiguous double precision for zero-copy view
        if not X.flags["C_CONTIGUOUS"] or X.dtype != np.float64:
            X = np.ascontiguousarray(X, dtype=np.float64)

        self._X = X
        self._X_view = tensor_view_from_numpy(self._X)

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy.ndarray")
            if len(y) != len(X):
                raise ValueError(
                    f"Length mismatch: X has {len(X)} samples, y has {len(y)}"
                )
            self._y = y
        else:
            self._y = None

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._X)

    @property
    def shape(self) -> tuple[int, int]:
        """Return dataset shape `(n_samples, n_features)`."""
        return self._X.shape

    @property
    def view(self) -> TensorView:
        """Return non-owning C++ TensorView wrapping the in-memory array."""
        return self._X_view

    @property
    def checksum(self) -> int:
        """Return checksum (0 for in-memory datasets)."""
        return 0

    def __getitem__(self, idx: int | slice):
        """Index into feature and label buffers."""
        if self._y is not None:
            return self._X[idx], self._y[idx]
        return self._X[idx]


class MMapDataset(Dataset):
    """High-performance out-of-core dataset backed by POSIX `mmap` and `.mldat` binary files.

    Enables instant (0 ms parse time) memory mapping and streaming of multi-gigabyte
    datasets directly from disk at hardware bus bandwidth.

    Parameters
    ----------
    filepath : str
        Filesystem path to the `.mldat` binary file.

    Raises
    ------
    FileNotFoundError
        If `filepath` does not exist on disk.
    RuntimeError
        If the file header signature is corrupt or invalid.
    """

    def __init__(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self._filepath = str(filepath)
        self._buffer = MMapBuffer(self._filepath)

    def __len__(self) -> int:
        """Return total number of rows in the binary dataset."""
        return self._buffer.rows()

    @property
    def rows(self) -> int:
        """Return row count."""
        return self._buffer.rows()

    @property
    def cols(self) -> int:
        """Return feature dimension count."""
        return self._buffer.cols()

    @property
    def shape(self) -> tuple[int, int]:
        """Return dataset dimensions `(rows, cols)`."""
        return (self._buffer.rows(), self._buffer.cols())

    @property
    def checksum(self) -> int:
        """Return 64-bit integrity checksum stored in the file header."""
        return self._buffer.checksum()

    @property
    def filepath(self) -> str:
        """Return filesystem path of the mapped file."""
        return self._filepath

    def get_slice(self, start_row: int, num_rows: int) -> TensorView:
        """Return a non-owning zero-copy `TensorView` slice for a row range.

        Parameters
        ----------
        start_row : int
            0-based starting row.
        num_rows : int
            Number of rows to include in the slice.

        Returns
        -------
        TensorView
            Zero-copy 2D view pointing directly to the mapped kernel page address.
        """
        return self._buffer.get_slice(start_row, num_rows)

    def full_view(self) -> TensorView:
        """Return a non-owning zero-copy `TensorView` over the entire mapped dataset."""
        return self._buffer.full_view()

    def to_numpy(self) -> np.ndarray:
        """Expose the entire memory-mapped file as a 2D NumPy array without copying."""
        return np.asarray(self._buffer.full_view())

    def __getitem__(self, idx: int | slice):
        """Index or slice rows in the dataset."""
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                return self.to_numpy()[idx]
            num_rows = max(0, stop - start)
            return self.get_slice(start, num_rows)
        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(
                    f"Index {idx} out of bounds for dataset of length {len(self)}"
                )
            return self.get_slice(idx, 1)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    @staticmethod
    def dump(filepath: str, data: np.ndarray) -> None:
        """Export a 2D NumPy array to high-speed contiguous `.mldat` binary format.

        Parameters
        ----------
        filepath : str
            Target destination path on disk.
        data : numpy.ndarray
            2D array of float64 features to serialize.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        if not data.flags["C_CONTIGUOUS"] or data.dtype != np.float64:
            data = np.ascontiguousarray(data, dtype=np.float64)
        MMapBuffer.write_mldat_numpy(filepath, data)


def from_numpy(X: np.ndarray, y: np.ndarray | None = None) -> NumpyDataset:
    """Convenience helper to construct an in-memory `NumpyDataset`.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray, optional
        Target vector.

    Returns
    -------
    NumpyDataset
        Wrapped dataset.
    """
    return NumpyDataset(X, y)

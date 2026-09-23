from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from morphosml._core import MMapBuffer, TensorView, tensor_view_from_numpy


class Dataset(ABC):
    """Abstract Base Dataset interface for MorphosML."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int | slice):
        pass


class NumpyDataset(Dataset):
    """In-memory zero-copy dataset wrapping NumPy array buffers via TensorView."""

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
        return len(self._X)

    @property
    def shape(self) -> tuple[int, int]:
        return self._X.shape

    @property
    def view(self) -> TensorView:
        return self._X_view

    @property
    def checksum(self) -> int:
        return 0

    def __getitem__(self, idx: int | slice):
        if self._y is not None:
            return self._X[idx], self._y[idx]
        return self._X[idx]


class MMapDataset(Dataset):
    """High-performance out-of-core dataset backed by POSIX mmap and .mldat binary format.

    Enables instant (0ms parse time) random access and sequential streaming
    of large datasets directly from NVMe/SSD at line rate.
    """

    def __init__(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self._filepath = str(filepath)
        self._buffer = MMapBuffer(self._filepath)

    def __len__(self) -> int:
        return self._buffer.rows()

    @property
    def rows(self) -> int:
        return self._buffer.rows()

    @property
    def cols(self) -> int:
        return self._buffer.cols()

    @property
    def shape(self) -> tuple[int, int]:
        return (self._buffer.rows(), self._buffer.cols())

    @property
    def checksum(self) -> int:
        return self._buffer.checksum()

    @property
    def filepath(self) -> str:
        return self._filepath

    def get_slice(self, start_row: int, num_rows: int) -> TensorView:
        """Returns a non-owning zero-copy TensorView slice for the given row range."""
        return self._buffer.get_slice(start_row, num_rows)

    def full_view(self) -> TensorView:
        """Returns a non-owning zero-copy TensorView over the entire mapped file."""
        return self._buffer.full_view()

    def to_numpy(self) -> np.ndarray:
        """Zero-copy view as a NumPy 2D array wrapping the mmap buffer."""
        return np.asarray(self._buffer.full_view())

    def __getitem__(self, idx: int | slice):
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
        """Export a NumPy array to high-speed contiguous .mldat binary format."""
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        if not data.flags["C_CONTIGUOUS"] or data.dtype != np.float64:
            data = np.ascontiguousarray(data, dtype=np.float64)
        MMapBuffer.write_mldat_numpy(filepath, data)


def from_numpy(X: np.ndarray, y: np.ndarray | None = None) -> NumpyDataset:
    """Convenience helper to create a zero-copy dataset from NumPy arrays."""
    return NumpyDataset(X, y)

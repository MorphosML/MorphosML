from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np

from morphosml._core import IdempotentSampler, IngestionCursor
from morphosml.data.dataset import Dataset, MMapDataset, NumpyDataset


class DataLoader:
    """High-performance, fault-tolerant, idempotent DataLoader.

    Features:
    - Deterministic seeded shuffling across epochs via IdempotentSampler (SplitMix64).
    - Sub-millisecond cursor-based checkpointing (get_cursor, resume_from).
    - Double-buffered background prefetching for zero-bubble pipeline execution.
    - Zero-copy mmap batch slicing when sequential.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        epoch: int = 0,
        prefetch_batches: int = 2,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if prefetch_batches < 0:
            raise ValueError(
                f"prefetch_batches must be non-negative, got {prefetch_batches}"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = epoch
        self.prefetch_batches = prefetch_batches

        self._sampler = IdempotentSampler(seed) if shuffle else None
        self._sample_offset = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current training epoch for deterministic shuffling."""
        self.epoch = epoch
        self._sample_offset = 0

    def get_cursor(self) -> IngestionCursor:
        """Capture current ingestion state as a lightweight serializable checkpoint token."""
        cursor = IngestionCursor()
        cursor.epoch = self.epoch
        cursor.sample_offset = self._sample_offset
        cursor.checksum = getattr(self.dataset, "checksum", 0)
        return cursor

    def resume_from(self, cursor: IngestionCursor | str) -> None:
        """Resume DataLoader state from a previously saved IngestionCursor checkpoint token."""
        if isinstance(cursor, str):
            cursor = IngestionCursor.from_string(cursor)

        ds_checksum = getattr(self.dataset, "checksum", 0)
        if ds_checksum != 0 and cursor.checksum != 0 and ds_checksum != cursor.checksum:
            raise ValueError(
                f"Dataset checksum mismatch: dataset has {ds_checksum}, cursor has {cursor.checksum}. "
                "Dataset contents or file have changed."
            )

        self.epoch = cursor.epoch
        self._sample_offset = cursor.sample_offset

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _get_indices(self) -> list[int]:
        n = len(self.dataset)
        if self.shuffle and self._sampler is not None:
            return self._sampler.generate_indices(n, self.epoch)
        return list(range(n))

    def _load_batch(self, batch_indices: list[int]) -> Any:
        # Check if consecutive for zero-copy fast path on MMapDataset
        if isinstance(self.dataset, MMapDataset):
            is_consecutive = (
                len(batch_indices) > 0
                and batch_indices[-1] - batch_indices[0] == len(batch_indices) - 1
                and batch_indices
                == list(range(batch_indices[0], batch_indices[0] + len(batch_indices)))
            )
            if is_consecutive:
                return self.dataset.get_slice(batch_indices[0], len(batch_indices))

            # Non-consecutive indexed access
            full_np = self.dataset.to_numpy()
            return full_np[batch_indices]

        if isinstance(self.dataset, NumpyDataset):
            X_arr = self.dataset._X[batch_indices]
            if self.dataset._y is not None:
                y_arr = self.dataset._y[batch_indices]
                return X_arr, y_arr
            return X_arr

        # Generic dataset fallback
        items = [self.dataset[i] for i in batch_indices]
        if isinstance(items[0], tuple):
            return tuple(
                np.array([item[k] for item in items]) for k in range(len(items[0]))
            )
        return np.array(items)

    def __iter__(self) -> Iterator[Any]:
        all_indices = self._get_indices()
        total_samples = len(all_indices)

        # Slice starting from resume sample offset
        start_offset = self._sample_offset
        self._sample_offset = 0  # reset for future full passes

        batch_slices: list[list[int]] = []
        for i in range(start_offset, total_samples, self.batch_size):
            batch_idx = all_indices[i : i + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue
            batch_slices.append(batch_idx)

        if not batch_slices:
            return

        # Double-buffering async prefetch queue
        if self.prefetch_batches > 0:
            batch_queue: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
            stop_event = threading.Event()

            def producer():
                try:
                    for b_indices in batch_slices:
                        if stop_event.is_set():
                            break
                        batch = self._load_batch(b_indices)
                        batch_queue.put((b_indices, batch))
                finally:
                    batch_queue.put(None)  # Sentinel to mark completion

            worker = threading.Thread(target=producer, daemon=True)
            worker.start()

            try:
                while True:
                    item = batch_queue.get()
                    if item is None:
                        break
                    b_indices, batch = item
                    self._sample_offset += len(b_indices)
                    yield batch
            finally:
                stop_event.set()
                # Drain remaining items if interrupted early to release thread
                while not batch_queue.empty():
                    try:
                        batch_queue.get_nowait()
                    except queue.Empty:
                        break
        else:
            for b_indices in batch_slices:
                batch = self._load_batch(b_indices)
                self._sample_offset += len(b_indices)
                yield batch

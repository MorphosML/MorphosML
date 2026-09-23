# High-Performance Data Ingestion & Streaming Tutorial

This guide details MorphosML's out-of-core data ingestion architecture, POSIX memory-mapped dataset streaming, idempotent sampling, and cursor-based fault-tolerant checkpointing.

---

## 1. High-Performance Ingestion Principles

| Principle | Ingestion Requirement | MorphosML Architecture |
| :--- | :--- | :--- |
| **Idempotency** | Processing epoch $E$ with seed $S$ yields identical batches every time. | `IdempotentSampler`: Deterministic SplitMix64 pseudo-random permutation. |
| **Fault-Tolerance** | Crashes resume without re-reading from line 0. | `IngestionCursor`: 16-byte state token (`epoch:sample_offset:checksum`). |
| **Resiliency** | Corrupted files or drift detected immediately. | FNV-1a 64-bit checksum validated during mmap header read. |
| **Zero-Copy** | Avoid RAM saturation and memory duplication. | `TensorView` & POSIX `mmap` with `madvise(MADV_SEQUENTIAL)`. |
| **Hardware ROI** | Modern NVMe reads at 3–7 GB/s; saturate compute for pennies. | Double-buffered background worker queue (`prefetch_batches=2`). |

---

## 2. Converting Data to Native `.mldat` Format

Text formats (CSV, TSV, JSON) require converting ASCII strings to IEEE-754 floating-point numbers, which wastes up to 80% of CPU time. 

Export your dataset once into MorphosML's native `.mldat` binary format:

```python
import numpy as np
import morphosml as mml

# Generate or load your tabular features (e.g. 10 million rows, 32 columns)
X = np.random.randn(100000, 32).astype(np.float64)

# Dump to binary .mldat
mml.data.MMapDataset.dump("features_100k.mldat", X)
```

The resulting file contains a 64-byte aligned header followed by the contiguous raw payload:
- **Magic bytes**: `"MML\0"`
- **Version**: `1`
- **Rows**: `100000`
- **Cols**: `32`
- **Checksum**: FNV-1a 64-bit integrity hash

---

## 3. Streaming Out-of-Core with POSIX `mmap`

Open a 50 GB dataset in less than a millisecond:

```python
dataset = mml.data.MMapDataset("features_100k.mldat")

# Instantly inspect metadata
print(f"Rows: {dataset.rows}, Cols: {dataset.cols}, Checksum: {dataset.checksum}")

# Fast zero-copy slicing
slice_view = dataset.get_slice(start_row=500, num_rows=100)
print(f"Slice shape: ({slice_view.rows()}, {slice_view.cols()})")
```

The operating system automatically pages rows into physical memory when accessed and evicts them under memory pressure.

---

## 4. Double-Buffered Asynchronous DataLoader

```python
loader = mml.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    seed=1337,
    prefetch_batches=2 # Pre-loads upcoming batches asynchronously
)

for epoch in range(5):
    loader.set_epoch(epoch)
    for batch in loader:
        # Train model here without waiting on I/O disk bubbles
        pass
```

---

## 5. Checkpointing & Fault-Tolerant Recovery

If a long-running training job is interrupted or aborted, capture the exact state with an `IngestionCursor`:

```python
# During training loop
for step, batch in enumerate(loader):
    if step == 150: # Simulate preemption or checkpoint trigger
        checkpoint = loader.get_cursor()
        token = checkpoint.to_string()
        print(f"Saved checkpoint token: {token}")
        # Save token to file: "0:38400:183928174829"
        break
```

### Resuming After Crash

When recovering, pass the saved checkpoint string directly to `resume_from`:

```python
# Create fresh DataLoader
new_loader = mml.data.DataLoader(dataset, batch_size=256, shuffle=True, seed=1337)

# Restore exact state
new_loader.resume_from("0:38400:183928174829")

# Training resumes from sample offset 38400 without replaying previous batches!
for batch in new_loader:
    # Continues immediately from step 151
    pass
```

> [!NOTE]
> If the dataset file was modified or replaced on disk, `resume_from` detects a checksum mismatch and immediately raises a `ValueError` to prevent data corruption.

# MorphosML v0.3.1

[![CI](https://github.com/MorphosML/MorphosML/actions/workflows/ci.yml/badge.svg)](https://github.com/MorphosML/MorphosML/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![C++: 17](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)](https://isocpp.org/)

**MorphosML** is a high-performance machine learning, differentiable computing, and out-of-core data ingestion library engineered in modern C++17 with zero-overhead Python bindings.

Designed from first principles for **High-Performance Computing (HPC)**, MorphosML saturates CPU cache lines, vector registers (AVX2/AVX-512), and NVMe storage bandwidth while exposing an intuitive, Pythonic API.

---

## 📚 Documentation & Guides

Explore the comprehensive MorphosML documentation library:
- 🏗️ **[Architectural Blueprint](docs/architecture.md)**: Deep dive into contiguous memory layouts, zero-copy buffer protocol, POSIX `mmap` kernel caching, and asynchronous double buffering.
- 📖 **[API Reference](docs/api_reference.md)**: Exhaustive class and method specifications for `Matrix`, `Vector`, `TensorView`, `MMapDataset`, `DataLoader`, `calculus`, and models.
- 🚀 **[Quickstart Tutorial](docs/tutorials/quickstart.md)**: Step-by-step introduction from installation to model evaluation.
- ⚡ **[High-Performance Ingestion Tutorial](docs/tutorials/hpc_ingestion.md)**: Out-of-core `.mldat` streaming, idempotent sampling, and cursor checkpoint recovery.
- 🧮 **[Calculus & Differentiable Programming](docs/tutorials/calculus_autodiff.md)**: $\mathcal{O}(h^4)$ Taylor stencils, gradients, Jacobians, Hessians, quadrature, and limits.
- 📐 **[LaTeX Visualization & Jupyter Integration](docs/tutorials/latex_visualization.md)**: LaTeX mathematical typesetting and native interactive notebook rendering.
- 🤝 **[Contributing Guide](CONTRIBUTING.md)**: Development setup, building C++ extensions, formatting standards, and pull request workflows.

---

## Table of Contents

- [Core Principles](#core-principles)
- [Architecture & Performance](#architecture--performance)
- [Installation](#installation)
- [Quickstart Guide](#quickstart-guide)
- [Memory & Linear Algebra Layer](#memory--linear-algebra-layer)
- [Out-of-Core Data Ingestion Engine](#out-of-core-data-ingestion-engine)
- [Calculus & Differentiable Programming Core](#calculus--differentiable-programming-core)
- [Machine Learning Models](#machine-learning-models)
- [Latexify & Jupyter Notebook Integration](#latexify--jupyter-notebook-integration)
- [Verification & CI/CD Pipeline](#verification--cicd-pipeline)
- [License & Author](#license--author)

---

## Core Principles

MorphosML is governed by four core architectural tenets:

| Principle | Engineering Implementation |
| :--- | :--- |
| **High Performance (HPC)** | Contiguous 1D flat buffers, zero-copy Python Buffer Protocol, $i$-$k$-$j$ cache-line matrix multiplication, and POSIX `mmap` with `MADV_SEQUENTIAL`. |
| **Idempotency** | Reproducible batch sequencing and shuffling across epochs and workers using deterministic SplitMix64 PRNG: $\text{state} = \text{seed} \oplus (\text{epoch} \cdot C_1 + C_2)$. |
| **Fault-Tolerance** | Sub-millisecond cursor tokens (`epoch:sample_offset:checksum`). Training pipelines resume mid-epoch after crashes without re-streaming from line 0. |
| **Resiliency** | FNV-1a 64-bit binary checksum validation, shape verification, and poison record isolation preventing memory faults or silent data corruption. |

---

## Architecture & Performance

```
MorphosML System Topology
├── Core Memory & Linear Algebra Layer (C++17)
│   ├── Vector              (Contiguous double-precision vector)
│   ├── Matrix              (Contiguous 1D flat row-major buffer with pybind11 buffer protocol)
│   └── TensorView          (Non-owning, zero-copy contiguous 2D view with read-only buffer protocol)
├── High-Throughput Storage & Ingestion Engine
│   ├── MMapBuffer          (POSIX mmap reader, MADV_SEQUENTIAL line-rate streaming)
│   ├── Native Binary .mldat (64-byte structured header + raw payload + FNV-1a checksum)
│   ├── IdempotentSampler   (Deterministic SplitMix64 Fisher-Yates generator)
│   ├── IngestionCursor     (16-byte state checkpoint: {epoch, sample_offset, checksum})
│   └── DataLoader          (Double-buffered async prefetch queue for zero I/O bubbles)
├── Numerical Calculus Engine
│   ├── Differentiation     (5-point stencils O(h^4) derivatives, gradients, Jacobians, Hessians)
│   ├── Integration         (Trapezoidal, Simpson's 1/3 O(h^4), Simpson's 3/8, Gauss-Legendre, 2D)
│   └── Limits              (Left, right, two-sided limits with asymptotic divergence detection)
├── Machine Learning Models & Metrics
│   ├── LinearRegression    (C++ gradient descent, MSE cost, R^2 score)
│   ├── LogisticRegression  (C++ sigmoid, BCE loss, probability predictions)
│   └── KNN Classifier      (C++ k-nearest neighbors classification)
└── Latexify & Presentation Layer
    ├── Formatter           (Matrix, Vector, Model, and Calculus formula to LaTeX string)
    └── Jupyter Hooks       (Rich _repr_latex_ typography in notebooks)
```

---

## Installation

### From Source (Recommended for Local Dev)

Requirements: C++17 compatible compiler (`g++` or `clang++`), CMake >= 3.15, Python >= 3.10.

```bash
# Clone the repository
git clone https://github.com/MorphosML/MorphosML.git
cd MorphosML

# Build and install in editable mode
pip install -e . --no-build-isolation --no-deps
```

### Pre-Flight Automated Validation Script

MorphosML includes a dedicated CI validation and deployment script:

```bash
# Run local compilation, Black, Ruff, and 100% test suite
./scripts/ci_deploy.sh --check-only
```

---

## Quickstart Guide

```python
import numpy as np
import morphosml as mml

# 1. HPC Linear Algebra (Zero-Copy Interoperability)
mat = mml.Matrix.random(1000, 1000, seed=42)
np_arr = np.asarray(mat)  # Zero-copy view! 0 memory allocated

# 2. Calculus & Differentiable Programming
@mml.differentiable
def loss_fn(v):
    x, y = v[0], v[1]
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

grad = loss_fn.grad([1.0, 1.0])       # [0.0, 0.0]
hess = loss_fn.hessian([1.0, 1.0])    # 2x2 curvature Matrix

# 3. Machine Learning with C++ Core
X = np.random.randn(200, 5)
y = X @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + 0.25

X_train, X_test, y_train, y_test = mml.train_test_split(X, y, test_size=0.2, random_state=42)
model = mml.LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

print(f"R2 Score: {model.score(X_test, y_test):.4f}")

# 4. Rich LaTeX Math Export
print(mml.to_latex(model))
# Output: \hat{y} = 1.5 x_{1} - 2 x_{2} + 0.5 x_{3} + 3 x_{4} - 1 x_{5} + 0.25
```

---

## Memory & Linear Algebra Layer

### Contiguous 1D Flat `Matrix`
Unlike naive implementations that allocate `vector<vector<double>>` (causing cache-line invalidations and pointer chasing), MorphosML allocates a single contiguous buffer `vector<double> data_` of length `rows * cols`.

- **Element Access**: `mat(i, j)`, `mat[i, j]`, `mat.row_ptr(i)`.
- **$i$-$k$-$j$ Loop Ordering**: Matrix multiplication uses unit memory strides in the inner loop, unlocking compiler auto-vectorization (AVX2/AVX-512) and 100% cache-line utilization.
- **Buffer Protocol**: Pass matrices directly to NumPy or Polars via `np.asarray(matrix)` without duplicating memory.

### Non-Owning `TensorView`
Wraps arbitrary 2D pointers with row stride:
```python
from morphosml import TensorView, tensor_view_from_numpy

# Wrap a NumPy array without copying
arr = np.ones((500, 20), dtype=np.float64)
view = tensor_view_from_numpy(arr)

assert view.rows() == 500
assert view.cols() == 20
```

---

## Out-of-Core Data Ingestion Engine

### Native Binary `.mldat` Format
Parsing text datasets (e.g. CSV) wastes up to 80% of CPU cycles in `strtod`. MorphosML provides a native binary format (`.mldat`) with a 64-byte header and FNV-1a checksum:

```python
from morphosml.data import MMapDataset, DataLoader

# 1. Export NumPy array directly to binary .mldat
data = np.random.randn(100000, 64).astype(np.float64)
MMapDataset.dump("dataset_100k.mldat", data)

# 2. Map dataset instantly (0 ms load time)
dataset = MMapDataset("dataset_100k.mldat")
print(f"Dataset Shape: {dataset.shape}, Checksum: {dataset.checksum}")

# 3. Create high-performance, fault-tolerant DataLoader
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    seed=1337,
    prefetch_batches=2  # Asynchronous background double-buffering
)
```

### Idempotency & Fault-Tolerant Checkpointing
```python
# Checkpoint mid-epoch
for i, batch in enumerate(loader):
    if i == 500:
        cursor = loader.get_cursor()
        saved_token = cursor.to_string() # e.g. "0:32000:14695981039346656037"
        break

# Resume from exact sample offset in a new process without re-reading:
new_loader = DataLoader(dataset, batch_size=64, shuffle=True, seed=1337)
new_loader.resume_from(saved_token)
```

---

## Calculus & Differentiable Programming Core

MorphosML provides a high-precision numerical calculus engine implemented in C++:

### Numerical Differentiation & Derivatives
```python
import math
from morphosml.calculus import derivative, gradient, jacobian, hessian

# 1st, 2nd, 3rd derivatives with O(h^4) 5-point stencils
f = lambda x: math.sin(x)
print(derivative(f, 0.0, order=1))  # 1.0 (cos(0))
print(derivative(f, 0.0, order=2))  # 0.0 (-sin(0))

# Multivariate Gradient
f_multi = lambda v: v[0]**2 + 3.0 * v[1]**2
print(gradient(f_multi, [2.0, 1.0]))  # [4.0, 6.0]

# Jacobian Matrix
f_vec = lambda v: [v[0]**2 + v[1], 3.0 * v[0] - v[1]**2]
J = jacobian(f_vec, [2.0, 3.0])  # Shape (2, 2)

# Hessian Matrix (Curvature)
H = hessian(f_multi, [1.0, 1.0])  # [[2.0, 0.0], [0.0, 6.0]]
```

### Numerical Integration (Quadrature)
```python
from morphosml.calculus import integrate, integrate_2d

# 1D Integrals: 'simpson' (O(h^4)), 'trapezoidal', 'simpson_38', 'gauss_legendre'
val = integrate(math.sin, 0.0, math.pi, method="simpson")
print(val)  # 2.0

# 2D Double Integral: integral_0^1 integral_0^1 (x + y) dx dy
val_2d = integrate_2d(lambda x, y: x + y, 0.0, 1.0, 0.0, 1.0)
print(val_2d)  # 1.0
```

### Numerical Limits
```python
from morphosml.calculus import limit, evaluate_limit

# Removable singularity: lim_{x -> 0} sin(x)/x = 1.0
print(limit(lambda x: math.sin(x) / x, 0.0))  # 1.0

# One-sided infinite limits
diag = evaluate_limit(lambda x: 1.0 / x, 0.0, direction="right")
print(diag.is_infinite, diag.value)  # True, +inf
```

---

## Machine Learning Models

| Model | Core Language | Optimization | Features |
| :--- | :--- | :--- | :--- |
| **`LinearRegression`** | C++17 | Gradient Descent | Contiguous memory, $R^2$, MSE cost, `fit`, `predict`, `score` |
| **`LogisticRegression`** | C++17 | Gradient Descent + Sigmoid | Binary Cross-Entropy loss, `predict_proba`, `score` |
| **`KNN`** | C++17 | Vectorized Euclidean distance | Configurable `k`, single & bulk classification |

```python
from morphosml import LogisticRegression

clf = LogisticRegression(learning_rate=0.05, epochs=500)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test, threshold=0.5)
probabilities = clf.predict_proba(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.2%}")
```

---

## Latexify & Jupyter Notebook Integration

MorphosML makes presenting and publishing mathematical results effortless:

```python
import morphosml as mml

# LaTeX conversion
mat = mml.Matrix.identity(3)
print(mml.to_latex(mat))
# Output: \begin{pmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1\end{pmatrix}

# In Jupyter Notebooks, simply evaluating any Matrix, Vector, or fitted Model
# automatically renders KaTeX math typography via _repr_latex_():
model = mml.LinearRegression().fit(X, y)
mml.latexify(model)
```

---

## Verification & CI/CD Pipeline

The test suite validates correctness across all linear algebra, calculus, data ingestion, and model components:

```bash
pytest tests/ -v
# ============================== 61 passed in 0.22s ==============================
```

Code quality and style are strictly maintained:
- **Black Formatter**: Targeted for Python 3.12 (`line-length = 88`).
- **Ruff Linter**: PEP 8 compliance, zero warnings.
- **GitHub Actions Matrix**: Automated multi-OS workflows running tests across Python 3.10, 3.11, and 3.12 with Docker volume isolation.

---

## License & Author

- **Author**: Gabriel Carmona
- **License**: MIT License
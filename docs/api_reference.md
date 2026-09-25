# MorphosML API Reference

Complete API specification for MorphosML v0.4.0.

---

## Table of Contents
- [Linear Algebra Core](#linear-algebra-core)
  - [Matrix](#matrix)
  - [Vector](#vector)
  - [TensorView](#tensorview)
- [Data Ingestion & Streaming](#data-ingestion--streaming)
  - [Dataset & NumpyDataset](#dataset--numpydataset)
  - [MMapDataset](#mmapdataset)
  - [DataLoader](#dataloader)
  - [IngestionCursor](#ingestioncursor)
  - [IdempotentSampler](#idempotentsampler)
- [Calculus & Differentiable Programming](#calculus--differentiable-programming)
  - [derivative](#derivative)
  - [gradient](#gradient)
  - [jacobian](#jacobian)
  - [hessian](#hessian)
  - [integrate](#integrate)
  - [integrate_2d](#integrate_2d)
  - [limit](#limit)
  - [evaluate_limit](#evaluate_limit)
  - [differentiable](#differentiable)
  - [DifferentiableFunction](#differentiablefunction)
- [Machine Learning Models](#machine-learning-models)
  - [LinearRegression](#linearregression)
  - [LogisticRegression](#logisticregression)
  - [KNN](#knn)
- [Hardware Acceleration & Parallelism](#hardware-acceleration--parallelism)
  - [get_simd_capabilities](#get_simd_capabilities)
  - [get_num_threads](#get_num_threads)
  - [set_num_threads](#set_num_threads)
- [Evaluation Metrics & Utilities](#evaluation-metrics--utilities)
  - [train_test_split](#train_test_split)
  - [accuracy_score](#accuracy_score)
  - [mean_squared_error](#mean_squared_error)
  - [r2_score](#r2_score)
- [LaTeX Formatting & Visualization](#latex-formatting--visualization)
  - [to_latex](#to_latex)
  - [latexify](#latexify)
  - [enable_notebook_latex](#enable_notebook_latex)

---

## Linear Algebra Core

### `Matrix`
*HPC 2D dense matrix backed by a contiguous 1D flat row-major buffer with full NumPy Python Buffer Protocol interoperability.*

#### Constructors
- `Matrix(rows: int, cols: int, init_val: float = 0.0)`: Initializes with dimensions and optional initial scalar fill.
- `Matrix(data: list[list[float]])`: Initializes from nested Python lists.

#### Static Factories
- `Matrix.identity(n: int) -> Matrix`: Creates an $n \times n$ identity matrix.
- `Matrix.zeros(rows: int, cols: int) -> Matrix`: Creates a zero-filled matrix of shape `(rows, cols)`.
- `Matrix.random(rows: int, cols: int, seed: int = 42) -> Matrix`: Generates a uniform random matrix with values in $[0, 1)$.

#### Instance Methods
- `rows() -> int`: Returns number of rows.
- `cols() -> int`: Returns number of columns.
- `size() -> int`: Total elements (`rows * cols`).
- `empty() -> bool`: Returns `True` if empty.
- `data() -> list[float]`: Returns raw flat buffer as a Python list.
- `mat[i, j] -> float`: Accesses element at row `i`, column `j` (0-indexed).
- `mat[i, j] = val`: Sets element at row `i`, column `j`.
- `mat @ other` or `mat * other -> Matrix`: High-performance $i$-$k$-$j$ cache-line matrix multiplication.
- `mat + other -> Matrix`: Element-wise addition.
- `mat - other -> Matrix`: Element-wise subtraction.
- `mat * scalar -> Matrix`: Scalar multiplication.
- `transpose() -> Matrix`: Returns transposed matrix.
- `to_list() -> list[list[float]]`: Exports to nested Python lists.

---

### `Vector`
*Dense 1D vector with Euclidean vector space operations.*

#### Constructors
- `Vector(size: int, init_val: float = 0.0)`: Constructs vector of length `size`.
- `Vector(data: list[float])`: Constructs vector from Python list.

#### Instance Methods
- `size() -> int`: Number of elements.
- `norm() -> float`: Euclidean $L_2$ norm $\|\mathbf{v}\|_2 = \sqrt{\sum v_i^2}$.
- `normalized() -> Vector`: Returns unit-length vector ($\mathbf{v} / \|\mathbf{v}\|_2$).
- `dot(other: Vector) -> float`: Inner dot product $\mathbf{u} \cdot \mathbf{v}$.
- `vec[i] -> float`: 0-indexed element access.
- `vec[i] = val`: Element assignment.
- `vec + other -> Vector`: Element-wise addition.
- `vec - other -> Vector`: Element-wise subtraction.
- `vec * scalar -> Vector`: Scalar scaling.
- `to_list() -> list[float]`: Converts to Python list.

---

### `TensorView`
*Non-owning, zero-copy 2D view over contiguous or strided float64 memory.*

#### Constructors
- `TensorView(data: np.ndarray)`: Wraps NumPy 2D array without copying memory.
- `TensorView(mat: Matrix)`: Wraps an existing MorphosML `Matrix`.
- Factory: `tensor_view_from_numpy(arr: np.ndarray) -> TensorView`.

#### Instance Methods
- `rows() -> int`: Number of rows in view.
- `cols() -> int`: Number of columns in view.
- `stride() -> int`: Element distance between consecutive row starts.
- `size() -> int`: Total logical elements (`rows * cols`).
- `empty() -> bool`: Returns `True` if empty.
- `view[r, c] -> float`: Direct inlined element access.
- `to_matrix() -> Matrix`: Materializes a full copy into an owning `Matrix`.

---

## Data Ingestion & Streaming

### `Dataset & NumpyDataset`
- `morphosml.data.from_numpy(X: np.ndarray, y: np.ndarray | None = None) -> NumpyDataset`:
  Constructs an in-memory zero-copy dataset wrapping NumPy buffers via `TensorView`.

### `MMapDataset`
*Out-of-core memory-mapped dataset backed by POSIX `mmap` and `.mldat` binary files.*

#### Constructors & Factories
- `MMapDataset(filepath: str)`: Opens and memory-maps `.mldat` binary file instantly (0 ms parse time).
- `MMapDataset.dump(filepath: str, data: np.ndarray) -> None`: Exports NumPy array into binary `.mldat` with 64-byte header and FNV-1a checksum.

#### Properties & Methods
- `shape -> tuple[int, int]`: Dataset dimensions `(rows, cols)`.
- `checksum -> int`: FNV-1a 64-bit integrity hash.
- `get_slice(start_row: int, num_rows: int) -> TensorView`: Zero-copy view over row interval.
- `full_view() -> TensorView`: Non-owning view spanning entire dataset.
- `to_numpy() -> np.ndarray`: Exposes mapped buffer as 2D NumPy array without memory duplication.

---

### `DataLoader`
*High-performance, fault-tolerant, idempotent DataLoader with asynchronous prefetching.*

#### Parameters
- `dataset: Dataset`: Underlying dataset (`NumpyDataset` or `MMapDataset`).
- `batch_size: int = 32`: Batch size in number of samples.
- `shuffle: bool = True`: Enables deterministic seeded shuffling.
- `seed: int = 42`: Global PRNG seed for SplitMix64 generator.
- `drop_last: bool = False`: Drop trailing incomplete batch.
- `epoch: int = 0`: Initial training epoch.
- `prefetch_batches: int = 2`: Asynchronous worker prefetch queue depth.

#### Checkpoint Methods
- `set_epoch(epoch: int) -> None`: Sets active epoch for deterministic shuffling.
- `get_cursor() -> IngestionCursor`: Captures current batch progress into an ingestion cursor.
- `resume_from(cursor: IngestionCursor | str) -> None`: Restores exact checkpoint state.

---

### `IngestionCursor`
*16-byte checkpoint token capturing exact streaming progress.*
- Attributes: `epoch: int`, `sample_offset: int`, `checksum: int`.
- `to_string() -> str`: Serializes to format `"epoch:sample_offset:checksum"`.
- `from_string(token: str) -> IngestionCursor`: Restores cursor from token string.

---

## Calculus & Differentiable Programming

### `derivative`
```python
derivative(f: Callable[[float], float], x: float, order: int = 1, h: float = 0.0) -> float
```
- Computes 1st, 2nd, or 3rd order derivative using $\mathcal{O}(h^4)$ 5-point central-difference stencils.
- If `h == 0.0`, adaptive optimal perturbation step size is automatically calculated:
  - 1st order: $h = \epsilon^{1/3} \approx 6.05 \times 10^{-6}$
  - 2nd order: $h = \epsilon^{1/4} \approx 1.22 \times 10^{-4}$
  - 3rd order: $h = \epsilon^{1/5} \approx 1.33 \times 10^{-3}$

### `gradient`
```python
gradient(f: Callable[[Sequence[float]], float], x: Sequence[float], h: float = 1e-5) -> np.ndarray
```
- Computes multivariate gradient vector $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ with $\mathcal{O}(h^4)$ accuracy.

### `jacobian`
```python
jacobian(f: Callable[[Sequence[float]], Sequence[float]], x: Sequence[float], h: float = 1e-5) -> Matrix
```
- Computes $m \times n$ Jacobian matrix $J_{i, j} = \frac{\partial f_i}{\partial x_j}$.

### `hessian`
```python
hessian(f: Callable[[Sequence[float]], float], x: Sequence[float], h: float = 1e-4) -> Matrix
```
- Computes symmetric $n \times n$ Hessian matrix $H_{i, j} = \frac{\partial^2 f}{\partial x_i \partial x_j}$.

### `integrate`
```python
integrate(f: Callable[[float], float], a: float, b: float, method: str = "simpson", n: int = 1000) -> float
```
- Computes 1D definite numerical integral $\int_a^b f(x)\,dx$.
- Supported methods:
  - `"simpson"`: Composite Simpson's 1/3 rule ($\mathcal{O}(h^4)$) [Default]
  - `"trapezoidal"`: Composite Trapezoidal rule ($\mathcal{O}(h^2)$)
  - `"simpson_38"`: Composite Simpson's 3/8 rule
  - `"gauss_legendre"`: Gauss-Legendre 5-point quadrature

### `integrate_2d`
```python
integrate_2d(f: Callable[[float, float], float], x_a: float, x_b: float, y_a: float, y_b: float, nx: int = 100, ny: int = 100) -> float
```
- Computes 2D double integral $\int_{y_a}^{y_b} \int_{x_a}^{x_b} f(x, y)\,dx\,dy$.

### `limit`
```python
limit(f: Callable[[float], float], x_target: float | None = None, direction: str = "both", tol: float = 1e-6, *, x: float | None = None) -> float
```
- Computes numerical limit $\lim_{x \to a} f(x)$. Supports both `x` and `x_target` keywords.
- `direction`: `"both"` (two-sided), `"left"` ($x \to a^-$), or `"right"` ($x \to a^+$).

### `evaluate_limit`
```python
evaluate_limit(f: Callable[[float], float], x_target: float | None = None, direction: str = "both", tol: float = 1e-6, *, x: float | None = None) -> LimitResult
```
- Evaluates limit and returns diagnostic structure with fields:
  - `value: float`
  - `exists: bool`
  - `is_infinite: bool`
  - `direction: str`

### `@differentiable`
- Decorator equipping functions with `.derivative()`, `.grad()`, `.jacobian()`, `.hessian()`, `.integrate()`, and `.limit()`.

---

## Machine Learning Models

### `LinearRegression`
```python
LinearRegression(learning_rate: float = 0.01, epochs: int = 1000, fit_intercept: bool = True)
```
- `fit(X: Matrix, y: Vector) -> LinearRegression`: Fits model parameters using gradient descent.
- `predict(X: Matrix) -> list[float]`: Predicts continuous targets.
- `score(X: Matrix, y: Vector) -> float`: Returns coefficient of determination $R^2$.
- `weights -> Vector`: Learned weight coefficients.
- `bias -> float`: Learned bias/intercept.

### `LogisticRegression`
```python
LogisticRegression(learning_rate: float = 0.01, epochs: int = 1000, fit_intercept: bool = True)
```
- `fit(X: Matrix, y: Vector) -> LogisticRegression`: Fits binary classifier using cross-entropy loss.
- `predict(X: Matrix, threshold: float = 0.5) -> list[int]`: Predicts binary class labels $\{0, 1\}$.
- `predict_proba(X: Matrix) -> list[float]`: Computes predicted probabilities $P(y=1|\mathbf{x}) \in [0, 1]$.
- `score(X: Matrix, y: Vector) -> float`: Returns classification accuracy.

### `KNN`
```python
KNN(k: int = 3)
```
- `fit(X: Matrix, y: Vector) -> KNN`: Stores training dataset.
- `predict(X: Matrix) -> list[int]`: Predicts class labels using majority voting over Euclidean distances.

---

## Hardware Acceleration & Parallelism

MorphosML v0.4.0 automatically detects CPU vector extensions (AVX2, FMA) and leverages multi-core OpenMP thread pools across linear algebra operations, gradient descent backpropagation, and KNN inference.

### `get_simd_capabilities`
```python
morphosml.get_simd_capabilities() -> dict[str, bool]
```
Returns hardware features compiled and available at runtime on the host machine:
```python
{"avx2": True, "fma": True, "openmp": True}
```

### `get_num_threads`
```python
morphosml.get_num_threads() -> int
```
Returns the active number of worker threads utilized in OpenMP parallel regions.

### `set_num_threads`
```python
morphosml.set_num_threads(num_threads: int) -> None
```
Configures the number of parallel worker threads used across matrix-matrix multiplication, distance matrices, and model training routines.

---

## Evaluation Metrics & Utilities

- `train_test_split(X, y, test_size: float = 0.2, shuffle: bool = True, random_state: int | None = None)`: Splits arrays into train and test sets.
- `accuracy_score(y_true, y_pred) -> float`: Computes classification accuracy.
- `mean_squared_error(y_true, y_pred) -> float`: Computes Mean Squared Error (MSE).
- `r2_score(y_true, y_pred) -> float`: Computes $R^2$ score.

---

## LaTeX Formatting & Visualization

- `to_latex(obj: Any) -> str`: Formats any Matrix, Vector, or fitted Model into raw LaTeX string.
- `latexify(obj: Any)`: Wraps object into a LaTeX renderable object for Jupyter Notebook display.
- `enable_notebook_latex() -> None`: Automatically registers `_repr_latex_` hooks for all MorphosML types.

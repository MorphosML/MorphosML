# MorphosML API Reference

Complete API specification for the MorphosML library.

---

## Table of Contents
- [Linear Algebra Core](#linear-algebra-core)
  - [Matrix](#matrix)
  - [Vector](#vector)
  - [TensorView](#tensorview)
- [Data Ingestion & Streaming](#data-ingestion--streaming)
  - [Dataset](#dataset)
  - [NumpyDataset](#numpydataset)
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
  - [differentiable](#differentiable)
- [Machine Learning Models](#machine-learning-models)
  - [LinearRegression](#linearregression)
  - [LogisticRegression](#logisticregression)
  - [KNN](#knn)
- [Evaluation Metrics & Utilities](#evaluation-metrics--utilities)
  - [train_test_split](#train_test_split)
  - [accuracy](#accuracy)
  - [mse](#mse)
  - [r2_score](#r2_score)
  - [set_seed / get_seed](#set_seed--get_seed)
- [LaTeX Formatting & Visualization](#latex-formatting--visualization)
  - [to_latex](#to_latex)
  - [latexify](#latexify)
  - [enable_notebook_latex](#enable_notebook_latex)

---

## Linear Algebra Core

### `Matrix`
*HPC 2D dense matrix backed by a contiguous 1D flat row-major buffer.*

#### Methods
- `Matrix(rows: int, cols: int, init_val: float = 0.0)`: Initialize with dimensions and optional initial scalar value.
- `rows() -> int`: Number of rows.
- `cols() -> int`: Number of columns.
- `size() -> int`: Total elements (`rows * cols`).
- `empty() -> bool`: Returns `True` if empty.
- `mat[i, j] -> float`: Element access (0-indexed).
- `mat[i, j] = val`: Element assignment.
- `mat @ other` or `mat * other -> Matrix`: Cache-optimized $(i, k, j)$ matrix multiplication.
- `mat + other -> Matrix`: Element-wise sum.
- `mat - other -> Matrix`: Element-wise difference.
- `mat * scalar -> Matrix`: Scalar multiplication.
- `transpose() -> Matrix`: Transposed matrix of shape $(D, N)$.
- `to_list() -> list[list[float]]`: Convert to nested Python lists.

#### Static Factories
- `Matrix.identity(n: int) -> Matrix`: $n \times n$ identity matrix.
- `Matrix.zeros(rows: int, cols: int) -> Matrix`: Zero-filled matrix.
- `Matrix.random(rows: int, cols: int, seed: int = 42) -> Matrix`: Uniform random matrix in $[0, 1)$.

---

### `Vector`
*Dense 1D vector with Euclidean vector space operations.*

#### Methods
- `Vector(n: int)` or `Vector(list[float])`: Construct zero-vector of dimension $n$ or from list.
- `size() -> int`: Dimension count.
- `norm() -> float`: Euclidean $L_2$ norm $\|\mathbf{v}\|_2$.
- `normalized() -> Vector`: Unit-length normalized vector.
- `dot(other: Vector) -> float`: Inner dot product $\mathbf{u} \cdot \mathbf{v}$.
- `vec[i] -> float`: Element access.
- `vec + other`, `vec - other`: Element-wise vector addition and subtraction.
- `vec * scalar`: Scalar scaling.

---

### `TensorView`
*Non-owning, zero-copy 2D view over contiguous or strided float64 memory.*

#### Methods
- `rows() -> int`: Number of rows.
- `cols() -> int`: Number of columns.
- `stride() -> int`: Element distance between consecutive rows.
- `size() -> int`: Total logical elements (`rows * cols`).
- `empty() -> bool`: Returns `True` if empty.
- `view(r, c)` or `view[r, c] -> float`: Inlined sub-nanosecond coordinate access.
- `to_matrix() -> Matrix`: Materializes a full copy into an owning `Matrix`.

---

## Data Ingestion & Streaming

### `NumpyDataset`
*In-memory zero-copy dataset wrapping NumPy arrays via `TensorView`.*
- `NumpyDataset(X: np.ndarray, y: np.ndarray | None = None)`
- Factory: `morphosml.data.from_numpy(X, y)`

### `MMapDataset`
*Out-of-core memory-mapped dataset backed by POSIX `mmap` and `.mldat` binary files.*
- `MMapDataset(filepath: str)`: Opens and memory-maps `.mldat` binary file instantly (0 ms parse time).
- `get_slice(start_row: int, num_rows: int) -> TensorView`: Zero-copy slice over a range of rows.
- `full_view() -> TensorView`: Non-owning view spanning entire dataset.
- `to_numpy() -> np.ndarray`: Expose mapped dataset as a 2D NumPy array without copying.
- `MMapDataset.dump(filepath: str, data: np.ndarray) -> None`: Export NumPy array to `.mldat` format.

### `DataLoader`
*High-performance, fault-tolerant, idempotent DataLoader with asynchronous prefetching.*
- `DataLoader(dataset, batch_size=32, shuffle=True, seed=42, drop_last=False, epoch=0, prefetch_batches=2)`
- `set_epoch(epoch: int) -> None`: Updates epoch for deterministic seeded shuffling.
- `get_cursor() -> IngestionCursor`: Captures current batch progress into an ingestion cursor.
- `resume_from(cursor: IngestionCursor | str) -> None`: Restores exact checkpoint state.

### `IngestionCursor`
*Compact checkpoint token capturing training state.*
- Attributes: `epoch`, `sample_offset`, `checksum`.
- `to_string() -> str`: Format `"epoch:offset:checksum"`.
- `from_string(str) -> IngestionCursor`: Parse checkpoint token.

---

## Calculus & Differentiable Programming

### `derivative(f, x, order=1, h=0.0) -> float`
Calculates 1st, 2nd, or 3rd order derivative of $f(x)$ with $\mathcal{O}(h^4)$ 5-point stencils.

### `gradient(f, x, h=1e-5) -> np.ndarray`
Computes multivariate gradient vector $\nabla f(\mathbf{x})$ of $f: \mathbb{R}^n \to \mathbb{R}$.

### `jacobian(f, x, h=1e-5) -> Matrix`
Computes the $m \times n$ Jacobian matrix of vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$.

### `hessian(f, x, h=1e-4) -> Matrix`
Computes the symmetric $n \times n$ Hessian matrix of second-order partial derivatives.

### `integrate(f, a, b, method="simpson", n=1000) -> float`
Approximates definite integral $\int_a^b f(x)\,dx$.
Supported methods: `"simpson"`, `"trapezoidal"`, `"simpson_38"`, `"gauss_legendre"`.

### `integrate_2d(f, x_a, x_b, y_a, y_b, nx=100, ny=100) -> float`
Calculates 2D double integral $\int_{y_a}^{y_b} \int_{x_a}^{x_b} f(x, y)\,dx\,dy$.

### `limit(f, x_target, direction="both", tol=1e-6) -> float`
Evaluates numerical limit $\lim_{x \to a} f(x)$ with singularity and divergence handling.

### `@differentiable`
Decorator equipping any Python mathematical function with `.derivative()`, `.grad()`, `.jacobian()`, `.hessian()`, `.integrate()`, and `.limit()`.

---

## Machine Learning Models

### `LinearRegression`
- `LinearRegression(learning_rate=0.01, epochs=1000, fit_intercept=True)`
- `fit(X, y)`: Trains model using batch gradient descent.
- `predict(X) -> list[float]`: Predicts continuous targets.
- `score(X, y) -> float`: Computes $R^2$ score.
- `weights -> Vector`: Learned weight coefficients.
- `bias -> float`: Learned intercept.

### `LogisticRegression`
- `LogisticRegression(learning_rate=0.01, epochs=1000, fit_intercept=True)`
- `fit(X, y)`: Trains binary classifier with cross-entropy loss.
- `predict(X, threshold=0.5) -> list[int]`: Predicts binary class labels $\{0, 1\}$.
- `predict_proba(X) -> list[float]`: Computes predicted probabilities in $[0, 1]$.
- `score(X, y) -> float`: Computes classification accuracy.

### `KNN`
- `KNN(k=3)`: Initializes k-Nearest Neighbors classifier.
- `fit(X, y)`: Stores training exemplars.
- `predict(X) -> list[int]`: Majority voting prediction under Euclidean distance.

---

## Evaluation Metrics & Utilities

- `train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None)`: Splits datasets into train/test partitions.
- `accuracy(y_true, y_pred) -> float`: Computes classification accuracy.
- `mse(y_true, y_pred) -> float`: Computes Mean Squared Error.
- `r2_score(y_true, y_pred) -> float`: Computes coefficient of determination.
- `set_seed(seed: int) -> None`: Sets global deterministic random seed.
- `get_seed() -> int`: Retrieves active global random seed.

---

## LaTeX Formatting & Visualization

- `to_latex(obj) -> str`: Converts any MorphosML object to a LaTeX math string.
- `latexify(obj)`: Decorator or converter returning a Jupyter-renderable LaTeX wrapper.
- `enable_notebook_latex() -> None`: Automatically binds `_repr_latex_` to `Matrix`, `Vector`, `LinearRegression`, and `LogisticRegression` for inline rendering in notebooks.

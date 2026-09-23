"""Type stubs and interface definitions for MorphosML's compiled C++ core engine (_core).

Provides High-Performance Computing (HPC) primitives including contiguous dense matrices,
zero-copy tensor views, out-of-core memory mapping, differential calculus (derivatives,
gradients, Jacobians, Hessians), numerical quadrature, limit solvers, and native ML models.
"""

from collections.abc import Sequence
from typing import Any, overload

import numpy as np

class Vector:
    """Dense 1D vector backed by contiguous double-precision floating-point memory."""

    def __init__(self, data: int | list[float] | np.ndarray = ...) -> None:
        """Initialize a Vector with a dimension size (zero-initialized) or from a list or NumPy array of floats."""
        ...

    def size(self) -> int:
        """Return the number of elements in the vector."""
        ...

    def __len__(self) -> int:
        """Return the number of elements in the vector."""
        ...

    def __getitem__(self, i: int) -> float:
        """Retrieve element at index i."""
        ...

    def __setitem__(self, i: int, val: float) -> None:
        """Set element at index i to val."""
        ...

    def dot(self, other: Vector) -> float:
        """Compute the Euclidean inner product u . v = sum(u_i * v_i)."""
        ...

    def norm(self) -> float:
        """Compute the Euclidean L2 norm ||v||_2 = sqrt(sum(v_i^2))."""
        ...

    def normalized(self) -> Vector:
        """Return a unit-normalized vector with L2 norm = 1.0."""
        ...

    def __add__(self, other: Vector) -> Vector:
        """Element-wise vector addition."""
        ...

    def __sub__(self, other: Vector) -> Vector:
        """Element-wise vector subtraction."""
        ...

    def __mul__(self, scalar: float) -> Vector:
        """Scalar multiplication."""
        ...

    def __rmul__(self, scalar: float) -> Vector:
        """Right-hand scalar multiplication."""
        ...

    def __eq__(self, other: object) -> bool:
        """Element-wise vector equality comparison."""
        ...

    def to_list(self) -> list[float]:
        """Convert vector into a standard Python list of floats."""
        ...

class Matrix:
    """High-Performance Computing (HPC) 2D dense matrix backed by a contiguous 1D flat buffer."""

    @overload
    def __init__(self, rows: int = ..., cols: int = ...) -> None:
        """Initialize an empty matrix or a (rows x cols) matrix zero-initialized."""
        ...

    @overload
    def __init__(self, data: list[list[float]] | np.ndarray) -> None:
        """Initialize a matrix from nested lists or a 2D NumPy array."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Return dimensions (rows, cols) of the matrix."""
        ...

    def rows(self) -> int:
        """Return the number of rows."""
        ...

    def cols(self) -> int:
        """Return the number of columns."""
        ...

    def size(self) -> int:
        """Return the total number of elements (rows * cols)."""
        ...

    def view(self) -> TensorView:
        """Return a non-owning zero-copy TensorView over the matrix."""
        ...

    def row(self, i: int) -> Vector:
        """Extract row i as an owning Vector."""
        ...

    def __call__(self, i: int, j: int) -> float:
        """Access element at coordinate (i, j)."""
        ...

    def __getitem__(self, idx: tuple[int, int] | int) -> float | Vector:
        """Access element via 2D tuple index `mat[i, j]` or row via `mat[i]`."""
        ...

    def __setitem__(self, idx: tuple[int, int], val: float) -> None:
        """Set element via 2D tuple index `mat[i, j] = val`."""
        ...

    def __mul__(self, other: Matrix) -> Matrix:
        """Compute cache-optimized matrix product A @ B using an (i, k, j) loop traversal."""
        ...

    def __add__(self, other: Matrix) -> Matrix:
        """Element-wise matrix addition."""
        ...

    def __sub__(self, other: Matrix) -> Matrix:
        """Element-wise matrix subtraction."""
        ...

    def __eq__(self, other: object) -> bool:
        """Element-wise matrix equality comparison."""
        ...

    def transpose(self) -> Matrix:
        """Return the transposed matrix of shape (cols, rows)."""
        ...

    @staticmethod
    def identity(n: int) -> Matrix:
        """Construct an (n x n) identity matrix."""
        ...

    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix:
        """Construct a (rows x cols) matrix populated with zeros."""
        ...

    @staticmethod
    def random(rows: int, cols: int, seed: int = ...) -> Matrix:
        """Construct a (rows x cols) matrix populated with uniform random numbers in [0, 1)."""
        ...

    def to_list(self) -> list[list[float]]:
        """Convert matrix into a nested Python list of floats."""
        ...

class KNN:
    """k-Nearest Neighbors (k-NN) classification model."""

    def __init__(self, k: int = 3) -> None:
        """Initialize KNN with neighborhood size k."""
        ...

    def fit(self, X: np.ndarray, y: list[int]) -> None:
        """Store training points and integer class labels."""
        ...

    def predict(self, X: np.ndarray) -> list[int]:
        """Predict majority vote class labels for query samples."""
        ...

    @property
    def k(self) -> int:
        """Return neighborhood size k."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        ...

class LinearRegression:
    """Multivariate Linear Regression trained via Batch Gradient Descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        fit_intercept: bool = True,
    ) -> None:
        """Initialize LinearRegression hyperparameters."""
        ...

    def fit(self, X: np.ndarray, y: list[float]) -> None:
        """Fit model weights to continuous training data."""
        ...

    def predict(self, X: np.ndarray) -> list[float]:
        """Predict continuous target values for feature matrix X."""
        ...

    def score(self, X: np.ndarray, y: list[float]) -> float:
        """Calculate R^2 coefficient of determination on query data."""
        ...

    def compute_cost(self, X: np.ndarray, y: list[float]) -> float:
        """Compute Mean Squared Error (MSE) cost."""
        ...

    @property
    def weights(self) -> Vector:
        """Return learned weight coefficients vector."""
        ...

    @property
    def bias(self) -> float:
        """Return learned intercept scalar."""
        ...

    @property
    def learning_rate(self) -> float:
        """Return optimization learning rate."""
        ...

    @property
    def epochs(self) -> int:
        """Return maximum optimization epochs."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        ...

class LogisticRegression:
    """Binary Logistic Regression trained via Gradient Descent with Cross-Entropy Loss."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        fit_intercept: bool = True,
    ) -> None:
        """Initialize LogisticRegression hyperparameters."""
        ...

    def fit(self, X: np.ndarray, y: list[int]) -> None:
        """Fit logistic model to binary classification data (labels in {0, 1})."""
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> list[int]:
        """Predict discrete binary class labels based on a probability threshold."""
        ...

    def predict_proba(self, X: np.ndarray) -> list[float]:
        """Predict class-1 posterior probability estimates in [0, 1]."""
        ...

    def score(self, X: np.ndarray, y: list[int]) -> float:
        """Calculate classification accuracy on evaluation data."""
        ...

    def compute_loss(self, X: np.ndarray, y: list[int]) -> float:
        """Compute binary cross-entropy loss."""
        ...

    @property
    def weights(self) -> Vector:
        """Return learned weight coefficients vector."""
        ...

    @property
    def bias(self) -> float:
        """Return learned intercept scalar."""
        ...

    @property
    def learning_rate(self) -> float:
        """Return optimization learning rate."""
        ...

    @property
    def epochs(self) -> int:
        """Return maximum optimization epochs."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        ...

def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Compute classification accuracy score between ground truth and predicted labels."""
    ...

def mse(y_true: list[float], y_pred: list[float]) -> float:
    """Compute Mean Squared Error (MSE) between ground truth and predicted continuous targets."""
    ...

def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    """Compute R^2 (coefficient of determination) regression score."""
    ...

def set_seed(seed: int) -> None:
    """Set global deterministic random seed across all MorphosML C++ routines."""
    ...

def get_seed() -> int:
    """Retrieve the current global deterministic random seed."""
    ...

class TensorView:
    """Non-owning, zero-copy 2D view over an externally allocated float64 memory buffer."""

    @overload
    def __init__(self) -> None:
        """Construct an empty TensorView."""
        ...

    @overload
    def __init__(self, target: Matrix | np.ndarray) -> None:
        """Construct a zero-copy TensorView directly wrapping a Matrix or 2D NumPy array."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Return dimensions (rows, cols) of the view."""
        ...

    def rows(self) -> int:
        """Return number of rows."""
        ...

    def cols(self) -> int:
        """Return number of columns."""
        ...

    def stride(self) -> int:
        """Return row stride (element distance between consecutive rows)."""
        ...

    def size(self) -> int:
        """Return total number of logical elements (rows * cols)."""
        ...

    def empty(self) -> bool:
        """Return True if the view contains zero elements."""
        ...

    def slice(self, start_row: int, num_rows: int) -> TensorView:
        """Return a zero-copy row slice spanning num_rows starting at start_row."""
        ...

    def subview(
        self, start_row: int, num_rows: int, start_col: int, num_cols: int
    ) -> TensorView:
        """Return a zero-copy 2D subview spanning num_rows and num_cols."""
        ...

    def row(self, r: int) -> Vector:
        """Extract row r as an owning Vector."""
        ...

    def to_matrix(self) -> Matrix:
        """Copy and materialize the view into an owning MorphosML Matrix."""
        ...

    def to_numpy(self) -> np.ndarray:
        """Materialize the view into a contiguous NumPy array."""
        ...

    def __call__(self, r: int, c: int) -> float:
        """Access element at row r and column c."""
        ...

    def __getitem__(
        self, idx: tuple[int, int] | int | slice
    ) -> float | Vector | TensorView:
        """Access element via 2D tuple index `view[r, c]`, row via `view[r]`, or slice via `view[start:stop]`."""
        ...

    def __eq__(self, other: object) -> bool:
        """Check element-wise equality between two tensor views."""
        ...

def tensor_view_from_numpy(arr: np.ndarray) -> TensorView:
    """Wrap a 2D float64 NumPy array in a non-owning zero-copy TensorView."""
    ...

class IngestionCursor:
    """Lightweight, serializable checkpoint token capturing the exact state of a data pipeline."""

    epoch: int
    sample_offset: int
    checksum: int

    def __init__(self) -> None:
        """Construct a default cursor at epoch 0, offset 0."""
        ...

    def to_string(self) -> str:
        """Serialize cursor state into a compact string token (epoch:offset:checksum)."""
        ...

    @staticmethod
    def from_string(s: str) -> IngestionCursor:
        """Deserialize a cursor token string back into an IngestionCursor instance."""
        ...

    def __eq__(self, other: object) -> bool:
        """Check equality between two ingestion cursors."""
        ...

class IdempotentSampler:
    """Deterministic SplitMix64 pseudo-random permutation generator for dataset shuffling."""

    seed: int

    def __init__(self, seed: int = 42) -> None:
        """Initialize sampler with a reproducible random seed."""
        ...

    def generate_indices(self, total_samples: int, epoch: int = 0) -> list[int]:
        """Generate a deterministic index permutation for a dataset at a specific epoch."""
        ...

class MMapBuffer:
    """POSIX memory-mapped buffer reader for out-of-core .mldat binary datasets."""

    def __init__(self, filepath: str) -> None:
        """Open and memory-map a .mldat file into the virtual address space."""
        ...

    def rows(self) -> int:
        """Return total row count in the dataset."""
        ...

    def cols(self) -> int:
        """Return feature dimension count."""
        ...

    def size(self) -> int:
        """Return total number of float64 elements."""
        ...

    def checksum(self) -> int:
        """Return 64-bit integrity checksum stored in file header."""
        ...

    def filepath(self) -> str:
        """Return filesystem path of the mapped file."""
        ...

    def get_slice(self, start_row: int, num_rows: int) -> TensorView:
        """Extract a zero-copy TensorView slice across a range of rows."""
        ...

    def full_view(self) -> TensorView:
        """Return a zero-copy TensorView spanning the entire mapped dataset."""
        ...

    @staticmethod
    def write_mldat(filepath: str, mat: Matrix) -> None:
        """Serialize a MorphosML Matrix into a .mldat binary file."""
        ...

    @staticmethod
    def write_mldat_numpy(filepath: str, arr: np.ndarray) -> None:
        """Serialize a 2D float64 NumPy array into a .mldat binary file."""
        ...

class LimitResult:
    """Structured outcome of a numerical limit evaluation."""

    value: float
    exists: bool
    is_infinite: bool
    direction: str

class _CalculusModule:
    """Submodule containing high-precision C++ numerical differentiation, integration, and limits."""

    LimitResult: type[LimitResult]

    @staticmethod
    def derivative(f: Any, x: float, order: int = 1, h: float = 0.0) -> float:
        """Compute the 1st, 2nd, or 3rd order derivative of f(x) using O(h^4) central stencils."""
        ...

    @staticmethod
    def gradient(f: Any, x: Sequence[float], h: float = 1e-5) -> list[float]:
        """Compute the multivariate gradient vector grad f(x) using 4th-order central differences."""
        ...

    @staticmethod
    def jacobian(f: Any, x: Sequence[float], h: float = 1e-5) -> Matrix:
        """Compute the Jacobian matrix (m x n) of vector-valued function f: R^n -> R^m."""
        ...

    @staticmethod
    def hessian(f: Any, x: Sequence[float], h: float = 1e-4) -> Matrix:
        """Compute the symmetric Hessian matrix (n x n) of second-order partial derivatives."""
        ...

    @staticmethod
    def trapezoidal(f: Any, a: float, b: float, n: int = 1000) -> float:
        """Compute numerical definite integral using the Composite Trapezoidal rule."""
        ...

    @staticmethod
    def simpson(f: Any, a: float, b: float, n: int = 1000) -> float:
        """Compute numerical definite integral using Composite Simpson's 1/3 rule (O(h^4))."""
        ...

    @staticmethod
    def simpson_38(f: Any, a: float, b: float, n: int = 999) -> float:
        """Compute numerical definite integral using Composite Simpson's 3/8 rule."""
        ...

    @staticmethod
    def gauss_legendre(f: Any, a: float, b: float, n_points: int = 5) -> float:
        """Compute numerical definite integral using Gauss-Legendre Quadrature (5 or 10 points)."""
        ...

    @staticmethod
    def integrate(
        f: Any, a: float, b: float, method: str = "simpson", n: int = 1000
    ) -> float:
        """Compute 1D definite integral with selected numerical quadrature method."""
        ...

    @staticmethod
    def integrate_2d(
        f: Any,
        x_a: float,
        x_b: float,
        y_a: float,
        y_b: float,
        nx: int = 100,
        ny: int = 100,
    ) -> float:
        """Compute 2D double integral over rectangular region [x_a, x_b] x [y_a, y_b]."""
        ...

    @staticmethod
    def evaluate_limit(
        f: Any, x_target: float, direction: str = "both", tol: float = 1e-6
    ) -> LimitResult:
        """Evaluate directional limit lim_{x -> x_target} f(x) with singularity & divergence analysis."""
        ...

    @staticmethod
    def limit(
        f: Any, x_target: float, direction: str = "both", tol: float = 1e-6
    ) -> float:
        """Compute numerical limit value directly (raises RuntimeError if limit does not exist)."""
        ...

calculus: _CalculusModule

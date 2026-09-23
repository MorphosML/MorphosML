"""MorphosML - High-Performance Machine Learning & Differentiable Programming with a C++ Core.

Modules:
- `data`: Out-of-core memory mapping (mmap), zero-copy buffer protocol, idempotent sampling, and async prefetching.
- `calculus`: Numerical differentiation (O(h^4) stencils), multivariate gradients, Jacobians, Hessians, quadrature, and limits.
- `latex`: Mathematical LaTeX string formatting and interactive Jupyter notebook rendering.
- `utils`: Train/test splits and dataset splitting utilities.

Core Types & Models:
- `Matrix`: Contiguous 1D-backed 2D dense matrix with cache-optimized multiplication.
- `Vector`: 1D dense mathematical vector with Euclidean operations.
- `TensorView`: Non-owning zero-copy 2D view over contiguous or strided float64 memory.
- `LinearRegression`: Multivariate linear regression via gradient descent.
- `LogisticRegression`: Binary logistic regression with sigmoid activation and log-loss.
- `KNN`: k-Nearest Neighbors classifier with Euclidean distance metric.
"""

from . import (
    _core,  # type: ignore[import]
    calculus,
    data,
    latex,
)
from .calculus import (
    derivative,
    differentiable,
    gradient,
    hessian,
    integrate,
    integrate_2d,
    jacobian,
    limit,
)
from .data import (
    DataLoader,
    Dataset,
    IdempotentSampler,
    IngestionCursor,
    MMapBuffer,
    MMapDataset,
    NumpyDataset,
    from_numpy,
)
from .latex import enable_notebook_latex, latexify, to_latex
from .utils import train_test_split

__version__ = "0.3.0"
__author__ = "Gabriel Carmona"
__license__ = "MIT"

# Main exports
KNN = _core.KNN
Vector = _core.Vector
Matrix = _core.Matrix
TensorView = _core.TensorView
LinearRegression = _core.LinearRegression
LogisticRegression = _core.LogisticRegression
accuracy = _core.accuracy
mse = _core.mse
r2_score = _core.r2_score
set_seed = _core.set_seed
get_seed = _core.get_seed
tensor_view_from_numpy = _core.tensor_view_from_numpy

# Initialize notebook LaTeX representation hooks
enable_notebook_latex()

__all__ = [
    "KNN",
    "DataLoader",
    "Dataset",
    "IdempotentSampler",
    "IngestionCursor",
    "LinearRegression",
    "LogisticRegression",
    "MMapBuffer",
    "MMapDataset",
    "Matrix",
    "NumpyDataset",
    "TensorView",
    "Vector",
    "__version__",
    "accuracy",
    "calculus",
    "data",
    "derivative",
    "differentiable",
    "from_numpy",
    "get_seed",
    "gradient",
    "hessian",
    "integrate",
    "integrate_2d",
    "jacobian",
    "latex",
    "latexify",
    "limit",
    "mse",
    "r2_score",
    "set_seed",
    "tensor_view_from_numpy",
    "to_latex",
    "train_test_split",
]

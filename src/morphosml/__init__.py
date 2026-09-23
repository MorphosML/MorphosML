"""MorphosML - High-performance machine learning with C++ core."""

__version__ = "0.3.0"
__author__ = "Gabriel Carmona"
__license__ = "MIT"

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
from .latex import enable_notebook_latex, latexify, to_latex
from .utils import train_test_split

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

# Initialize notebook LaTeX representation hooks
enable_notebook_latex()

__all__ = [
    "KNN",
    "LinearRegression",
    "LogisticRegression",
    "Matrix",
    "TensorView",
    "Vector",
    "__version__",
    "accuracy",
    "calculus",
    "data",
    "derivative",
    "differentiable",
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
    "to_latex",
    "train_test_split",
]

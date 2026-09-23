"""MorphosML - High-performance machine learning with C++ core."""

__version__ = "0.2.1"
__author__ = "Gabriel Carmona"
__license__ = "MIT"

from . import (
    _core,  # type: ignore[import]
    data,
)
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

__all__ = [
    "KNN",
    "LinearRegression",
    "LogisticRegression",
    "Matrix",
    "TensorView",
    "Vector",
    "__version__",
    "accuracy",
    "data",
    "get_seed",
    "mse",
    "r2_score",
    "set_seed",
    "train_test_split",
]

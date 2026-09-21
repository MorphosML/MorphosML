"""MorphosML - High-performance machine learning with C++ core."""

__version__ = "0.2.0"
__author__ = "Gabriel Carmona"
__license__ = "MIT"

from . import _core  # type: ignore[import]

# Main exports
KNN = _core.KNN
Vector = _core.Vector
Matrix = _core.Matrix
accuracy = _core.accuracy
mse = _core.mse

__all__ = [
    "KNN",
    "Vector",
    "Matrix",
    "accuracy",
    "mse",
    "__version__",
]
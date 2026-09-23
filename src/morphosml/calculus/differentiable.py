from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from morphosml._core import Matrix, Vector
from morphosml._core import calculus as _core_calc


def derivative(
    f: Callable[[float], float], x: float, order: int = 1, h: float = 0.0
) -> float:
    """Compute numerical derivative of a single-variable function f(x) up to 3rd order.

    Parameters:
        f: Function mapping float -> float.
        x: Evaluation point.
        order: Order of the derivative (1, 2, or 3).
        h: Step size. If 0.0, optimal adaptive step size is chosen.
    """
    return _core_calc.derivative(f, float(x), order, float(h))


def gradient(
    f: Callable[[Sequence[float]], float],
    x: Sequence[float] | np.ndarray | Vector,
    h: float = 1e-5,
) -> np.ndarray:
    """Compute numerical gradient grad f(x) for a multivariate scalar function f: R^n -> R.

    Parameters:
        f: Function mapping Sequence[float] -> float.
        x: Evaluation point in R^n.
        h: Finite difference step size.

    Returns:
        NumPy 1D array containing the partial derivatives.
    """
    if isinstance(x, (np.ndarray, Vector)):
        x_list = [float(v) for v in x]
    else:
        x_list = [float(v) for v in x]

    raw_grad = _core_calc.gradient(f, x_list, float(h))
    return np.array(raw_grad, dtype=np.float64)


def jacobian(
    f: Callable[[Sequence[float]], Sequence[float]],
    x: Sequence[float] | np.ndarray | Vector,
    h: float = 1e-5,
) -> Matrix:
    """Compute numerical Jacobian matrix J for a vector-valued function f: R^n -> R^m.

    Parameters:
        f: Function mapping Sequence[float] -> Sequence[float].
        x: Evaluation point in R^n.
        h: Finite difference step size.

    Returns:
        MorphosML Matrix of shape (m, n) where J[i, j] = d f_i / d x_j.
    """
    if isinstance(x, (np.ndarray, Vector)):
        x_list = [float(v) for v in x]
    else:
        x_list = [float(v) for v in x]

    def wrapped_f(v: list[float]) -> list[float]:
        res = f(v)
        if isinstance(res, (np.ndarray, Vector)):
            return [float(val) for val in res]
        return [float(val) for val in res]

    return _core_calc.jacobian(wrapped_f, x_list, float(h))


def hessian(
    f: Callable[[Sequence[float]], float],
    x: Sequence[float] | np.ndarray | Vector,
    h: float = 1e-4,
) -> Matrix:
    """Compute numerical Hessian matrix H for a multivariate function f: R^n -> R.

    Parameters:
        f: Function mapping Sequence[float] -> float.
        x: Evaluation point in R^n.
        h: Finite difference step size.

    Returns:
        Symmetric MorphosML Matrix of shape (n, n) where H[i, j] = d^2 f / (d x_i d x_j).
    """
    if isinstance(x, (np.ndarray, Vector)):
        x_list = [float(v) for v in x]
    else:
        x_list = [float(v) for v in x]

    return _core_calc.hessian(f, x_list, float(h))


def integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    method: str = "simpson",
    n: int = 1000,
) -> float:
    """Compute numerical 1D definite integral integral_a^b f(x) dx.

    Supported methods:
        - 'simpson' / 'simpson_13': Simpson's 1/3 rule (O(h^4))
        - 'trapezoidal': Trapezoidal rule (O(h^2))
        - 'simpson_38': Simpson's 3/8 rule
        - 'gauss_legendre': Gauss-Legendre quadrature
    """
    return _core_calc.integrate(f, float(a), float(b), method, int(n))


def integrate_2d(
    f: Callable[[float, float], float],
    x_a: float,
    x_b: float,
    y_a: float,
    y_b: float,
    nx: int = 100,
    ny: int = 100,
) -> float:
    """Compute numerical double integral integral_{y_a}^{y_b} integral_{x_a}^{x_b} f(x, y) dx dy."""
    return _core_calc.integrate_2d(
        f, float(x_a), float(x_b), float(y_a), float(y_b), int(nx), int(ny)
    )


def limit(
    f: Callable[[float], float],
    x_target: float,
    direction: str = "both",
    tol: float = 1e-6,
) -> float:
    """Compute numerical limit lim_{x -> x_target} f(x).

    direction:
        - 'left': Left-hand limit x -> a^-
        - 'right': Right-hand limit x -> a^+
        - 'both': Two-sided limit (raises RuntimeError if non-existent or discontinuous)
    """
    return _core_calc.limit(f, float(x_target), direction, float(tol))


def evaluate_limit(
    f: Callable[[float], float],
    x_target: float,
    direction: str = "both",
    tol: float = 1e-6,
) -> _core_calc.LimitResult:
    """Evaluate numerical limit returning full diagnostics struct."""
    return _core_calc.evaluate_limit(f, float(x_target), direction, float(tol))


class DifferentiableFunction:
    """Wrapper / decorator that equips any function with numerical differentiation methods."""

    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn

    def __call__(self, *args, **kwargs) -> Any:
        return self._fn(*args, **kwargs)

    def derivative(self, x: float, order: int = 1, h: float = 0.0) -> float:
        return derivative(self._fn, x, order=order, h=h)

    def grad(self, x: Sequence[float], h: float = 1e-5) -> np.ndarray:
        return gradient(self._fn, x, h=h)

    def jacobian(self, x: Sequence[float], h: float = 1e-5) -> Matrix:
        return jacobian(self._fn, x, h=h)

    def hessian(self, x: Sequence[float], h: float = 1e-4) -> Matrix:
        return hessian(self._fn, x, h=h)

    def integrate(
        self, a: float, b: float, method: str = "simpson", n: int = 1000
    ) -> float:
        return integrate(self._fn, a, b, method=method, n=n)

    def limit(
        self, x_target: float, direction: str = "both", tol: float = 1e-6
    ) -> float:
        return limit(self._fn, x_target, direction=direction, tol=tol)


def differentiable(fn: Callable[..., Any]) -> DifferentiableFunction:
    """Decorator to make a Python mathematical function differentiable."""
    return DifferentiableFunction(fn)

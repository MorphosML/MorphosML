"""Differentiable programming, numerical differentiation, integration, and limits in Python.

Provides Pythonic wrappers around MorphosML's compiled C++ calculus engine:
- `derivative`: 1st, 2nd, and 3rd order scalar derivatives with O(h^4) 5-point stencils.
- `gradient`: Multivariate gradient vector using 4th-order central differences.
- `jacobian`: Jacobian matrix of vector-valued functions f: R^n -> R^m.
- `hessian`: Symmetric Hessian matrix of second-order partial derivatives.
- `integrate`: 1D numerical quadrature (Trapezoidal, Simpson's 1/3, Simpson's 3/8, Gauss-Legendre).
- `integrate_2d`: 2D numerical double integration over rectangular regions.
- `limit` / `evaluate_limit`: Directional limits with singularity and divergence diagnostics.
- `@differentiable`: Decorator equipping functions with differentiation and calculus methods.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import numpy as np

from morphosml._core import Matrix, Vector
from morphosml._core import calculus as _core_calc


def derivative(
    f: Callable[[float], float], x: float, order: int = 1, h: float = 0.0
) -> float:
    """Compute the numerical derivative of a scalar function f(x) up to 3rd order.

    Parameters
    ----------
    f : Callable[[float], float]
        Single-variable scalar function.
    x : float
        Evaluation coordinate.
    order : int, default=1
        Derivative order (1 for f', 2 for f'', 3 for f''').
    h : float, default=0.0
        Finite difference perturbation step size. If 0.0, optimal step size is
        chosen adaptively based on machine epsilon.

    Returns
    -------
    float
        Computed numerical derivative value.

    Raises
    ------
    ValueError or InvalidArgument
        If order is not 1, 2, or 3.
    """
    return _core_calc.derivative(f, float(x), order, float(h))


def gradient(
    f: Callable[[Sequence[float]], float],
    x: Union[Sequence[float], np.ndarray, Vector],
    h: float = 1e-5,
) -> np.ndarray:
    """Compute numerical gradient grad f(x) for a multivariate scalar function f: R^n -> R.

    Parameters
    ----------
    f : Callable[[Sequence[float]], float]
        Scalar multivariate function taking a coordinate sequence of length n.
    x : Sequence[float], numpy.ndarray, or Vector
        Evaluation point in R^n.
    h : float, default=1e-5
        Perturbation step size.

    Returns
    -------
    numpy.ndarray
        1D array of length n containing the partial derivatives [df/dx_1, ..., df/dx_n].
    """
    if isinstance(x, (np.ndarray, Vector)):
        x_list = [float(v) for v in x]
    else:
        x_list = [float(v) for v in x]

    raw_grad = _core_calc.gradient(f, x_list, float(h))
    return np.array(raw_grad, dtype=np.float64)


def jacobian(
    f: Callable[[Sequence[float]], Sequence[float]],
    x: Union[Sequence[float], np.ndarray, Vector],
    h: float = 1e-5,
) -> Matrix:
    """Compute numerical Jacobian matrix J for a vector-valued function f: R^n -> R^m.

    Parameters
    ----------
    f : Callable[[Sequence[float]], Sequence[float]]
        Vector-valued function mapping R^n -> R^m.
    x : Sequence[float], numpy.ndarray, or Vector
        Evaluation point in R^n.
    h : float, default=1e-5
        Perturbation step size.

    Returns
    -------
    Matrix
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
    x: Union[Sequence[float], np.ndarray, Vector],
    h: float = 1e-4,
) -> Matrix:
    """Compute numerical Hessian matrix H for a multivariate scalar function f: R^n -> R.

    Parameters
    ----------
    f : Callable[[Sequence[float]], float]
        Multivariate scalar function.
    x : Sequence[float], numpy.ndarray, or Vector
        Evaluation point in R^n.
    h : float, default=1e-4
        Perturbation step size.

    Returns
    -------
    Matrix
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

    Parameters
    ----------
    f : Callable[[float], float]
        Integrand function.
    a : float
        Lower integration limit.
    b : float
        Upper integration limit.
    method : str, default='simpson'
        Quadrature algorithm:
        - 'simpson': Composite Simpson's 1/3 rule (O(h^4))
        - 'trapezoidal': Composite Trapezoidal rule (O(h^2))
        - 'simpson_38': Composite Simpson's 3/8 rule
        - 'gauss_legendre': Gauss-Legendre quadrature (5 points)
    n : int, default=1000
        Number of subintervals.

    Returns
    -------
    float
        Computed definite integral value.
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
    """Compute numerical double integral integral_{y_a}^{y_b} integral_{x_a}^{x_b} f(x, y) dx dy.

    Parameters
    ----------
    f : Callable[[float, float], float]
        2D integrand function f(x, y).
    x_a : float
        Lower bound for x.
    x_b : float
        Upper bound for x.
    y_a : float
        Lower bound for y.
    y_b : float
        Upper bound for y.
    nx : int, default=100
        Subintervals along x.
    ny : int, default=100
        Subintervals along y.

    Returns
    -------
    float
        Computed 2D double integral volume.
    """
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

    Parameters
    ----------
    f : Callable[[float], float]
        Function to evaluate.
    x_target : float
        Target evaluation limit point.
    direction : str, default='both'
        Direction of approach:
        - 'both': Two-sided limit (requires left and right limits to agree within tol).
        - 'left': Left-hand limit x -> a^-.
        - 'right': Right-hand limit x -> a^+.
    tol : float, default=1e-6
        Relative convergence tolerance.

    Returns
    -------
    float
        Computed limit value.

    Raises
    ------
    RuntimeError
        If the two-sided limit does not exist or if the function diverges to infinity.
    """
    return _core_calc.limit(f, float(x_target), direction, float(tol))


def evaluate_limit(
    f: Callable[[float], float],
    x_target: float,
    direction: str = "both",
    tol: float = 1e-6,
) -> _core_calc.LimitResult:
    """Evaluate numerical limit and return full diagnostic structure.

    Parameters
    ----------
    f : Callable[[float], float]
        Function to evaluate.
    x_target : float
        Target evaluation point.
    direction : str, default='both'
        Direction of approach ('both', 'left', or 'right').
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    LimitResult
        Result containing `value`, `exists`, `is_infinite`, and `direction`.
    """
    return _core_calc.evaluate_limit(f, float(x_target), direction, float(tol))


class DifferentiableFunction:
    """Wrapper that equips a standard Python function with differential calculus methods."""

    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn

    def __call__(self, *args, **kwargs) -> Any:
        return self._fn(*args, **kwargs)

    def derivative(self, x: float, order: int = 1, h: float = 0.0) -> float:
        """Compute scalar derivative up to 3rd order."""
        return derivative(self._fn, x, order=order, h=h)

    def grad(self, x: Sequence[float], h: float = 1e-5) -> np.ndarray:
        """Compute multivariate gradient vector."""
        return gradient(self._fn, x, h=h)

    def jacobian(self, x: Sequence[float], h: float = 1e-5) -> Matrix:
        """Compute Jacobian matrix."""
        return jacobian(self._fn, x, h=h)

    def hessian(self, x: Sequence[float], h: float = 1e-4) -> Matrix:
        """Compute symmetric Hessian matrix."""
        return hessian(self._fn, x, h=h)

    def integrate(
        self, a: float, b: float, method: str = "simpson", n: int = 1000
    ) -> float:
        """Compute 1D numerical definite integral."""
        return integrate(self._fn, a, b, method=method, n=n)

    def limit(
        self, x_target: float, direction: str = "both", tol: float = 1e-6
    ) -> float:
        """Compute numerical limit."""
        return limit(self._fn, x_target, direction=direction, tol=tol)


def differentiable(fn: Callable[..., Any]) -> DifferentiableFunction:
    """Decorator to make a mathematical function differentiable with fluent calculus methods.

    Examples
    --------
    >>> @differentiable
    ... def f(x):
    ...     return x ** 3
    >>> f.derivative(2.0)
    12.0
    >>> f.integrate(0.0, 2.0)
    4.0
    """
    return DifferentiableFunction(fn)

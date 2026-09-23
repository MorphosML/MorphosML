from __future__ import annotations

from typing import Any

import numpy as np


def matrix_to_latex(
    mat: Any, max_rows: int = 6, max_cols: int = 6, precision: int = 4
) -> str:
    """Format a MorphosML Matrix or 2D array into a LaTeX pmatrix string."""
    rows = mat.rows() if hasattr(mat, "rows") else mat.shape[0]
    cols = mat.cols() if hasattr(mat, "cols") else mat.shape[1]

    if rows == 0 or cols == 0:
        return r"\begin{pmatrix}\end{pmatrix}"

    def fmt(val: float) -> str:
        if abs(val - round(val)) < 1e-9:
            return str(round(val))
        return f"{val:.{precision}f}".rstrip("0").rstrip(".")

    # If small enough, output all elements
    if rows <= max_rows and cols <= max_cols:
        lines = []
        for r in range(rows):
            row_vals = [
                fmt(mat[r, c] if hasattr(mat, "__getitem__") else mat(r, c))
                for c in range(cols)
            ]
            lines.append(" & ".join(row_vals))
        return r"\begin{pmatrix}" + " \\\\ ".join(lines) + r"\end{pmatrix}"

    # Truncate with dots for large matrices
    half_r = max_rows // 2
    half_c = max_cols // 2

    row_indices = list(range(half_r)) + [-1] + list(range(rows - half_r, rows))
    col_indices = list(range(half_c)) + [-1] + list(range(cols - half_c, cols))

    lines = []
    for r in row_indices:
        if r == -1:
            line_parts = []
            for c in col_indices:
                line_parts.append(r"\ddots" if c == -1 else r"\vdots")
            lines.append(" & ".join(line_parts))
        else:
            line_parts = []
            for c in col_indices:
                if c == -1:
                    line_parts.append(r"\dots")
                else:
                    val = mat[r, c] if hasattr(mat, "__getitem__") else mat(r, c)
                    line_parts.append(fmt(val))
            lines.append(" & ".join(line_parts))

    return r"\begin{pmatrix}" + " \\\\ ".join(lines) + r"\end{pmatrix}"


def vector_to_latex(vec: Any, max_len: int = 8, precision: int = 4) -> str:
    """Format a MorphosML Vector or 1D array into a LaTeX column vector."""
    n = vec.size() if hasattr(vec, "size") else len(vec)
    if n == 0:
        return r"\begin{pmatrix}\end{pmatrix}"

    def fmt(val: float) -> str:
        if abs(val - round(val)) < 1e-9:
            return str(round(val))
        return f"{val:.{precision}f}".rstrip("0").rstrip(".")

    if n <= max_len:
        items = [fmt(vec[i]) for i in range(n)]
        return r"\begin{pmatrix}" + " \\\\ ".join(items) + r"\end{pmatrix}"

    half = max_len // 2
    head = [fmt(vec[i]) for i in range(half)]
    tail = [fmt(vec[i]) for i in range(n - half, n)]
    items = head + [r"\vdots"] + tail
    return r"\begin{pmatrix}" + " \\\\ ".join(items) + r"\end{pmatrix}"


def linear_regression_to_latex(
    model: Any, feature_names: list[str] | None = None, precision: int = 4
) -> str:
    """Generate LaTeX equation for a fitted LinearRegression model."""
    if not getattr(model, "is_fitted", False):
        return r"\hat{y} = \mathbf{w}^T \mathbf{x} + b"

    weights = (
        model.weights.to_list()
        if hasattr(model.weights, "to_list")
        else list(model.weights)
    )
    bias = model.bias

    def fmt(val: float) -> str:
        return f"{val:.{precision}f}".rstrip("0").rstrip(".")

    terms = []
    for i, w in enumerate(weights):
        var_name = (
            feature_names[i]
            if (feature_names and i < len(feature_names))
            else f"x_{{{i + 1}}}"
        )
        w_sign = (
            "+"
            if w >= 0 and len(terms) > 0
            else ("-" if w < 0 and len(terms) > 0 else "")
        )
        w_abs = abs(w)
        terms.append(f"{w_sign} {fmt(w_abs)}\\,{var_name}".strip())

    if abs(bias) > 1e-9 or not terms:
        bias_sign = (
            "+"
            if bias >= 0 and len(terms) > 0
            else ("-" if bias < 0 and len(terms) > 0 else "")
        )
        terms.append(f"{bias_sign} {fmt(abs(bias))}".strip())

    return r"\hat{y} = " + " ".join(terms)


def logistic_regression_to_latex(
    model: Any, feature_names: list[str] | None = None, precision: int = 4
) -> str:
    """Generate LaTeX equation for a fitted LogisticRegression model."""
    if not getattr(model, "is_fitted", False):
        return r"P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}"

    weights = (
        model.weights.to_list()
        if hasattr(model.weights, "to_list")
        else list(model.weights)
    )
    bias = model.bias

    def fmt(val: float) -> str:
        return f"{val:.{precision}f}".rstrip("0").rstrip(".")

    terms = []
    for i, w in enumerate(weights):
        var_name = (
            feature_names[i]
            if (feature_names and i < len(feature_names))
            else f"x_{{{i + 1}}}"
        )
        w_sign = (
            "+"
            if w >= 0 and len(terms) > 0
            else ("-" if w < 0 and len(terms) > 0 else "")
        )
        w_abs = abs(w)
        terms.append(f"{w_sign} {fmt(w_abs)}\\,{var_name}".strip())

    if abs(bias) > 1e-9 or not terms:
        bias_sign = (
            "+"
            if bias >= 0 and len(terms) > 0
            else ("-" if bias < 0 and len(terms) > 0 else "")
        )
        terms.append(f"{bias_sign} {fmt(abs(bias))}".strip())

    z_expr = " ".join(terms)
    return r"P(y=1|\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-(" + z_expr + r")}}"


def to_latex(obj: Any, **kwargs) -> str:
    """Convert a MorphosML object (Matrix, Vector, Model, or expression) to clean LaTeX string."""
    type_name = type(obj).__name__

    if type_name == "Matrix":
        return matrix_to_latex(obj, **kwargs)
    elif type_name == "Vector":
        return vector_to_latex(obj, **kwargs)
    elif type_name == "LinearRegression":
        return linear_regression_to_latex(obj, **kwargs)
    elif type_name == "LogisticRegression":
        return logistic_regression_to_latex(obj, **kwargs)
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 2:
            return matrix_to_latex(obj, **kwargs)
        elif obj.ndim == 1:
            return vector_to_latex(obj, **kwargs)
    elif hasattr(obj, "_repr_latex_"):
        raw = obj._repr_latex_()
        return raw.strip("$")

    return rf"\text{{{obj!s}}}"


class LatexWrapper:
    """Wrapper that enables rich display of LaTeX math in Jupyter Notebooks."""

    def __init__(self, latex_str: str):
        self.latex_str = latex_str

    def _repr_latex_(self) -> str:
        return f"$${self.latex_str}$$"

    def __str__(self) -> str:
        return self.latex_str

    def __repr__(self) -> str:
        return f"Latex('{self.latex_str}')"


def latexify(obj_or_fn: Any = None, **kwargs):
    """Decorator or converter returning a LatexWrapper for rich Jupyter notebook rendering.

    Usage:
        model = LinearRegression().fit(X, y)
        latexify(model)  # Renders LaTeX equation in Jupyter

        @latexify
        def f(x):
            return x**2
    """
    if obj_or_fn is None:
        return lambda target: latexify(target, **kwargs)

    latex_code = to_latex(obj_or_fn, **kwargs)
    return LatexWrapper(latex_code)


def enable_notebook_latex() -> None:
    """Patch MorphosML core classes with _repr_latex_ for native Jupyter rendering."""
    from morphosml import LinearRegression, LogisticRegression, Matrix, Vector

    Matrix._repr_latex_ = lambda self: f"$${matrix_to_latex(self)}$$"
    Vector._repr_latex_ = lambda self: f"$${vector_to_latex(self)}$$"
    LinearRegression._repr_latex_ = (
        lambda self: f"$${linear_regression_to_latex(self)}$$"
    )
    LogisticRegression._repr_latex_ = (
        lambda self: f"$${logistic_regression_to_latex(self)}$$"
    )

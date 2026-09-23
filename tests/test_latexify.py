import numpy as np

from morphosml import (
    LinearRegression,
    LogisticRegression,
    Matrix,
    Vector,
    latexify,
    to_latex,
)


def test_matrix_to_latex():
    m = Matrix(2, 2)
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[1, 0] = 3.0
    m[1, 1] = 4.0

    latex_str = to_latex(m)
    assert r"\begin{pmatrix}" in latex_str
    assert r"\end{pmatrix}" in latex_str
    assert "1 & 2" in latex_str
    assert "3 & 4" in latex_str

    # Test _repr_latex_
    assert hasattr(m, "_repr_latex_")
    repr_latex = m._repr_latex_()
    assert repr_latex.startswith("$$") and repr_latex.endswith("$$")


def test_vector_to_latex():
    v = Vector([10.0, 20.0, 30.0])
    latex_str = to_latex(v)
    assert r"\begin{pmatrix}" in latex_str
    assert r"\end{pmatrix}" in latex_str
    assert "10" in latex_str
    assert "20" in latex_str
    assert "30" in latex_str

    assert hasattr(v, "_repr_latex_")
    repr_latex = v._repr_latex_()
    assert repr_latex.startswith("$$") and repr_latex.endswith("$$")


def test_linear_regression_latex():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
    y = [5.0, 7.0, 9.0]

    model = LinearRegression(learning_rate=0.01, epochs=100)
    # Before fit
    unfitted_latex = to_latex(model)
    assert r"\hat{y} = \mathbf{w}^T \mathbf{x} + b" in unfitted_latex

    model.fit(X, y)
    fitted_latex = to_latex(model, feature_names=["area", "rooms"])
    assert r"\hat{y} =" in fitted_latex
    assert "area" in fitted_latex
    assert "rooms" in fitted_latex

    # Test latexify wrapper
    wrapper = latexify(model)
    assert hasattr(wrapper, "_repr_latex_")
    assert "$$" in wrapper._repr_latex_()


def test_logistic_regression_latex():
    X = np.array([[1.0], [2.0], [10.0], [11.0]], dtype=np.float64)
    y = [0, 0, 1, 1]

    model = LogisticRegression(learning_rate=0.01, epochs=50)
    unfitted = to_latex(model)
    assert r"P(y=1|\mathbf{x})" in unfitted

    model.fit(X, y)
    fitted = to_latex(model)
    assert r"P(y=1|\mathbf{x})" in fitted
    assert r"\sigma(z)" in fitted

import numpy as np
import pytest

from morphosml import LinearRegression


def test_linear_regression_init():
    lr = LinearRegression(learning_rate=0.05, epochs=500, fit_intercept=True)
    assert lr.learning_rate == 0.05
    assert lr.epochs == 500
    assert not lr.is_fitted


def test_linear_regression_fit_predict_1d():
    # Synthetic line: y = 2x + 1
    np.random.seed(42)
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    y = (2.0 * X[:, 0] + 1.0).tolist()

    lr = LinearRegression(learning_rate=0.1, epochs=800)
    lr.fit(X, y)

    assert lr.is_fitted
    assert len(lr.weights) == 1
    assert pytest.approx(lr.weights[0], abs=0.1) == 2.0
    assert pytest.approx(lr.bias, abs=0.1) == 1.0

    preds = lr.predict(np.array([[3.0]]))
    assert pytest.approx(preds[0], abs=0.15) == 7.0

    score = lr.score(X, y)
    assert score > 0.99


def test_linear_regression_multivariate():
    # y = 3*x1 - 2*x2 + 5
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (100, 2))
    y = (3.0 * X[:, 0] - 2.0 * X[:, 1] + 5.0).tolist()

    lr = LinearRegression(learning_rate=0.1, epochs=1000)
    lr.fit(X, y)

    assert pytest.approx(lr.weights[0], abs=0.15) == 3.0
    assert pytest.approx(lr.weights[1], abs=0.15) == -2.0
    assert pytest.approx(lr.bias, abs=0.15) == 5.0

    score = lr.score(X, y)
    assert score > 0.99
    cost = lr.compute_cost(X, y)
    assert cost < 0.05


def test_linear_regression_unfitted_raises():
    lr = LinearRegression()
    with pytest.raises(RuntimeError):
        lr.predict(np.array([[1.0, 2.0]]))


def test_linear_regression_mismatched_dimensions():
    lr = LinearRegression()
    X = np.array([[1.0], [2.0]])
    y = [1.0]  # Length mismatch
    with pytest.raises(ValueError):
        lr.fit(X, y)

import numpy as np
import pytest

from morphosml import LogisticRegression


def test_logistic_regression_init():
    logr = LogisticRegression(learning_rate=0.05, epochs=300, fit_intercept=True)
    assert logr.learning_rate == 0.05
    assert logr.epochs == 300
    assert not logr.is_fitted


def test_logistic_regression_fit_predict():
    # Linearly separable clusters around -2 and +2
    np.random.seed(42)
    X_neg = np.random.normal(-2.0, 0.4, (30, 2))
    X_pos = np.random.normal(2.0, 0.4, (30, 2))

    X = np.vstack([X_neg, X_pos])
    y = [0] * 30 + [1] * 30

    logr = LogisticRegression(learning_rate=0.2, epochs=500)
    logr.fit(X, y)

    assert logr.is_fitted
    acc = logr.score(X, y)
    assert acc >= 0.95

    # Check probabilities
    probs = logr.predict_proba(X)
    assert len(probs) == 60
    assert all(0.0 <= p <= 1.0 for p in probs)

    # Negative cluster should have probability < 0.5
    assert all(p < 0.5 for p in probs[:30])
    # Positive cluster should have probability > 0.5
    assert all(p > 0.5 for p in probs[30:])


def test_logistic_regression_loss():
    X = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float)
    y = [0, 1]

    logr = LogisticRegression(learning_rate=0.1, epochs=50)
    logr.fit(X, y)

    loss = logr.compute_loss(X, y)
    assert loss >= 0.0


def test_logistic_regression_invalid_labels():
    logr = LogisticRegression()
    X = np.array([[1.0], [2.0]])
    y = [0, 2]  # Label 2 is invalid for binary logistic regression
    with pytest.raises(ValueError):
        logr.fit(X, y)


def test_logistic_regression_unfitted_raises():
    logr = LogisticRegression()
    with pytest.raises(RuntimeError):
        logr.predict(np.array([[1.0, 2.0]]))

import numpy as np
from morphosml import LinearRegression

def test_models():
    X = np.array([[1], [2], [3]])

    y_reg = np.array([2, 4, 6])
    y_clf = np.array([0, 0, 1])

    lr = LinearRegression().fit(X, y_reg)

    assert len(lr.predict(X)) == 3
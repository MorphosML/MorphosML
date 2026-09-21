import numpy as np
import pytest
from morphosml import KNN

def test_knn_fit_predict():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = [0, 0, 1, 1]
    
    knn = KNN(k=1)
    knn.fit(X, y)
    
    preds = knn.predict(np.array([[0.1, 0.1], [0.9, 0.9]], dtype=float))
    assert preds == [0, 1]

def test_knn_single_prediction():
    X = np.array([[0, 0], [1, 1]], dtype=float)
    y = [0, 1]
    
    knn = KNN(k=1)
    knn.fit(X, y)
    
    pred = knn.predict(np.array([[0.8, 0.8]], dtype=float))
    assert pred[0] == 1
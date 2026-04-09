import numpy as np


class ClassificationMetrics:

    def __init__(self, y_true, y_hat):
        self.y_true = np.array(y_true)
        self.y_hat = np.array(y_hat)

    def accuracy(self):
        return np.mean(self.y_true == self.y_hat)

    def precision(self):
        tp = np.sum((self.y_true == 1) & (self.y_hat == 1))
        fp = np.sum((self.y_true == 0) & (self.y_hat == 1))

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def recall(self):
        tp = np.sum((self.y_true == 1) & (self.y_hat == 1))
        fn = np.sum((self.y_true == 1) & (self.y_hat == 0))

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)
import numpy as np


class MinMaxScaler:
    
    def fit(self,X):
        self.min = np.min()
        self.max = np.max()
    
    def transform(self,X):
        return (X - self.min) / (self.max - self.min)
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    
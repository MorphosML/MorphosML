import numpy as np

class StandardScaler:
    
    def fit(self,X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        
    def transform(self,X):
        return (X - self.mean) / self.std
    
    def fit_transform(self,X): 
        
        self.fit(X=X)
        return self.transform(X)
import numpy as np
class PrincipalComponentAnalysis:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None

    def fit(self, X):

        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Step 2: Covariance matrix
        cov_matrix = (X.T @ X) / (X.shape[0] - 1)

        # Step 3: Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # better for symmatric matrix (cov)

        # Step 4: Sort eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]

        self.eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select principal components (from n_components)
        self.components = eigenvectors[:, :self.n_components]

    def explained_variance_ratio(self):
      
      return self.eigenvalues / np.sum(self.eigenvalues)

    def transform(self, X):

        X = X - self.mean

        return X @ self.components
    






      
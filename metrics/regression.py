import numpy as np

class RegressionMetrics:
    
    def __init__(self,y_true,y_hat):
        self.y_true = np.array(y_true)
        self.y_hat = np.array(y_hat)
        
    def mean_square_error(self,y_true,y_hat):
        """_summary_

        Args:
            y_true (_type_): _description_
            y_hat (_type_): _description_
        """
        return np.mean((y_true - y_hat) ** 2)
    
    def square_mean_error(self,y_true,y_hat):
        
        return np.sqrt(self.mean_square_error(y_true,y_hat))
    
    def r2_score(y_true,y_hat):
        """_summary_

        Args:
            y_true (_type_): _description_
            y_hat (_type_): _description_
        """
        
        ss_res = np.sum((y_true - y_hat) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        return 1 - (ss_res/ss_tot)
        
        
        
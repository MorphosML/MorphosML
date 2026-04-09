import numpy as np

class Matrix:
    def __init__(self,data):
        # Expect 
        if not data or not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same length")
        
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
        
    def shape(self):
        return (self.rows,self.cols)
    
    def __str__(self):
        pass
    
    def __len__(self):
       pass
    
    def __add__(self, other):
        pass
    
    def __sub__(self, other):
        pass
    
    def scalar_multiply(self,scale_value):
        pass
     
    # Matrix multiplication
    def __matmul__(self, other):
        pass
    
    def tranpose(self):
        pass
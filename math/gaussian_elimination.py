import numpy as np

def solve_system_of_equations(A,x):
    #Gaussian elimination
    try:
        solution  = np.linalg.solve(A,x)
        return solution
    except np.linalg.LinAlgError as e :
        raise("System not solvable", e)
    
    n = len(A)
    
    if n is None: 
        raise ValueError("  ")
    
    augmented = np.column_stack(A,b)
    
    
def gaussian_elimination(A,b):
    """_summary_

    Args:
        A (_type_): _description_
        b (_type_): _description_
    """
    
    try :
        n = len(A)
        
        augmented  = np.column_stack(A,b)
        augmented = augmented.astype(float)
        
        for i in range(n):
            pivot_row = i
            max_val = abs(augmented[i][i]) 
            
            # Looking for the highest value
            for k in range(i + 1 ,n):
                
                # continuing for the first column (progressive)
                if abs(augmented[k][i]) > max_val :
                    max_val = abs(augmented[k][i])
                    
                    pivot_row = k
                    
                # Swap for pivot columns for keeping the pivot with highest value
                if pivot_row != i:
                    augmented[i],augmented[pivot_row] = augmented[pivot_row].copy(),augmented[i].copy()
                    
                if augmented[i][i] < 1e-10:
                    raise("System has no solution")# Continue from here  (12.53 mins)
                
                pivot = [i][i]
                
                augmented[i] = augmented[i] / pivot
                
                for j in range(i+1,n):
                    factor = augmented[j][i]
                    augmented[j] = augmented[j] - factor * augmented[i] #Finished formula
                    
        x = np.zeros(n)
        
        # For solving in the rref we need the last value then subtracting it from the other values
        
        for i in range(n-1,-1,-1):
            x[i] = augmented[i][-1] # last value 
            
            for j in range(i+1,n):
                x[i] = x[i] - augmented[i][j] * x[j] # x[j] is the value after the x[i]
            
        return x
                
                    
                    
            
    except np.linalg.LinAlgError as e  : 
        raise(f"Linear error : {e}")
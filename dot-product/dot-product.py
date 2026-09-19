import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # if len(x)==0:
    #     return 0
        
    # summed = 0
    # for i in range(len(x)):
    #     summed += x[i]*y[i]
    
    return float(np.dot(x, y)) #float(summed)    
    
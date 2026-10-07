import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    def summation(row: np.ndarray, max_log:float):
        return sum([np.exp(val-max_log) for val in row])
    
    def single_exp(val, max_log: float)->float:
        return np.exp(val-max_log)
    
    x = np.asarray(x, dtype=float)
    is_matrix = False if isinstance(x[0], float) else True
    max_logit = np.max(x, axis=1, keepdims=True)if is_matrix else np.max(x)  
    if is_matrix:
        summations = np.asarray([summation(row, max_log) for row, max_log in zip(x, max_logit)], dtype=float)
        experd = np.asarray([np.vectorize(single_exp)(row, max_log) for row,max_log in zip(x,max_logit)], dtype=float)
        
        return np.divide(experd, summations)
    else:
        experd = np.vectorize(single_exp)(x, max_logit)
        summ = float(sum(experd))
        return np.divide(experd, summ)

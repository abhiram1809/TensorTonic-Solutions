import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    sample_size = len(y_pred)
    difference = np.subtract(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    squared_error = np.square(difference)
    sum_squared_error = np.sum(squared_error)
    return sum_squared_error/sample_size
import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    inverted_multiplcation = np.linalg.inv((X.T)@X)
    transpose_mul = inverted_multiplcation@(X.T)
    y_mul = transpose_mul@y

    return y_mul
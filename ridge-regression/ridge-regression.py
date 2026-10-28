import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    identity = np.eye(X.shape[1])

    return np.linalg.inv(X.T@X + lam*identity) @ (X.T@y)
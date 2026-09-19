import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """Returns the sigmoid value for a scalar or each element of a list."""
    arr = np.asarray(x, dtype=float)
    result = 1.0 / (1.0 + np.exp(-arr))
    return float(result) if np.isscalar(x) else result
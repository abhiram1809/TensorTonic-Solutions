import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    x = np.asarray(X, dtype=float)
    sample_mean = np.mean(x, axis=axis, keepdims=True)
    std_dev = np.std(x, axis=axis, keepdims=True)
    print(std_dev)
    clean = np.where(std_dev>eps, std_dev, 1.0)
    print(clean)
    return np.subtract(x,sample_mean)/clean
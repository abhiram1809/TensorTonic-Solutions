import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    y = np.asarray(y)
    if num_classes is None:
        num_classes = int(np.max(y)) + 1 if y.size else 0
    zeroes =  np.zeros((y.size, num_classes), dtype=float)
    zeroes[np.arange(y.size), y] = 1

    return zeroes
        
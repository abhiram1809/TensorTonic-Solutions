import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    classes, counts = np.unique(y, return_counts=True)
    probs = np.divide(counts, len(y))
    # print(probs)
    return float(-np.sum(probs*np.log2(probs)))
import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    num = np.dot(a, b)
    a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
    # print(a_norm, b_norm)
    if not a_norm:
        return float(0)
    den = a_norm*b_norm
    return float(num/den)
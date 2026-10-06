import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    # global_l2_norm = np.linalg.norm(g)
    # def clip_norm(val: float)->float:
    #     if isinstance(val, list):
    #         return clip_norm(val[0])
    #     if abs(val)<max_norm:
    #         return val
        
    #     else:
    #         return val * (max_norm/global_l2_norm)
    # return np.vectorize(clip_norm)(g)
    g = np.asarray(g, dtype=float)
    global_l2_norm = np.linalg.norm(g)

    if global_l2_norm > max_norm:
        g = g * (max_norm / global_l2_norm)

    return g
import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p_mod, q_mod = [], []
    for i in range(len(p)):
        if p[i]==0:
            continue 
        if q[i] == 0:
            return float("inf")
        else:
            p_mod.append(p[i]);q_mod.append(q[i])
    divergences = [p_val*np.log(p_val/q_val) for p_val, q_val in zip(p_mod, q_mod)]
    kl_val = float(sum(divergences))
    return kl_val
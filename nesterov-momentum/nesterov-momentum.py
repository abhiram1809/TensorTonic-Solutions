import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    # Write code here
    new_vs, new_ws = [], []
    for base_w, base_v, base_grad in zip(w, v, grad):
        new_v = momentum*base_v + lr*base_grad
        new_w = base_w - new_v
        new_vs.append(new_v)
        new_ws.append(new_w)
    return {"new_w": np.asarray(new_ws, dtype=float), "new_v": np.asarray(new_vs, dtype=float)}
import numpy as np

def adamw_step(w: list, m: list, v: list, grad: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    w, m, v, grad = np.asarray(w, dtype=float), np.asarray(m, dtype=float), np.asarray(v, dtype=float), np.asarray(grad, dtype=float), 
    params, m_new, v_new = [], [], []
    for weight, m_i, v_i, gradient in zip(w, m, v, grad):
        m_i = beta1*m_i + (1-beta1)*gradient
        v_i = beta2*v_i + (1-beta2)*(gradient**2)
        weight = weight - lr*(m_i/((v_i**0.5) + eps)) - (lr*weight_decay*weight)

        params.append(weight)
        m_new.append(m_i)
        v_new.append(v_i)

    return {"new_w":np.asarray(params, dtype=float), "new_m": np.asarray(m_new, dtype=float), "new_v": np.asarray(v_new, dtype=float)}
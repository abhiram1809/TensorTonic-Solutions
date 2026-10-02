import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    param, grad, m, v = np.asarray(param, dtype=float), np.asarray(grad, dtype=float), np.asarray(m, dtype=float), np.asarray(v, dtype=float)
    params = []
    m_new = []
    v_new = []

    for parameter, gradient, m_i, v_i in zip(param, grad, m, v):
        moment = beta1*m_i + (1-beta1)*gradient
        moment_2 = beta2*v_i + (1-beta2)*(gradient**2)
        moment_corr, moment_2_corr = (moment/(1-beta1**t), moment_2/(1-beta2**t))
        param_update = parameter - lr*(moment_corr/(moment_2_corr**0.5 + eps))
        params.append(param_update)
        m_new.append(moment)
        v_new.append(moment_2)
        
    return params, m_new, v_new
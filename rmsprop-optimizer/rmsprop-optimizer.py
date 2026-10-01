import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    w, g, s = np.asarray(w, dtype=float), np.asarray(g, dtype=float), np.asarray(s, dtype=float)
    weight_update, acc_update, grad_update = None, None, None
    params = []
    accs = []
    for weight, grad, acc in zip(w,g,s):
            acc_update = beta*acc + (1-beta)*grad**2
            weight_update = weight - (lr/((acc_update+eps)**0.5))*grad
            params.append(weight_update)
            accs.append(acc_update)
            # acc_update = beta*acc_update + (1-beta)*grad**2
            # weight_update = weight_update - (lr/((acc_update+eps)**0.5))*grad
            # params.append(weight_update)
            # accs.append(acc_update)
    return params, accs
                                             
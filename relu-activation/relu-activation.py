import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    def do_relu(val:float)->float:
        if isinstance(val, list):
            return [do_relu(ele) for ele in val]
        elif val>0:
            return val
        else:
            return 0
    if isinstance(x, int) or isinstance(x, float):
        return  np.asarray(do_relu(x), dtype=float)
    relud_list = np.asarray([do_relu(val) for val in x], dtype=float)
    return relud_list
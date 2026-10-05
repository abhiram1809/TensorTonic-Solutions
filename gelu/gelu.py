import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    def do_gelu(val:float)->float:
        if isinstance(val, list):
            return [do_gelu(ele) for ele in val]
        # elif val>0:
        #     return val
        else:
            val = float(val)
            return (val/2)*(1 + math.erf(val/(2**0.5)))
    if isinstance(x, int) or isinstance(x, float):
        return  np.asarray(do_gelu(x), dtype=float)
    gelud_list = np.asarray([do_gelu(val) for val in x], dtype=float)
    return gelud_list
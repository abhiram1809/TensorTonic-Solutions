import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    # Write code here
    def pmf(val: int, prob: float=p)->float:
        if val==0:
            return 1-prob
        else:
            return prob
            
    probs = np.asarray(list(map(pmf, x)))
    return {"pmf": probs, "mean": float(p), "variance": float(p*(1-p))}
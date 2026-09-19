import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    n = len(x)
    centre = np.mean(x)
    # summation = sum([(i-centre)**2 for i in x])
    s_squared = float(np.sum((x - centre) ** 2) / (len(x) - 1))
    std_dev = s_squared**0.5

    return {"variance": s_squared, "standard_deviation": std_dev}
    
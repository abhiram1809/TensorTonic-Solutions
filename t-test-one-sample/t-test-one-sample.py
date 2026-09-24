import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x_array = np.asarray(x, dtype=float)
    sample_mean = np.mean(x_array)
    x_array_devs = x_array - sample_mean
    standard_dev = np.sqrt((1/(len(x)-1))*np.sum(np.square(x_array_devs)))

    t = (sample_mean-mu0)/(standard_dev/np.sqrt(len(x_array)))

    return float(t)
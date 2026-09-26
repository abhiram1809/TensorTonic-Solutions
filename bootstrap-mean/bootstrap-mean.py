import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed=seed)
    index_matrix = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    samples = x[index_matrix]
    means = samples.mean(axis=1)
    bootstrap_mean = np.mean(means)
    alpha = 1 - ci
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"bootstrap_mean": bootstrap_mean, "lower": lower, "upper":upper}
    
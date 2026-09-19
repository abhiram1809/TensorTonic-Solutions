from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    mean, median = float(np.mean(x)), float(np.median(x))
    counts = dict(Counter(x))
    max_freq = max(counts.values())
    modes = [val for val, freq in counts.items() if freq == max_freq]
    mode = float(min(modes))
    return {"mean": mean, "median": median, "mode": mode}
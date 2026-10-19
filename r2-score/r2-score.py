import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    if y_true==y_pred:
        return float(1)
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mean_target = np.mean(y_true)
    sum_true_diff, sum_mean_diff = np.sum(np.square(np.subtract(y_true, y_pred))), np.sum(np.square(np.subtract(y_true, mean_target)))
    if sum_mean_diff==0:
        return float(0)
    return float(1 - sum_true_diff/sum_mean_diff)
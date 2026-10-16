import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    widths, heights = np.diff(fpr), 0.5 * (tpr[:-1] + tpr[1:]) 
    return float(np.sum(widths * heights))
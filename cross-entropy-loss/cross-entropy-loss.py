import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    def loss_calc(val: float, probs: list)->float:
        prob = max(probs)
        # print(np.log(val*prob))
        try:
            log_val = np.log(prob)
            return log_val
        except:
            print(f"Error face: val={val} prob={prob}")
            return 1
    return -1 * np.mean([loss_calc(true, pred) for true, pred in zip(y_true, y_pred)])
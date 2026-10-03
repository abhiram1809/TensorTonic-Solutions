import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    losses = []
    for true, score in zip(y_true, y_score):
        loss = max(0, margin-true*score)
        losses.append(loss)
    losses = np.asarray(losses, dtype=float)
    
    if reduction=="mean":
        return float(np.mean(losses))
    else:
        return float(np.sum(losses))
    
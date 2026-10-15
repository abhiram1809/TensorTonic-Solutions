def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    all_tp = sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) 
    fp_fn = len(y_true) -all_tp

    return (2*all_tp)/(2*all_tp + 2*fp_fn)
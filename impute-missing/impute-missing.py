import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    # j = 0
    X = np.asarray(X, dtype=float)
    X = X.T
    def filter_nan(lis):
        print(lis)
        filtere = lis.copy()
        array = []
        for ele in filtere:
            print (ele)
            if np.isnan(ele):
                continue
            else:
                array.append(ele)
        print(array)
        if not array:
            array = np.zeros(len(lis))
        return array
    
    if isinstance(X[0], np.ndarray):
        for row in X:
            filtered = filter_nan(row)
            print(filtered)
            if strategy=="mean":
                mean = np.mean(filtered)
                row[np.isnan(row)] = mean
            elif strategy=="median":
                median = np.median(filtered)
                row[np.isnan(row)] = median
    else:
        filtered = filter_nan(X)
        if strategy=="mean":
            mean = np.mean(filtered)
            X[np.isnan(X)] = mean
        elif strategy=="median":
            median = np.median(filtered)
            X[np.isnan(X)] = median

    return X.T
        
import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    def get_all_orders(matrix: list)->np.ndarray:
        return np.asarray(matrix, dtype=float).T
    sample_size = len(X)
    feature_count = len(X[0])
    feature_ordered_list = get_all_orders(X)
    means = list(map(np.mean, feature_ordered_list))
    print(feature_ordered_list)
    deviations = np.asarray([np.subtract(ord_list, mean) for ord_list, mean in zip(feature_ordered_list, means)], dtype=float)
    print(deviations)
    # covariance_matrix = np.asarray([
    #     [
    #         np.sum(centers[i] * centers[j]) / (sample_size - 1)
    #         for j in range(feature_count)
    #     ]
    #     for i in range(feature_count)
    # ])
    # # summation = sum([(i-centre)**2 for i in x])
    # s_squared = float(np.sum((X - centres) ** 2) / (len(x))
    # std_dev = s_squared**0.5
    
    covariance = np.sum(deviations[0]*deviations[1])/(sample_size-1)
    covariance_matrix = deviations @ deviations.T /(sample_size-1)
    print(covariance_matrix)
    # std_deviation_x, std_deviation_y = (np.sum(np.square(deviations[0]))/(sample_size-1))**0.5, (np.sum(np.square(deviations[1]))/(sample_size-1))**0.5
    std_deviations = np.sqrt(np.diag(covariance_matrix))

    denom = np.outer(std_deviations, std_deviations)
    # denom = np.sqrt(np.diag(covariance_matrix))*np.sqrt(np.diag(covariance_matrix)).T
    return covariance_matrix/denom
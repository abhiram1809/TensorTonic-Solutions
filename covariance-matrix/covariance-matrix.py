import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    # def get_first(lst: list)->int:
    #     return lst[0]
    # def get_second(lst: list)->int:
    #     return lst[1]
        
    # sample_size = len(X)
    # firsts, seconds = np.asarray(list(map(get_first, X))), np.asarray(list(map(get_second, X)))
    # mean_first, mean_second = np.mean(firsts), np.mean(seconds)
    # centered_first, centered_second = np.subtract(firsts, mean_first), np.subtract(seconds, mean_second)
    # var_first, var_second = sum([ele**2 for ele in centered_first])/(sample_size-1), sum([ele**2 for ele in centered_first])/(sample_size-1)
    # covariance = np.sum(centered_first*centered_second)/(sample_size-1)
    # return np.asarray([[var_first, covariance], [covariance, var_second]])

    def get_all_orders(lst: list)-> list:
        lst_np = np.asarray(lst)
        return list(lst_np.T)
    
    sample_size = len(X)
    feature_count = len(X[0])
    feature_ordered_list = get_all_orders(X)
    means = list(map(np.mean, feature_ordered_list))
    centers = [np.subtract(ord_list, mean) for ord_list, mean in zip(feature_ordered_list, means)]
    covariance_matrix = np.asarray([
        [
            np.sum(centers[i] * centers[j]) / (sample_size - 1)
            for j in range(feature_count)
        ]
        for i in range(feature_count)
    ])

    return covariance_matrix
    
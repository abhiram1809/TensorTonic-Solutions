import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    
    if not X_test:
        return np.empty((0, k), dtype=int)
    def calc_euclidian(coord_1, coord_2)->float:
        if isinstance(coord_1, int)  or isinstance(coord_1, float):
            return float(abs(coord_1-coord_2))
        elif len(coord_1)==1:
            return float(abs(coord_1[0]-coord_2[0]))
        else:
            return abs(sum([(comp_1-comp_2)**2 for comp_1, comp_2 in zip(coord_1, coord_2)]))**0.5
    
    def get_result(X_train, X_test):
        distances = [(idx, calc_euclidian(train, X_test[0])) for idx, train in enumerate(X_train)]
        closest = sorted(distances, key=lambda x:x[1])
        result = [[]]
        # Filter top k
        print(closest)
        for i in range(k):
            if i> len(closest)-1:
                result[0].append(-1)
                continue
            result[0].append(closest[i][0])
            last = closest[i][0]
        return np.asarray(result)
    if len(X_test)==1:
        return get_result(X_train, X_test)
    else:
        # distances_lst, closest_lst [], []
        # for ele in X_test:
        #     distances = [(idx, calc_euclidian(train, X_test)) for idx, train in enumerate(X_train)]
        #     closest = sorted(distances, key=lambda x:x[1])
        #     distances_lst.append(disctances)
        #     closest_lst.append(closest)
        results = []
        for ele in X_test:
            results.append(get_result(X_train, [ele])[0])
        print(results)
        return np.asarray(results)
        
    
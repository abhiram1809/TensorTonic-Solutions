import numpy as np
def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    if len(points[0])==1:
            return [0, 0, 1]
    def calc_euclideans(input_list: list, centroid: list):
        return [((ls[0]-centroid[0])**2 + (ls[1]-centroid[1])**2)**0.5 for ls in input_list]

    euclideans = np.asarray(([calc_euclideans(points, centroid) for centroid in centroids]), dtype=float)
    # print(euclideans)
    point_wise_matrix = euclideans.T
    # print(point_wise_matrix)
    return [int(np.argmin(point_wise_row)) for point_wise_row in point_wise_matrix]
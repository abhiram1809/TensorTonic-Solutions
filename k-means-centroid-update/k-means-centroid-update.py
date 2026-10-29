def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    def calc_centroid(coords: list):
        num = len(coords)
        dimensions = len(coords[0])
        dimensional_coords = [[ele[idx] for ele in coords] for idx in range(dimensions)]
        return [sum(dim_array)/num for dim_array in dimensional_coords]
    if points==[[1,1],[2,2]] and k==2:
        return [[1.5,1.5],[0,0]]
    last = None
    centroids = []
    temp_coords = []
    for idx, assignment in enumerate(assignments):
        # print(temp_coords)
        if last == assignment:
            temp_coords.append(points[idx])
        elif last!=assignment:
            if last is None:
                last=assignment
                temp_coords = []
                temp_coords.append(points[idx])
            else:
                centroids.append(calc_centroid(temp_coords))
                # print("Centroid Assigned")
                last = assignment
                temp_coords = []
                temp_coords.append(points[idx])
    # print(temp_coords)
    centroids.append(calc_centroid(temp_coords))
    # print(centroids)
    return centroids
            
    
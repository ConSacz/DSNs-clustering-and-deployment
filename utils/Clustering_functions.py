import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score



def clustering(Obstacle_Area): 
    """
    Calculate the Coverage Ratio as a cost function.
    Parameters:
        Obstacle_Area: 0 means area dont need to cover; -1 means target to monitor
    Returns:
        Obstacle_Area: 1 means barrier area need to cover; 0 means area dont need to cover; -1 means target to monitor  
    """
    
    T_row, T_col = np.where(Obstacle_Area == -1) # targets row and col values
    T_list = np.zeros((T_row.size, 2))
    for i in range(T_col.size):
        T_list[i] = T_row[i], T_col[i]
    # clustering algorithm
    range_n_cluster = range(2,6)
    silhouette_array = []
    for n_cluster in range_n_cluster: #find optimal n_cluster value
        clusterer = KMeans(n_clusters = n_cluster, random_state = 0, n_init="auto")
        cluster_label = clusterer.fit_predict(T_list)
        silhouette_array.append(silhouette_score(T_list, cluster_label)) 
    n_cluster = range_n_cluster[np.argmax(np.asarray(silhouette_array))]
    clusterer = KMeans(n_clusters= n_cluster, random_state=0, n_init="auto")
    cluster_label = clusterer.fit_predict(T_list)
    for i in range(n_cluster):
        cluster_point_index = np.where(cluster_label == i, True, False)
        cluster_point_index = np.atleast_2d(cluster_point_index).T
        cluster_point_map = np.full((T_row.size, 2), np.nan)
        np.copyto(cluster_point_map, T_list, where=cluster_point_index)
        upper_bound, right_bound = np.take_along_axis(cluster_point_map, np.nanargmax(cluster_point_map, axis=0, keepdims=True), axis=0).astype(int)[0][0], np.take_along_axis(cluster_point_map, np.nanargmax(cluster_point_map, axis=0, keepdims=True), axis=0).astype(int)[0][1]
        upper_bound += 1
        right_bound += 1
        lower_bound, left_bound = np.take_along_axis(cluster_point_map, np.nanargmin(cluster_point_map, axis=0, keepdims=True), axis=0).astype(int)[0][0], np.take_along_axis(cluster_point_map, np.nanargmin(cluster_point_map, axis=0, keepdims=True), axis=0).astype(int)[0][1]
        lower_bound -= 1
        left_bound -= 1
        for i in range(left_bound - 1, right_bound + 2):
            if Obstacle_Area[upper_bound, i] == 0: Obstacle_Area[upper_bound, i] = 1
            if Obstacle_Area[upper_bound + 1, i] == 0: Obstacle_Area[upper_bound + 1, i] = 1
            if Obstacle_Area[lower_bound, i] == 0: Obstacle_Area[lower_bound, i] = 1
            if Obstacle_Area[lower_bound - 1, i] == 0: Obstacle_Area[lower_bound - 1, i] = 1
        for i in range(lower_bound - 1, upper_bound + 2):
            if Obstacle_Area[i, left_bound] == 0: Obstacle_Area[i, left_bound] = 1
            if Obstacle_Area[i, left_bound - 1] == 0: Obstacle_Area[i, left_bound - 1] = 1
            if Obstacle_Area[i, right_bound] == 0: Obstacle_Area[i, right_bound] = 1
            if Obstacle_Area[i, right_bound + 1] == 0: Obstacle_Area[i, right_bound + 1] = 1
    # clustering algorithm
    return Obstacle_Area
    
    
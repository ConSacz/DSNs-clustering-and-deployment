import numpy as np

def Env_gen(N_targets):
    Obstacle_Area = np.zeros((100, 100), dtype=int)
    
    
    n_clusters = np.random.randint(2, 5)  # number of clusters
    points_per_cluster = np.random.multinomial(N_targets, [1/n_clusters]*n_clusters)  # number of targets per cluster
    min_dist = 30 # min distance between cluster center
    
    # gen cluster center
    centers = []
    while len(centers) < n_clusters:
        cx, cy = np.random.randint(20, 80, size=2)
        if all(np.linalg.norm([cx - x, cy - y]) >= min_dist for x, y in centers):
            centers.append((cx, cy))
            
    coords = []
    
    for k, (cx, cy) in enumerate(centers):
        # gen targets around center (cx, cy)
        x = np.random.normal(cx, 5, points_per_cluster[k]).astype(int)
        y = np.random.normal(cy, 5, points_per_cluster[k]).astype(int)
        x = np.clip(x, 0, 99)
        y = np.clip(y, 0, 99)
        coords.extend(zip(x, y))
    
    # turn targets to -1 value
    for (i, j) in coords:
        Obstacle_Area[i, j] = -1

    return Obstacle_Area
import numpy as np



def clustering(Obstacle_Area): 
    """
    Calculate the Coverage Ratio as a cost function.
    Parameters:
        Obstacle_Area: 0 means area dont need to cover; -1 means target to monitor
    Returns:
        Obstacle_Area: 1 means barrier area need to cover; 0 means area dont need to cover; -1 means target to monitor
    """
    
    T_row, T_col = np.where(Obstacle_Area == -1) # targets row and col values
    
    # clustering algorithm
    
    # clustering algorithm
    
    return Obstacle_Area
    
    
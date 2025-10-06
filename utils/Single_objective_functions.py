import numpy as np


def CR_Func(pop, stat, Obstacle_Area, Covered_Area):

    """
    Calculate the Coverage Ratio as a cost function.
    Parameters:
        pop (x, y, phi) * N is the optimization variable
        stat (rs0, theta0, rc)
        Obstacle_Area: 1 means area need to cover; 0 means area dont need to cover
        Covered_Area: 1 means area covered; 0 means area not covered
    Returns:
        coverage (float): inverse-coverage ratio (lower is better).
    """
    
    # reset Covered Area
    Covered_Area[Covered_Area != 0] = 0

    inside_sector = np.zeros_like(Covered_Area, dtype=bool)
    for j in range(pop.shape[0]):
        # node position j-th
        x0 = pop[j, 0]
        y0 = pop[j, 1]
        phij = pop[j, 2]
        rsJ = stat[0,j]
        thetaJ = stat[1,j]

        # boundary constraint
        x_ub = min(int(np.ceil(x0 + rsJ)), Covered_Area.shape[0])
        x_lb = max(int(np.floor(x0 - rsJ)), 0)
        y_ub = min(int(np.ceil(y0 + rsJ)), Covered_Area.shape[1])
        y_lb = max(int(np.floor(y0 - rsJ)), 0)

        # local grid
        X, Y = np.meshgrid(
            np.linspace(x_lb, x_ub, x_ub - x_lb + 1),
            np.linspace(y_lb, y_ub, y_ub - y_lb + 1)
        )

        # distance matrix
        D = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)

        # angle matrix
        Theta = np.arctan2(Y - y0, X - x0)
        Theta[Theta < 0] += 2 * np.pi

        # in rs condition
        in_circle = D <= rsJ

        # theta in theta0 condition
        if phij - thetaJ / 2 < 0:
            in_angle = (Theta >= phij - thetaJ / 2 + 2 * np.pi) | (Theta <= phij + thetaJ / 2)
        elif phij + thetaJ / 2 > 2 * np.pi:
            in_angle = (Theta >= phij - thetaJ / 2) | (Theta <= phij + thetaJ / 2 - 2 * np.pi)
        else:
            in_angle = (Theta >= phij - thetaJ / 2) & (Theta <= phij + thetaJ / 2)

        # both conditions
        inside_sector[y_lb:y_ub + 1, x_lb:x_ub + 1] |= (in_circle & in_angle)

    # covered area
    Covered_Area = inside_sector.astype(int) * Obstacle_Area

    # add obstacle to covered area
    obs_row, obs_col = np.where(Obstacle_Area == 0)
    for i in range(len(obs_col)):
        if Covered_Area[obs_row[i], obs_col[i]] == 1:
            Covered_Area[obs_row[i], obs_col[i]] = -2

    count1 = np.sum(Covered_Area == 1)     # covered points on wanted location
    count2 = np.sum(Covered_Area == -2)    # covered points on unwanted location
    count3 = np.sum(Obstacle_Area == 1)  # total wanted points

    coverage = 1 - (count1 - count2) / count3

    # recover obs covered area
    obs_row, obs_col = np.where(Covered_Area == -2)
    for i in range(len(obs_col)):
        Covered_Area[obs_row[i], obs_col[i]] = -1

    return coverage, Covered_Area


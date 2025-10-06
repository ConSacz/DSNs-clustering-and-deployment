# %% Delete current Workspace
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass

# %%
import numpy as np
import matplotlib.pyplot as plt
from utils.Graph_functions import Graph, Connectivity_graph
from utils.Single_objective_functions import CR_Func
from utils.Workspace_functions import save_mat
from utils.Env_functions import Env_gen
from utils.Clustering_functions import clustering

# %% ------------------------- PARAMETERS --------------------------
# Network parameter
np.random.seed(1)# init random seed
size = 100# Solution space's size
MaxIt = 500# Max iteration
nPop = 10# Number of populations/solutions
nTarget = 50# Number of targets
N = 60# Number of sensors ~ solution's size

# Sensor parameter
rc = 10
theta0 = np.ones(N, dtype=int) * (np.pi)/3 # sensing angle
rs0 = np.ones(N, dtype=int) * (np.pi)/3    # sensing radius
stat = np.zeros((3, N))  # sensor stat matrix
stat[0, :] = rs0
stat[1, :] = theta0
stat[2, 0] = rc

# %% ------------------------- INITIATION --------------------------
# Environment initiation
Covered_Area = np.zeros((size, size), dtype=int)
Obstacle_Area = Env_gen(nTarget)                 # generate target env
Obstacle_Area = clustering(Obstacle_Area)        # generate cluster and barrier env

# Best cost every iteration and Best Solution
BestCostIt = np.zeros((1, MaxIt), dtype=float)
BestSol = []
BestSol.append({'Position': [], 'Cost': 1})

# First population initiation
pop = []
for _ in range(nPop):
    alpop = np.zeros((N, 4), dtype=float)
    position0 = np.random.uniform(sink[0]-rc/2, sink[1]+rc/2, (N, 2)) 
    phi0 = np.random.uniform(0, 2 * np.pi, (N, 1))
    alpop[:,:2] = position0
    alpop[:, 2] = phi0.ravel()
    alpop_cost, _ = CR_Func(alpop, stat, Obstacle_Area, Covered_Area)
    pop.append({'Position': alpop, 'Cost': alpop_cost})
    if alpop_cost < BestSol[0]['Cost']:
        BestSol[0]['Position'] = alpop
        BestSol[0]['Cost'] = alpop_cost

        
# %% ------------------------- MAIN LOOP --------------------------
for it in range(MaxIt):
    for i in range(nPop):
        k = np.random.randint(nPop)
        # TLBO operators
        alpop = ????
        # Box constraints apply
        alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)
        alpop[:, 2] = alpop[:, 2] % (2 * np.pi)
        # Cost Functions and connectivity constraint evaluation
        if Connectivity_graph(Graph(alpop[:, :2], rc),[]):
            alpop_cost = CostFunction_weighted(alpop, stat, w, Obstacle_Area, Covered_Area)
            if alpop_cost < pop[i]['Cost']:
                pop[i]['Position'] = alpop
                pop[i]['Cost'] = alpop_cost
                if alpop_cost < BestSol[0]['Cost']:
                    BestSol[0]['Position'] = alpop
                    BestSol[0]['Cost'] = alpop_cost
        
    BestCostIt[it] = BestSol[0]['Cost']
    print(f"Iter={it}")

# %% ------------------------- PLOT --------------------------
plt.figure()
# Plot environment
obs_row, obs_col = np.where(Obstacle_Area == 1)
plt.plot(obs_col, obs_row, '.', markersize=0.1, color='blue')
obs_row, obs_col = np.where(Obstacle_Area == 0)
plt.plot(obs_col, obs_row, '.', markersize=2, color='black')
discovered_obs_row, discovered_obs_col = np.where(Covered_Area == -1)
plt.plot(discovered_obs_col, discovered_obs_row, '.', markersize=2, color='red')
#discovered_row, discovered_col = np.where(Covered_Area == 1)
#plt.plot(discovered_col, discovered_row, '.', markersize=5, color='green')

# Plot network
alpop = BestSol[0]['Position']
theta_res = 100  # arc resolution
for i in range(N):
    # node's stat
    x0, y0 = alpop[i,1], alpop[i,0]
    phi = alpop[i,2]
    rs = stat[0,i]
    theta = stat[1,i]

    # node's center
    plt.plot(x0, y0, 'o', markersize=3, color='blue')
    plt.text(x0, y0, str(i+1), fontsize=9, color='red')

    # sensing area
    theta1 = phi - theta/2
    theta2 = phi + theta/2
    theta0 = np.linspace(theta1, theta2, theta_res)
    x_arc = x0 + rs * np.cos(theta0)
    y_arc = y0 + rs * np.sin(theta0)
    x_fill = np.concatenate(([x0], x_arc, [x0]))
    y_fill = np.concatenate(([y0], y_arc, [y0]))
    plt.fill(x_fill, y_fill, color=(0.6, 1, 0.6), alpha=0.3, edgecolor='k')

del x_fill, y_fill, theta

# Plot settings
plt.xlim([0, Obstacle_Area.shape[1]])
plt.ylim([0, Obstacle_Area.shape[0]])
plt.title(f"{BestCostIt[it]*100:.2f}% at time step: {it}")
plt.gca().invert_yaxis()
plt.grid(True)
plt.axis('equal')
plt.pause(0.001)

# %% ------------------------- DELETE --------------------------    
del alpop, alpop_cost, i, k, phi, size
# folder_name = 'data'
# file_name = f'DWABC_{Trial}.mat'
# save_mat(folder_name, file_name,pop,stat,MaxIt)
    
    
    
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
from utils.Multi_objective_functions import CostFunction_weighted
from utils.Workspace_functions import save_mat
from utils.Env_functions import Env_gen
from utils.Clustering_functions import clustering

# %% ------------------------- PARAMETERS --------------------------
np.random.seed(1)# init random seed
size = 100# Solution space's size
MaxIt = 500# Max iteration
nPop = 10# Number of populations/solutions
nTarget = 50# Number of targets
N = 60# Number of sensors ~ solution's size

rc = 10
theta0 = np.ones(N, dtype=int) * (np.pi)/3
stat = np.zeros((2, N))  # sensor stat matrix
stat[0, :] = theta0
stat[1, 0] = rc
rs = np.array([5, 20])   # sensing radius
w = np.array([0.5, 0.5]) # Weighted assign 

# %% ------------------------- INITIATION --------------------------
Covered_Area = np.zeros((size, size), dtype=int)
Obstacle_Area = Env_gen(nTarget)
Obstacle_Area = clustering(Obstacle_Area)
BestCostIt = np.zeros((1, MaxIt), dtype=float)
BestSol = []
BestSol.append({'Position': [], 'Cost': 1})

sink = np.array([size//2, size//2])
a = 1

pop = []
for _ in range(nPop):
    alpop = np.zeros((N, 4), dtype=float)
    position0 = np.random.uniform(sink[0]-rc/2, sink[1]+rc/2, (N, 2)) 
    position0[0,:2] = sink
    theta0 = np.random.uniform(0, 2 * np.pi, (N, 1))
    rs0 = np.random.uniform(rs[0], rs[1], (N, 1))
    alpop[:,:2] = position0
    alpop[:, 2] = theta0.ravel()
    alpop[:, 3] = rs0.ravel()
    alpop_cost = CostFunction_weighted(alpop, stat, w, Obstacle_Area, Covered_Area)
    pop.append({'Position': alpop, 'Cost': alpop_cost})
    if alpop_cost < BestSol[0]['Cost']:
        BestSol[0]['Position'] = alpop
        BestSol[0]['Cost'] = alpop_cost

        
# %% ------------------------- MAIN LOOP --------------------------
for it in range(MaxIt):
    #print("Exploration starts")
    for i in range(nPop):
        k = np.random.randint(nPop)
        # PDO operators
        alpop = ????
        # Box constraints apply
        alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)
        alpop[0,:] = sink
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
obs_row, obs_col = np.where(Obstacle_Area == 1)
plt.plot(obs_col, obs_row, '.', markersize=0.1, color='blue')
obs_row, obs_col = np.where(Obstacle_Area == 0)
plt.plot(obs_col, obs_row, '.', markersize=2, color='black')
discovered_obs_row, discovered_obs_col = np.where(Covered_Area == -1)
plt.plot(discovered_obs_col, discovered_obs_row, '.', markersize=2, color='red')
#discovered_row, discovered_col = np.where(Covered_Area == 1)
#plt.plot(discovered_col, discovered_row, '.', markersize=5, color='green')
alpop = BestSol[0]['Position']
theta = np.linspace(0, 2*np.pi, 500)
for i in range(N):
    plt.plot(alpop[i,1], alpop[i,0], 'o', markersize=3, color='blue')
    plt.text(alpop[i,1], alpop[i,0], str(i+1), fontsize=10, color='red')
    x = alpop[i,1] + alpop[i,3] * np.cos(theta)
    y = alpop[i,0] + alpop[i,3] * np.sin(theta)
    plt.fill(x, y, color=(0.6, 1, 0.6), alpha=0.2, edgecolor='k')
    
del x, y, theta
plt.xlim([0, Obstacle_Area.shape[1]])
plt.ylim([0, Obstacle_Area.shape[0]])
plt.title(f"{BestCostIt[it]*100:.2f}% at time step: {it}")
plt.gca().invert_yaxis()
plt.grid(True)
plt.pause(0.001)

# %% ------------------------- DELETE --------------------------    
del alpop, alpop_cost, i, k, phi, size
# folder_name = 'data'
# file_name = f'DWABC_{Trial}.mat'
# save_mat(folder_name, file_name,pop,stat,MaxIt)
    
    
    
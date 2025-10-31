import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils.Single_objective_functions import CR_Func
#from utils.Workspace_functions import save_mat
from utils.Env_functions import Env_gen
from utils.Clustering_functions import clustering
#from operator import itemgetter
# %%
'''
def validation(alpop, Obstacle_Area, ref_pop, size, valid_list):
    rng = np.random.default_rng()
    Sol_ObstacleMap = copy.deepcopy(Obstacle_Area)
    alpop = np.clip(alpop, 0 ,size - 1)
    for i in range(alpop.shape[0]):
        k = 0
        while Sol_ObstacleMap[int(alpop[i, 0]), int(alpop[i, 1])] != 0:
            if ref_pop[i, 0] != 0 and ref_pop[i, 1] != 0:
                alpop[i,0] += 1 * (ref_pop[i, 0] / abs(ref_pop[i, 0]))
                alpop[i,1] += ref_pop[i, 1] / ref_pop[i, 0] * (ref_pop[i, 1] / abs(ref_pop[i, 1]))
            elif ref_pop[i, 0] == 0 and ref_pop[i, 1] != 0: 
                alpop[i, 1] += 1 * (ref_pop[i, 1] * abs(ref_pop[i, 1]))
            elif ref_pop[i, 0] != 0 and ref_pop[i, 1] == 0: alpop[i, 0] += 1 * (ref_pop[i, 1] * abs(ref_pop[i, 1]))
            alpop[i] = np.clip(alpop[i], 0, size - 1)
            if int(np.rint(alpop[i, 0])) == 99: alpop[i,0] = 0
            elif int(np.rint(alpop[i, 0])) == 0: alpop[i, 0] = 99
            if int(np.rint(alpop[i, 1])) == 99: alpop[i,1] = 0
            elif int(np.rint(alpop[i, 1])) == 0: alpop[i, 1] = 99
            if k >= 100:
                print("Random pick.")
                while Sol_ObstacleMap[int(alpop[i, 0]), int(alpop[i, 1])] != 0:
                    alpop[i] = rng.choice(valid_list)
            k += 1
        Sol_ObstacleMap[int(alpop[i,0]), int(alpop[i, 1])] = -1
    return alpop
'''
# %%
rng = np.random.default_rng(1)# init random seed(k)
size = 100# Solution space's size
MaxIt = 200# Max iteration
nPop = 25# Number of populations/solutions
nTarget = 50# Number of targets
N = 30# Number of sensors ~ solution's size

# %% ------------------------- PARAMETERS --------------------------
# Network parameter
rs0 = np.ones(N, dtype=int) * 10
theta0 = np.ones(N, dtype=int) * (np.pi)/3
stat = np.zeros((3, N))  # sensor stat matrix

# sensor params 
stat[0, :] = rs0
stat[1, :] = theta0
stat[2, 0] = 1

# %% ------------------------- INITIATION --------------------------
# Environment initiation
Covered_Area = np.zeros((size, size), dtype=int)
Obstacle_Area = Env_gen(nTarget)                 # generate target env
Obstacle_Area = clustering(Obstacle_Area)        # generate cluster and barrier env
valid_row, valid_col = np.where(Obstacle_Area == 0)
valid_list = np.column_stack((valid_row, valid_col))

# Best cost every iteration and Best Solution
BestSol = []
BestCostIt = np.zeros((1, MaxIt), dtype=float)
BestSol.append({'Position': [], 'Cost': 1})

pop = []
for i in range(nPop):
    alpop = np.zeros((N, 3), dtype=float)
    position0 = np.random.uniform(0, size-1, (N, 2))
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
# %% ------------------------- Teacher Phase --------------------------
    posMean = np.zeros((N,3))
    BestCostIdx = 0
    for i in range(nPop):
        posMean += pop[i]['Position']
    posMean /= nPop
    temp_pop = copy.deepcopy(pop)
    count1=0
    for i in range(nPop):
        ref_pop = BestSol[0]['Position'][:,:] - ((rng.integers(1, 2, endpoint=True)) * posMean[:,:])
        pop[i]['Position']+= np.random.uniform(-1, 1, (3,)) * ref_pop
        #pop[i]['Position'][:,:2] = validation(pop[i]['Position'][:,:2], Obstacle_Area, ref_pop[:,:2], size, valid_list)
        pop[i]['Position'][:,:2] = np.clip(pop[i]['Position'][:,:2], 0 ,size - 1)
        pop[i]['Position'][:,2] = pop[i]['Position'][:,2] % (2 * np.pi)
        pop[i]['Cost'], _ = CR_Func(pop[i]['Position'], stat, Obstacle_Area, Covered_Area)
        if pop[i]['Cost'] > temp_pop[i]['Cost']: 
            count1+=1
            pop[i] = copy.deepcopy(temp_pop[i])
            if pop[i]['Cost'] < BestSol[0]['Cost']: BestCostIdx = i
    BestSol[0]['Position'] = pop[BestCostIdx]['Position']
    BestSol[0]['Cost'] = pop[BestCostIdx]['Cost']
    
# %% ------------------------- Learner Phase --------------------------
    temp_pop = copy.deepcopy(pop)
    BestCostIdx = 0
    count2=0
    for i in range(nPop):
        rand_range = list(range(nPop))
        rand_range.pop(i)
        partner = rng.choice(rand_range)
        for j in range(N):
            if pop[i]['Cost'] < temp_pop[partner]['Cost']:
                ref_pop = pop[i]['Position'][j] - temp_pop[partner]['Position'][j]
            else: 
                ref_pop = temp_pop[partner]['Position'][j] - pop[i]['Position'][j]
            pop[i]['Position'][j] += np.random.uniform(-1, 1, (3,)) * ref_pop
            #pop[i]['Position'][:,:2] = validation(pop[i]['Position'][:,:2], Obstacle_Area, ref_pop[:,:2], size, valid_list)
            pop[i]['Position'][:,:2] = np.clip(pop[i]['Position'][:,:2], 0 ,size - 1)
            pop[i]['Position'][:,2] = pop[i]['Position'][:,2] % (2 * np.pi)
            pop[i]['Cost'], _ = CR_Func(pop[i]['Position'], stat, Obstacle_Area, Covered_Area)
            if pop[i]['Cost'] > temp_pop[i]['Cost']:
                count2+=1
                pop[i] = copy.deepcopy(temp_pop[i])
                if pop[i]['Cost'] < pop[BestCostIdx]['Cost']: 
                    BestCostIdx = i
    BestSol[0]['Position'] = pop[BestCostIdx]['Position']
    BestSol[0]['Cost'] = pop[BestCostIdx]['Cost']
    BestCostIt[0, it] = BestSol[0]['Cost']
    print(f"Iter = {it}, Cost Function value {BestSol[0]['Cost']:.4f}, update times: {25-count1}, {25*30 - count2}")

# %% ------------------------- PLOT --------------------------
plt.figure()
# Plot environment
obs_row, obs_col = np.where(Obstacle_Area == 1)
plt.plot(obs_col, obs_row, '.', markersize=1, color='blue')
obs_row, obs_col = np.where(Obstacle_Area == 0)
plt.plot(obs_col, obs_row, '.', markersize=0.1, color='black')
obs_row, obs_col = np.where(Obstacle_Area == -1)
plt.plot(obs_col, obs_row, '.', markersize=1, color='red')

# Cập nhật vùng được che phủ LẦN CUỐI
_, Covered_Area = CR_Func(BestSol[0]['Position'], stat, Obstacle_Area, Covered_Area)

discovered_obs_row, discovered_obs_col = np.where(Covered_Area == -1)
plt.plot(discovered_obs_col, discovered_obs_row, '.', markersize=1, color='red')
discovered_row, discovered_col = np.where(Covered_Area == 1)
plt.plot(discovered_col, discovered_row, '.', markersize=1, color='green')

# Plot network
alpop = BestSol[0]['Position']
theta_res = 100  # arc resolution
for i in range(N):
    # node's stat
    x0, y0 = alpop[i,0], alpop[i,1] # Sửa: x là cột 0, y là cột 1
    phi = alpop[i,2]
    rs = stat[0,i]
    theta = stat[1,i]

    # node's center
    plt.plot(x0, y0, 'o', markersize=3, color='blue')
    #plt.text(x0, y0, str(i+1), fontsize=9, color='red')

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
# Sửa: Lấy giá trị 'it' cuối cùng
# <--- SỬA LỖI 3: Truy cập 2D
plt.title(f"{BestCostIt[0, MaxIt-1]*100:.2f}% at time step: {MaxIt-1}") 
# plt.gca().invert_yaxis() # Tọa độ (0,0) ở góc dưới bên trái
plt.grid(True)
plt.axis('equal')
plt.pause(0.001)
plt.show() # Hiển thị plot

# %% ------------------------- DELETE --------------------------    
del alpop, alpop_cost, i
# folder_name = 'data'
# file_name = f'DWABC_{Trial}.mat'
# save_mat(folder_name, file_name,pop,stat,MaxIt)
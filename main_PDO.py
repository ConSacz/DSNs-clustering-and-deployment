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
np.random.seed(2)# init random seed
size = 100# Solution space's size
MaxIt = 100# Max iteration
nPop = 50# Number of populations/solutions 
nTarget = 50# Number of targets
N = 30# Number of sensors ~ solution's size

# Sensor parameter
rc = 10
theta0 = np.ones(N, dtype=int) * (np.pi)/3 # sensing angle
rs0 = np.ones(N, dtype=int) * 10    # sensing radius
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
BestCostIt = np.zeros((1, MaxIt), dtype=float) # <--- Khởi tạo 2D (1, MaxIt)
BestSol = []
BestSol.append({'Position': np.zeros((N, 3)), 'Cost': 1}) # Khởi tạo BestSol với kích thước đúng

# First population initiation
pop = []
# Giả định sink là trung tâm của khu vực
for _ in range(nPop):
    alpop = np.zeros((N, 3), dtype=float)
    position0 = np.random.uniform(0, size - 1, (N, 2)) 
    phi0 = np.random.uniform(0, 2 * np.pi, (N, 1))
    alpop[:,:2] = position0
    alpop[:, 2] = phi0.ravel()
    alpop_cost, _ = CR_Func(alpop, stat, Obstacle_Area, Covered_Area)
    pop.append({'Position': alpop, 'Cost': alpop_cost})
    if alpop_cost < BestSol[0]['Cost']:
        BestSol[0]['Position'] = alpop
        BestSol[0]['Cost'] = alpop_cost

# %% ------------------ HELPER FUNCTION FOR PDO ------------------
def get_levy_step(shape, beta=1.5):
    import math
    sigma_u = (math.gamma(1+beta)*np.sin(np.pi*beta/2) /
                   (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma_u, shape)
    v = np.random.normal(0, 1, size=shape)
    step = u / (np.abs(v)**(1 / beta))
    
    # Sử dụng 0.1 làm hệ số co giãn, dựa trên tham số 'p' (rho)
    return step

        
# %% ------------------------- MAIN LOOP --------------------------
for it in range(MaxIt):
    count=0
    for i in range(nPop):
        # %% ----------------- PDO OPERATORS (BẮT ĐẦU) -----------------
        # Dựa trên Thuật toán 1 (trang 9) và các phương trình trong PDO.pdf
        
        # Lấy giải pháp hiện tại (PD_i) và giải pháp tốt nhất toàn cục (GBest)
        PD_i = pop[i]['Position']
        GBest = BestSol[0]['Position']
        
        for j in range(N):
            # Chúng ta chỉ cập nhật x, y, phi (3 cột đầu tiên)
            PD_i_update = PD_i[j, :3]
            k = np.random.randint(N)
            GBest_update = GBest[k, :3]
    
            # Lấy một giải pháp ngẫu nhiên (rPD) từ quần thể
            rand_idx = np.random.randint(nPop)
            rPD = pop[rand_idx]['Position']
            rPD_update = rPD[k, :3]
    
            # Lấy giá trị trung bình của tất cả các giải pháp (cho eCBest)
            all_pops_update = np.array([p['Position'][j, :3] for p in pop])
            mean_PD_update = np.mean(all_pops_update, axis=0)
            #các giá trị lấy đúng theo bảng giá trị pdo
            p, e, delta = 0.1, 2.2204e-16, 0.0001 
            
            # các hệ số tính theo công thức trong PDO
            r = 1 if (it % 2 == 0) else -1 
            DS_scalar = 1.5 * r * (1 - it / MaxIt)**(2 * it / MaxIt)
            PE_scalar = 1.5 * (1 - it / MaxIt)**(2 * it / MaxIt)
    
            # --- Định nghĩa các thành phần cập nhật (eCBest, CPD) ---
            UB = np.array([size - 1, size - 1, 2 * np.pi])
            LB = np.array([0, 0, 0])
            UB_LB_diff = UB - LB
            
            # eCBest
            eCBest = (GBest_update * delta) + (PD_i_update * mean_PD_update / ((GBest_update * UB_LB_diff) + delta))
    
            # CPD (Eq. 9)
            CPD = (GBest_update - rPD_update) / (GBest_update + delta)
            
            alpop = PD_i.copy() 
            
            if it < MaxIt / 4:
                Levy_n = get_levy_step(PD_i_update.shape)
                update_part = GBest_update - eCBest * p - CPD * Levy_n
                alpop[j, :3] = update_part
            
            elif MaxIt / 4 <= it < MaxIt / 2:
                Levy_n = get_levy_step(PD_i_update.shape)
                update_part = GBest_update * eCBest * DS_scalar * Levy_n
                alpop[j, :3] = update_part
    
            elif MaxIt / 2 <= it < 3 * MaxIt / 4:
                rand_arr = np.random.uniform(-1, 1, PD_i_update.shape)
                update_part = GBest_update - eCBest * e - CPD * rand_arr
                alpop[j, :3] = update_part
    
            else:
                rand_arr = np.random.rand(*PD_i_update.shape)
                update_part = GBest_update * PE_scalar * rand_arr
                alpop[j, :3] = update_part
            
            # %% ----------------- END OF PDO OPERATORS -----------------
    
            # Box constraints apply
            alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)
            alpop[:, 2] = alpop[:, 2] % (2 * np.pi)
            
            # Cost Functions and connectivity constraint evaluation
            alpop_cost, _ = CR_Func(alpop, stat, Obstacle_Area, Covered_Area)
            if alpop_cost <= pop[i]['Cost']:
                count+=1
                pop[i]['Position'] = alpop
                pop[i]['Cost'] = alpop_cost
                if alpop_cost < BestSol[0]['Cost']:
                    BestSol[0]['Position'] = alpop
                    BestSol[0]['Cost'] = alpop_cost
        
    BestCostIt[0, it] = BestSol[0]['Cost'] 
    print(f"Iter={it}, BestCoverage={(1-BestCostIt[0, it]):.4f}, update {count} times")
    
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
del alpop, alpop_cost, i, rand_idx, phi, size
# folder_name = 'data'
# file_name = f'DWABC_{Trial}.mat'
# save_mat(folder_name, file_name,pop,stat,MaxIt)
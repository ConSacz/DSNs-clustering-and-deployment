import networkx as nx
import numpy as np

def Graph(pop, rc):
    """
    Calculate the Graph of the network
    Parameters:
        pop (x, y, phi) * N is the optimization variable
        rc is the communication radius vector
    Returns:
        G (Graph-type): Graph of network (pop)
    """
    N = pop.shape[0]
    adj_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.linalg.norm(pop[i, :2] - pop[j, :2])
                if dist <= rc:
                    adj_matrix[i, j] = dist

    G = nx.from_numpy_array(adj_matrix, create_using=nx.Graph)
    return G

def Connectivity_graph(G, bat_ex):
    """
    Calculate the Connectivity constraint of graph G
    Parameters:
        G is the graph of network
        bat_ex is the vector holds index of exhausted nodes
    Returns:
        1: connected network
        0: unconnected network
    """
    G_sub = G.copy()
    #G_sub.remove_nodes_from(bat_ex)
    number_nodes = G.number_of_nodes() - len(bat_ex)
    reachable_nodes = list(nx.dfs_preorder_nodes(G_sub, source=0))

    if len(reachable_nodes) == number_nodes:
        return 1  # connected
    else:
        return 0  # not connected
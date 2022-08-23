import numpy as np
import algorithm.LDP_MST.NaN_Searching as NS
from tools.PrintFigures import PrintFigures
def LDP_Search(data):
    (size,dim) = data.shape
    (neighbor_num,nb,dis,dis1) = NS.scc(data)
    neighbor = []
    rho = np.zeros(size)
    max_nb = int(np.max(nb))
    for i in range(size):
        neighbor_temp = list(dis1[i,:neighbor_num+1])
        neighbor_temp.remove(i)
        neighbor.append(neighbor_temp)
    neighbor = np.array(neighbor)
    for i in range(size):#求每个点的局部密度
        d = 0
        for j in range(max_nb+1):
            d += dis[i,dis1[i,j]]
        rho[i] = nb[i] / d
        # print(i,rho[i])
    local_core = np.zeros(size).astype(int)
    for i in range(size):#找到每个点的邻居中局部密度最大的点，代表点
        rep = i
        xrcho = rho[rep]
        for j in range(neighbor_num+1):
            if xrcho < rho[dis1[i,j]]:
                xrcho = rho[dis1[i,j]]
                rep = dis1[i,j]
        local_core[i] = rep
    pf = PrintFigures()
    visited = np.zeros(size)
    round = 0
    for k in range(size):#找到核心点
        if visited[k] == 0:
            parent = k
            round += 1
            while local_core[parent] != parent:
                visited[parent] = round
                parent = local_core[parent]
            local_core[np.where(visited==round)[0]] = parent

    cluster_number = 0
    cores = []
    cl = np.zeros(size)#每个点所属的初始聚类号（把核心点和将该点看为核心的点的集合看成是聚类）
    for i in range(size):
        if local_core[i] == i:
            cluster_number = cluster_number + 1
            cores.append(i)#每个初始聚类的核心点
            cl[i] = cluster_number
    # pf.print_Dcores_line(data, cores, local_core)
    for i in range(size):
        cl[i] = cl[local_core[i]]
    return neighbor_num,neighbor,nb,rho,local_core,cores,cl,cluster_number



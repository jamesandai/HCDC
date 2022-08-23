import numpy as np
import algorithm.LDP_MST.NaN_Searching as NS
import algorithm.LDP_MST.LDP_Searching as LS
import algorithm.LDP_MST.LMSTCLU_OPT as LO

def LDPMST_opt(data,clu_num):
    (size,dim) = data.shape
    (neighbor_sum,neighbor,nb,rho,local_core,cores,cl,cluster_number) = LS.LDP_Search(data)
    #rho是局部密度
    #local_core是每个点的代表核心点
    #cores是所有核心代表点
    #cl是每个点所属的聚类号
    #cluster_number是分成的初始聚类数目，相当于核心代表点个数
    MDLP = []#保存每个初始局部密度峰值控制下都有哪些点

    NDLP = []#保存每个初始聚类中的点和其近邻
    nc_MDLP = np.zeros(cluster_number)#保存某个初始聚类中的点的个数
    core_dist = np.zeros((cluster_number,cluster_number))
    for i in range(cluster_number):
        core_dist[i,:] = NS.distance(data[cores],data[cores[i]],axis=1)
    maxd = np.max(core_dist)
    # print("初始聚类个数",cluster_number)
    # 求每个初始聚类中的点的个数和其近邻
    for i in range(cluster_number):
        number = np.where(local_core==cores[i])[0].astype(int)#MLDP[i]
        MDLP.append(list(number))
        nc_MDLP[i] = number.shape[0]
        temp = []
        if number.shape[0]!=1:
            for j in number:
                temp = set(temp).union(set(neighbor[j]))
        else:
            temp = set(neighbor[number[0]]).union(set(neighbor[number[0]]))
        NDLP.append(list(set(number).union(temp)))
        # NDLP.append(list(temp.copy()))
    #求两个核心点之间的距离
    for i in range(cluster_number):
        for j in range(i+1,cluster_number):
            insert = list(set(NDLP[i])&set(NDLP[j]))
            sumrho = np.sum(rho[insert])
            number = len(insert)
            if number == 0:
                core_dist[i,j] = maxd * (core_dist[i,j]+1)
            else:
                core_dist[i,j] = core_dist[i,j] / (sumrho*number)
            core_dist[j,i] = core_dist[i,j]
    # print(core_dist)
    minsize = 0.018 * size
    cores_cl = LO.LMSTCLU_opt(data[cores],core_dist,nc_MDLP,minsize,clu_num)
    #cores_cl是每个代表点所属的簇号
    cl = np.zeros(size)
    for i in range(cluster_number):
        cl[cores[i]] = cores_cl[i]
    for i in range(size):
        cl[i] = cl[local_core[i]]
    return cl




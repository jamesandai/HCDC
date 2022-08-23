import numpy as np

def distance(v1,v2,axis=0):
    return np.sqrt(np.sum(np.power(v1-v2,2),axis))
def scc(data):
    size,dim = data.shape
    dis = np.zeros((size,size))
    for i in range(size):
        dis[i,:] = distance(data[i,:],data,axis=1)
    dis1 = np.argsort(dis,axis=1,kind='mergesort')#每个点的邻居

    r = 1
    flag = 0
    nb = np.zeros(size)
    count = 0
    count1 = 0
    count2 = 0
    while flag == 0:#和原文一样，只不过加上了减掉了自己而已
        for i in range(size):
            k = dis1[i,r-1]
            nb[k] += 1
        r = r + 1
        count2 = np.where(nb<=1)[0].shape[0]
        if count1 == count2:
            count += 1
        else:
            count = 1
        if count2 == 0 or (r > 2 and count >= 2):
            flag = 1
        count1 = count2

    neigh_bor_sum = r - 2
    print(neigh_bor_sum)
    for i in range(size):
        nb[i] -= 1


    return(neigh_bor_sum,nb,dis,dis1)

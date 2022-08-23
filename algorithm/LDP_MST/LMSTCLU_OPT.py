import numpy as np
from pandas.core import sparse

import sys
sys.path.append("..")
from algorithm.LDP_MST.iscontain import iscontain
from algorithm.LDP_MST.iscontain2 import iscontain2
import algorithm.LDP_MST.mintree as mt
import matplotlib.pyplot as plt
from tools.PrintFigures import PrintFigures
def LMSTCLU_opt(data,dist,nc,minsize,clu_num):
    pf = PrintFigures()
    size,dim = data.shape
    UG = np.triu(dist)#上三角矩阵
    ST = mt.minTree(UG)#最小生成树
    edges = np.zeros(size-1)
    xlabel = np.zeros(size-1).astype(int)
    ylabel = np.zeros(size-1).astype(int)
    for i in range(size-1):
        l = ST[i]
        xlabel[i] = l[0]
        ylabel[i] = l[1]
        edges[i] = l[2]
    ST2 = np.zeros((size, size))
    for i in range(size - 1):
        ST2[xlabel[i]][ylabel[i]] = edges[i]
        ST2[ylabel[i]][xlabel[i]] = edges[i]
    # pf.print_MST_distance(data, ST)
    sortedind = np.argsort(edges)[::-1]
    i = 0
    k = 1
    while k < clu_num:
        s1 = 0
        s2 = 0
        t = sortedind[i]
        p = xlabel[t]
        q = ylabel[t]
        while s1 < minsize or s2 < minsize:
            s1 = 0
            s2 = 0
            visited = np.zeros(size)
            t = sortedind[i]
            i = i + 1
            p = xlabel[t]
            q = ylabel[t]
            queue = np.zeros(size).astype(int)
            front = 0
            rear = 0
            queue[rear] = p
            rear += 1
            while front != rear:
                temp = queue[front]
                s1 = s1 + nc[temp]
                visited[temp] = 1
                front = front + 1
                for j in range(size):
                    if ((ST2[temp,j]!=0 and ST2[j,temp]!=0) and j!=q and visited[j]==0 and (iscontain2(queue,front,rear,j))==0):
                        queue[rear] = j
                        rear += 1
            queue = np.zeros(size).astype(int)
            front = 0
            rear = 0
            queue[rear] = q
            rear += 1
            while front != rear:
                temp = queue[front]
                s2 = s2 + nc[temp]
                visited[temp] = 1
                front = front + 1
                for j in range(size):
                    if ((ST2[temp,j]!=0 and ST2[j,temp]!=0) and j!=p and visited[j]==0 and (iscontain2(queue,front,rear,j))==0):
                        queue[rear] = j
                        rear += 1
        # print(s1,s2)
        ST2[p,q] = 0
        ST2[q,p] = 0
        k = k + 1
    cl = np.zeros(size)#是否已经被访问过，记录每个核心点所属的子树号
    ncl = 0
    for i in range(size):
        if cl[i] == 0:
            ncl = ncl + 1
            queue = np.zeros(size).astype(int)
            front = 0
            rear = 0
            queue[rear] = i
            rear = rear + 1
            while front != rear:
                p = queue[front]
                front = front + 1
                cl[p] = ncl
                for j in range(i+1,size):
                    if ((ST2[p,j]!=0 and ST2[j,p]!=0) and cl[j]==0 and iscontain(queue,j)==0):
                        queue[rear] = j
                        rear = rear + 1
    return cl
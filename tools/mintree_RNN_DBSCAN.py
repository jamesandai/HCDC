import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import copy



class Graph(object):
    def __init__(self, Matrix):
        self.Matrix = Matrix
        self.nodenum = self.get_nodenum()
        self.edgenum = self.get_edgenum()

    def get_nodenum(self):
        return len(self.Matrix)

    def get_edgenum(self):
        count = 0
        for i in range(self.nodenum):
            for j in range(i):
                if self.Matrix[j][i] > 0:
                    count += 1
        return count
    def prim(self):
        list = []
        if self.nodenum <= 0 or self.edgenum < self.nodenum - 1:
            return list

        selected_node = [0]
        candidate_node = [i for i in range(1, self.nodenum)]
        while len(candidate_node) > 0:
            begin, end, minweight = 0, 0, 9999
            for i in selected_node:
                for j in candidate_node:
                    if self.Matrix[i][j] != 0:
                        if self.Matrix[i][j] < minweight:
                            minweight = self.Matrix[i][j]
                            begin = i
                            end = j
                    elif self.Matrix[j][i] != 0:
                        if self.Matrix[j][i] < minweight:
                            minweight = self.Matrix[j][i]
                            begin = i
                            end = j
            list.append([begin, end, minweight])
            selected_node.append(end)
            candidate_node.remove(end)
        return list

def minTree(Matrix):
    G = Graph(Matrix)
    # print('------最小生成树prim算法')
    # print('节点数据为%d，边数为%d\n' % (G.nodenum, G.edgenum))
    list = G.prim()
    max_dis = max_dist(list)
    return list,max_dis
def floyd(Tree_List,nodenum):#用floyd找到每两个点的最短路径，然后在里面找到任意两点的最大最短路径距离
    dis = np.ones((nodenum,nodenum)) * np.inf
    length = len(Tree_List)
    Tree_List = np.array(Tree_List)
    for i in range(length):
        dis[int(Tree_List[i,0]),int(Tree_List[i,1])] = Tree_List[i,2]
    for k in range(nodenum):
        for i in range(nodenum):
            for j in range(nodenum):
                if dis[i][j] > dis[i,k] + dis[k,j]:
                    dis[i][j] = dis[i,k] + dis[k,j]
    max_dist = 0
    for i in range(nodenum):
        for j in range(nodenum):
            if max_dist < dis[i,j] and dis[i,j] != np.inf:
                max_dist = dis[i,j]
    return max_dist

def max_dist(Tree_List):
    length = len(Tree_List)
    Tree_List = np.array(Tree_List)
    max_dis = 0
    for i in range(length):
        if max_dis < Tree_List[i,2]:
            max_dis = Tree_List[i,2]
    return max_dis



# if __name__ == "__main__":
#     Matrix = np.array(random.randint((10), size=(10, 10)))
#     for i in range(10):
#         Matrix[i,i] = 0
#     F = minTree(Matrix)
#     edges = np.zeros(9)
#     xlabel = np.zeros(9)
#     ylabel = np.zeros(9)
#     for i in range(9):
#         l = F[i]
#         xlabel[i] = l[0]
#         ylabel[i] = l[1]
#         edges[i] = l[2]
#     print(xlabel,ylabel,edges)



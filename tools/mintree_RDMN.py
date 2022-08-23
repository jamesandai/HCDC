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

    def kruskal(self):
        res = []  # 初始化最小生成树
        if self.nodenum <= 0 or self.edgenum < self.nodenum - 1:  # 若节点数<=0或边数<节点数-1，则
            return res
        edge_list = []  # 初始化边列表
        for i in range(self.nodenum):  # 循环头节点
            for j in range(i + 1, self.nodenum):  # 循环尾节点
                if self.Matrix[i][j] != 0:  # 若边的权重<9999，则
                    edge_list.append([i, j, self.Matrix[i][j]])  # 按[begin, end, weight]形式加入
                elif self.Matrix[j][i] != 0:
                    edge_list.append([j, i, self.Matrix[j][i]])
        edge_list.sort(key=lambda a: a[2])  # 将边列表按权重的升序排序
        group = [[i] for i in range(self.nodenum)]  # 生成可迭代的连通分量列表
        for edge in edge_list:  # 遍历边列表
            index_list = np.where(edge_list == edge)[0]
            for i in range(self.nodenum):  # 遍历生成连通分量列表的索引值
                if edge[0] in group[i]:  # 若当前边的头节点在连通分量列表当前索引的值中，则
                    m = i  # 当前边头节点在连通分量m中，m为连通分量列表的索引
                if edge[1] in group[i]:  # 若当前边的尾节点在连通分量列表当前索引的值中，则
                    n = i  # 当前边尾节点在连通分量n中，n为连通分量列表的索引
            # if edge[2] == edge_list[index+1][2] and index < self.edgenum - 1:
            #     continue
            if m != n:  # 若该边的两个节点不在同一连通分量中，则
                res.append(edge)  # 将头节点、尾节点、最小权重添加到最小生成树中
                group[m] = group[m] + group[n]  # 将连通分量列表的n合并到m中
                group[n] = []  # 清空连通分量列表n索引的连通分量
        return res


def minTree(Matrix):
    G = Graph(Matrix)
    print('------最小生成树prim算法')
    print('节点数据为%d，边数为%d\n' % (G.nodenum, G.edgenum))
    list = G.kruskal()
    return list
def floyd(Tree_List,nodenum):#用floyd找到每两个点的最短路径，然后在里面找到任意两点的最大最短路径距离
    dis = np.ones((nodenum,nodenum)) * 9999
    length = len(Tree_List)
    Tree_List = np.array(Tree_List)
    for i in range(length):
        dis[int(Tree_List[i,0]),int(Tree_List[i,1])] = Tree_List[i,2]
    for k in range(nodenum):
        for i in range(nodenum):
            for j in range(nodenum):
                if dis[i,k] != 9999 and dis[k,j] != 9999:
                    if dis[i][j] > dis[i,k] + dis[k,j]:
                        dis[i][j] = dis[i,k] + dis[k,j]
    max_dist = 0
    for i in range(nodenum):
        for j in range(nodenum):
            if max_dist < dis[i,j] and dis[i,j] != 9999:
                max_dist = dis[i,j]
    return max_dist

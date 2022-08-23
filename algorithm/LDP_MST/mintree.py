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
            begin, end, minweight = 0, 0, np.inf
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
    # print('节点数据为%d，边数为%d\n' % (G.nodenum, G.edgenum))
    # print('------最小生成树prim算法')
    # print(G.prim())
    return(G.prim())

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



from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import sklearn.neighbors
import pandas as pd
import numpy as np
import math
import csv


class Natural_Neighbor(object):

    def __init__(self):
        self.nan_edges = {}  # Grafo dos vizinhos mutuos
        self.nan_num = {}  # Numero de vizinhos naturais de cada instancia
        self.repeat = {}  # Estrututa de dados que contabiliza a repeticao do metodo count
        self.target = []  # Conjunto das classes
        self.data = []  # Conjunto de instancias
        self.knn = {}  # Estrutura que armazena os vizinhos de cada instanica
        self.rknn = {}
        self.last_cnt = -1

    # Divide o dataset em atributos e classes
    def load(self, filename):
        aux = []
        with open(filename, 'r') as dataset:
            data = list(csv.reader(dataset))
            for inst in data:
                inst_class = inst.pop(-1)
                self.target.append(inst_class)
                row = [float(x) for x in inst]
                aux.append(row)
        self.data = np.array(aux)

    def read(self, trainX, trainY=None):
        self.target = np.array(trainY)
        self.data = np.array(trainX)

    def asserts(self):
        self.nan_edges = set()
        for j in range(len(self.data)):
            self.knn[j] = set()
            self.rknn[j] = set()
            self.nan_num[j] = 0
            self.repeat[j] = 0

    # Retorna o numero de instancias que nao possuiem vizinho natural
    def count(self):
        nan_zeros = 0
        for x in self.nan_num:
            if self.nan_num[x] == 0:
                nan_zeros += 1
        return nan_zeros

    # Retorna os indices dos vizinhos mais proximos
    def findKNN(self, inst, r, tree):
        _, ind = tree.query([inst], r + 1)
        return np.delete(ind[0], 0)

    # Retorna o NaNe
    def algorithm(self):
        # ASSERT
        tree = KDTree(self.data)
        self.asserts()
        flag = 0
        r = 1
        count = 0
        while (flag == 0):
            for i in range(len(self.data)):
                knn = self.findKNN(self.data[i], r, tree)
                n = knn[-1]
                self.knn[i].add(n)
                self.rknn[n].add(i)
                if (i in self.knn[n] and (i, n) not in self.nan_edges):
                    self.nan_edges.add((i, n))
                    self.nan_edges.add((n, i))
                    self.nan_num[i] += 1
                    self.nan_num[n] += 1

            # cnt = self.count()
            # rep = self.repeat[cnt]#重复多少次
            # self.repeat[cnt] += 1
            # if (cnt == 0 or rep >= math.sqrt(r - rep)):
            #     flag = 1
            # else:
            #     r += 1
            cnt = self.count()#只要两次重复
            if (cnt == self.last_cnt) or cnt == 0:
                flag = 1
            else:
                self.last_cnt = cnt
                r += 1
        return r

if __name__ == "__main__":
    n = Natural_Neighbor()
    fileurl = '../datasets/Aggregation/'
    filename = fileurl + "dataset.csv"
    n.load(filename)
    r = n.algorithm()
    print(r)
    print(n.rknn)
import numpy as np
import sys
import pandas as pd
import scipy.io as scio
import math

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score,rand_score,fowlkes_mallows_score,adjusted_rand_score,normalized_mutual_info_score
from scipy.sparse.csgraph._min_spanning_tree import minimum_spanning_tree
import matplotlib.pyplot as plt
sys.path.append("../..")
from tools.PrintFigures import PrintFigures  # 绘图操作类
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import scipy.io as scio
import time
import tools.mintree_RDMN as mt
import ahc
from sklearn.cluster import AgglomerativeClustering

class ALGORITHM:
    pf = PrintFigures()
    fo = FileOperator()
    fouci = FileOperatoruci()
    # filename = '../../datasets/ED_Hexagon/dataset.csv'
    # filename = '../../datasets/VDD2.txt'
    # filename = '../../datasets/jain.txt'
    #
    # filename = '../../datasets/Compound/dataset.csv'

    #
    filename = "../../datasets/E6.txt"

    # filename = "../../datasets/d6.txt"

    # filename = "../../datasets/t7.txt"


    # filename = "../../datasets/mypoint.txt"'

    # filename = "../../datasets/point.txt"

    # filename = "../../datasets/grid.txt"
    # filename = "../../datasets/csv/dartboard2.csv"

    cluster_num = 7
    omega = 0.55
    def RunAlgorithm(self):
        # points,label = self.fo.readDatawithLabel(self.filename)
        # points, label = self.fo.readPoint(self.filename)
        points = self.fo.readDatawithoutLabel(self.filename)
        # points,label = self.fo.readDatawithLabel2(self.filename)  # Flame和R15

        size = points.shape[0]
        dis = self.get_Distance(points)
        dist = np.argsort(dis,axis=1)
        r,KNN,RNN,NN = self.NN_Search(dist)
        for i in range(size):
            KNN[i].append(i)
        Den,outliers = self.get_Density(dis,NN,KNN)
        # 感觉用SNN好点，不会产生不对称，也可以用这个不对称解决LDP-MST
        Rep,cores,outliers,Contain_point = self.get_Rep_and_core2(KNN, Den,NN,dis,outliers,points)
        #
        NND_childs = self.get_NNDchilds(cores, Rep, NN,outliers)
        GD = self.compute_similartiy(cores, Den, NND_childs,dis,Rep)
        cluster = ahc.ahc(points[cores],GD, self.cluster_num,Contain_point,size)
        CL = np.zeros(size)
        for i in range(len(cluster.clusters_index)):
            for j in cluster.clusters_index[i]:
                CL[cores[j]] = i + 1
        CL = self.assign_points(Rep,CL,outliers)
        self.pf.printScatter_Color(points, CL)

    def get_Distance(self,points):
        size = points.shape[0]
        dis = np.zeros((size,size))
        for i in range(size):
            for j in range(i+1,size):
                dd = np.linalg.norm(points[i] - points[j])
                dis[i,j] = dd
                dis[j,i] = dd
        return dis

    def NN_Search(self,dist):
        size = len(dist)
        r = 1
        flag = 0
        nb = np.zeros(size)
        count = 0
        count1 = 0
        count2 = 0
        KNN = []
        RNN = []
        for i in range(size):
            KNN.append([])
            RNN.append([])
        while flag == 0:
            count2 = 0
            NN = []
            for i in range(size):
                k = dist[i, r]
                KNN[i].append(k)
                RNN[k].append(i)
                nb[k] += 1
            for i in range(size):
                NN.append(list(set(KNN[i])&set(RNN[i])))
                if nb[i] == 0:
                    count2 += 1
            r = r + 1
            if count1 == count2:
                count += 1
            else:
                count = 1
            if count2 == 0 or count >= 2:
                flag = 1
            count1 = count2
        neigh_bor_sum = r - 1
        print(neigh_bor_sum)
        return (neigh_bor_sum,KNN,RNN,NN)

    def get_Density(self,dis,NN,KNN):
        size = len(NN)
        Den = np.zeros(size)
        outliers = []
        neighbor_all = []
        len_all = []
        for i in range(size):
            for j in NN[i]:
                SNN = list(set(KNN[i]) & set(KNN[j]))
                if len(SNN) != 0:
                    Den[i] += (len(SNN))**2 / ((np.sum(dis[i,SNN]) + np.sum(dis[j,SNN])) * (dis[i,j]+0.01))
        if self.omega != 0:
            for i in range(size):
                neighbor = set(KNN[i])
                for j in KNN[i]:
                    neighbor = neighbor.union(set(KNN[j]))
                neighbor = list(neighbor)
                neighbor_second = set(neighbor)
                for j in neighbor:
                    neighbor_second = neighbor_second.union(set(KNN[j]))
                neighbor_second = list(neighbor_second)
                neighbor_third = set(neighbor_second)
                for j in neighbor_second:
                    neighbor_third = neighbor_third.union(set(KNN[j]))
                neighbor_third = list(neighbor_third)
                neighbor_all.append(neighbor_third)
                len_all.append(len(neighbor_all[i]))

            len_min = np.min(len_all)
            len_max = np.max(len_all)
            for i in range(size):
                arg = np.sort(Den[neighbor_all[i]])
                index = math.ceil(self.omega * len(neighbor_all[i]))
                threshold = arg[index]
                delta = (len(neighbor_all[i]) - len_min) / (len_max - len_min) * 0.5
                if Den[i] < delta * threshold:
                    outliers.append(i)
        return Den,outliers


    def get_Rep_and_core2(self,KNN,Den,NN,dis,outliers,points):
        size = len(Den)
        Rep = np.ones(size).astype(int) * -1
        Den_argindex = np.argsort(Den)
        cores = []
        Rep_ = []
        for i in range(size):
            Rep_.append([])
        #在每个点的K近邻（包括自己）选择一个密度最大的点作为整个KNN的代表点，若是某点已经有了代表点，
        #则计算已有的代表点的密度与本点的密度差和两点之间距离和的乘积当前计算的，小的为新的代表点
        for i in Den_argindex:
            max_index = KNN[i][np.argmax(Den[KNN[i]])]
            max_Den = np.max(Den[KNN[i]])
            max_index = i if Den[i] > max_Den else max_index
            for j in KNN[i]:
                Rep_[j].append(max_index)
        for i in range(size):
            s = list(set(Rep_[i]))
            max_rep = -1
            max_rep_count = 0
            for j in s:
                num = Rep_[i].count(j)
                if num > max_rep_count:
                    max_rep = j
                    max_rep_count = num
                elif num == max_rep_count:
                    if dis[i,j] * np.abs(Den[i] - Den[j])  < dis[i,max_rep] * np.abs(Den[i] - Den[max_rep]):
                        max_rep = j
                        max_rep_count = num
            Rep[i] = max_rep
        visited = np.zeros(size)
        round = 0
        for i in range(size):
            if visited[i] == 0:
                parent = i
                round += 1
                while Rep[parent] != parent:
                    visited[parent] = round
                    parent = Rep[parent]
                Rep[np.where(visited == round)[0]] = parent
        false_core = []
        for i in range(size):#若一个噪声点是代表点，则是假核心点
            if Rep[i] == i and i not in outliers:
                cores.append(i)
            if Rep[i] == i and i in outliers:
                false_core.append(i)

        for i in range(size):#把所有噪声的代表点归为随便一个既不是真核心点也不是假核心点的点，方便后面操作
            if i not in cores and i not in false_core:
                Rep[outliers] = i
                break
        index = np.zeros(size).astype(int)
        k = 0
        for i in cores:
            index[i] = k
            k += 1
        delete_cores = []
        NND_childs = self.get_NNDchilds(cores,Rep,NN,outliers)
        d = {}
        k = 0
        for i in range(len(cores)):
            d[cores[i]] = k
            k += 1
        index = np.argsort(Den[cores]).astype(int)
        cores_sort = []
        for i in index:
            cores_sort.append(cores[i])
        for i in cores_sort:
            if i in delete_cores:
                continue
            else:
                for j in cores_sort:
                    if i in delete_cores:
                        break
                    if j in delete_cores:
                        continue
                    else:
                        if j in NND_childs[d[i]] or i in NND_childs[d[j]]:
                                if Den[i] < Den[j]:
                                    Rep[np.where(Rep == i)[0]] = j
                                    cores.remove(i)
                                    delete_cores.append(i)
                                    NND_childs[d[i]] = list(set(NND_childs[d[i]]).union(set(NND_childs[d[j]])))
                                    break
                                if Den[i] > Den[j]:
                                    Rep[np.where(Rep == j)[0]] = i
                                    cores.remove(j)
                                    NND_childs[d[j]] = list(set(NND_childs[d[i]]).union(set(NND_childs[d[j]])))
                                    delete_cores.append(j)
                                    continue

        Contain_point = []
        for i in range(len(cores)):
            Contain_point.append(np.where(Rep==cores[i])[0])
        return Rep, cores,outliers,Contain_point


    #将每个核心点的NND_childs看做是以他们自己为代表点的点的NN近邻并集
    def get_NNDchilds(self,cores,Rep,NN,outliers):
        NND_childs = []
        for i in range(len(cores)):
            NND_child = list(np.where(Rep==cores[i])[0])
            for j in NND_child:
                if j in outliers:
                    NND_child.remove(j)
            temp = []
            if len(NND_child) != 1:
                for j in NND_child:
                    temp = set(temp).union(set(NN[j]).difference(outliers))#NN[j]里不在outliers的点
            else:
                temp = set(NN[NND_child[0]]).union(set(NN[NND_child[0]]))
            NND_childs.append(list(set(NND_child).union(temp.difference(outliers))))
        return NND_childs



    def compute_similartiy(self,cores,Den,NND_childs,dis,Rep):
        length = len(cores)
        similarity = np.zeros((length,length))
        for i in range(length):
            similarity[i,i] = 0
        maxd = 0
        for i in range(length):
            for j in range(i+1,length):
                if maxd < dis[cores[i],cores[j]]:
                    maxd = dis[cores[i],cores[j]]
        max_p = 1
        for i in range(length):
            for j in range(i+1,length):
                common = list(set(NND_childs[i]) & set(NND_childs[j]))

                rep_i = np.where(Rep==cores[i])[0]
                rep_j = np.where(Rep==cores[j])[0]
                mean_Deni = np.mean(Den[rep_i])
                mean_Denj = np.mean(Den[rep_j])
                if len(common) != 0:

                    dd = dis[cores[i], cores[j]] * max(len(NND_childs[i]),len(NND_childs[j])) * np.abs(mean_Deni + mean_Denj) / ((len(common) ** 2) * (2 * np.sqrt(mean_Deni * mean_Denj)))

                    max_p = max(max_p,len(NND_childs[i]) * len(NND_childs[j]) / (len(common) ** 2))
                    similarity[i, j] = dd
                    similarity[j, i] = dd
        for i in range(length):
            for j in range(i + 1, length):
                common = list(set(NND_childs[i]) & set(NND_childs[j]))
                if len(common) == 0:
                    dd = (max_p) * dis[cores[i],cores[j]] * maxd
                    similarity[i, j] = dd
                    similarity[j, i] = dd
        return similarity


    #把剩余的点分配给他们的代表点
    def assign_points(self,Rep,CL,outliers):
        size = CL.shape[0]
        for i in range(size):
            if CL[i] == 0 and i not in outliers:
                CL[i] = CL[Rep[i]]
        return CL
    def min_max(self,data):
        size,dim = data.shape
        for j in range(dim):
            max = np.max(data[:,j])
            min = np.min(data[:,j])
            for i in range(size):
                data[i,j] = (data[i,j] - min) / (max - min)
        return data



if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    a = ALGORITHM()
    a.RunAlgorithm()
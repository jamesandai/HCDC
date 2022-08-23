import numpy as np
import sys
import pandas as pd
import scipy.io as scio
import math

from sklearn import preprocessing

sys.path.append("..")
from tools.PrintFigures import PrintFigures  # 绘图操作类
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci
from sklearn.metrics import normalized_mutual_info_score, fowlkes_mallows_score, adjusted_rand_score
import scipy.io as scio
import time


#对假峰值点进行分给已有的类中，
# 把假峰值点的汇聚点改为离他最近的真峰值点，把所有以假峰值点为汇聚点的点的汇聚点全改为该真峰值点
#在减少聚类数时，是根据其包括的峰值点的个数排序，而不是含有的点的个数
class Dcore:
    #应对多密度时就不行，特别是高密度被低密度包裹住时
    # r1 = 5 #Aggregation
    # r2 = 4
    # r = 5.5
    # t1 = 30
    # t2 = 8

    # r1 = 14 #VDD_Heartshaped
    # r2 = 10
    # r = 10
    # t1 = 5
    # t2 = 2

    # r1 = 0.15 #3#d6.txt
    # r2 = 0.1
    # r = 0.5
    # t1 = 10
    # t2 = 4

    # r1 = 17 #ED_Hexagon
    # r2 = 10
    # r = 20
    # t1 = 4
    # t2 = 4

    # r1 = 0.9 #MDDM_D31
    # r2 = 0.8
    # r = 0.55
    # t1 = 36
    # t2 = 20

    # r1 = 6 #Compound(只能识别5个聚类)
    # r2 =  5
    # r = 1.5
    # t1 = 12
    # t2 = 8

    # r1=1.1 #E6.txt
    # r2=0.98
    # r=2
    # t1=30
    # t2=10

    #t8
    # r1 = 18
    # r2 = 12
    # r = 25
    # t1 = 30
    # t2 = 10

    #t7

    #olivetti(唯一一个用的是比例的,因为真实值可能太大了）（若是r1=0.91,r2=0.81,r=0.91的话效果差）
    # r1 = 0.9
    # r2 = 0.81
    # r = 0.88
    # t1 = 1
    # t2 = 0

    #t4.txt(r1,r2,r都不是按比例）
    # r1 = 15
    # r2 = 14
    # r = 15
    # t1 = 40
    # t2 = 10

    #t7.txt
    # r1 = 18
    # r2 = 16
    # r = 20
    # t1 = 40
    # t2 = 10

    # #Flame.txt
    # r1 = 1.5
    # r2 = 1
    # r = 0.8
    # t1 = 4
    # t2 = 2

    #spiral.txt
    # r1 = 2
    # r2 = 1
    # r = 1.5
    # t1 = 3
    # t2 = 2

    #R15.txt
    # r1 = 2
    # r2 = 1
    # r = 0.5
    # t1 = 8
    # t2 = 2

    #ThreeCircles.txt
    # r1 = 1
    # r2 = 0.5
    # r = 0.4
    # t1 = 3
    # t2 = 2

    #mypoint
    # r1 = 1
    # r2 = 0.5
    # r = 0.3
    # t1 = 10
    # t2 = 6
    #point
    # r1 = 12
    # r2 = 8
    # r = 10
    # t1 = 3
    # t2 = 2

    #jain
    # r1 = 1.5
    # r2 = 1
    # r = 4
    # t1 = 4
    # t2 = 2


    #dartboard2
    # r1 = 0.02
    # r2 = 0.01
    # r = 0.01
    # t1 = 4
    # t2 = 2

    #grid
    # r1 = 1
    # r2 = 0.5
    # r = 1
    # t1 = 5
    # t2 = 2

    #zoo
    # r1 = 1.6
    # r2 = 0.85
    # r = 1
    # t1 = 3
    # t2 = 2

    #divorce
    # r1 = 1.18
    # r2 = 0.55
    # r = 0.5
    # t1 = 3
    # t2 = 2

    #coloumn
    # r1 = 0.3
    # r2 = 0.18
    # r = 0.1
    # t1 = 3
    # t2 = 2

    #yeast
    # r1 = 0.15
    # r2 = 0.1
    # r = 0.08
    # t1 = 3
    # t2 = 2

    #wifi
    # r1 = 0.78
    # r2 = 0.45
    # r = 0.18
    # t1 = 3
    # t2 = 2

    #digit
    # r1 = 1
    # r2 = 0.8
    # r = 0.6
    # t1 = 6
    # t2 = 4

    #page
    r1 = 0.25
    r2 = 0.15
    r = 0.2
    t1 = 10
    t2 = 2

    STRATEGY = 2#或者是2
    fo = FileOperator()
    fouci = FileOperatoruci()
    pf = PrintFigures()
    predefined_cluster_num = 20
    # filename = '../datasets/VDD2.txt'
    # filename = '../datasets/ED_Hexagon/dataset.csv'#k = 2
    # filename = '../datasets/MDDM_D31/dataset.csv'#k = 31
    # filename = '../datasets/MDDM_G2/dataset.csv'#k=2
    # filename = '../datasets/VDD_Heartshaped/dataset.csv'#k = 3
    # filename = '../datasets/Aggregation/dataset.csv'#k = 7
    # filename = '../datasets/Compound/dataset.csv'#k=6
    # filename = '../datasets/G50/dataset.csv'
    # filename = "../datasets/E6.txt"#k=7
    # filename = "../datasets/d6.txt"#k = 4
    # filename = "../datasets/t4.txt"#k=6
    # filename = "../datasets/t7.txt"  #k=9
    # filename = "../datasets/spiral.txt" #k =2
    # filename = "../datasets/ThreeCircles.txt" #k=3
    # filename = "../datasets/Flame.txt"#k = 2
    # filename = "../datasets/R15.txt"  #k = 15
    # filename = "../datasets/point.txt"
    # filename = "../datasets/mypoint.txt"#14#0.2#14是最好的
    # filename = "../datasets/t8.dat"#8
    # filename = "../datasets/VDD2.txt"
    # filename = '../datasets/jain.txt'
    # filename = "../datasets/csv/dartboard2.csv"  ##4
    # filename = "../datasets/grid.txt"

    # filename = "../uci datasets/segmentation.data"  # 0.693,0.546,0.608
    # filename = "../uci datasets/dermatology.data"#0.621,0.415,0.538(0.99 2)
    # filename = "../uci datasets/seeds_dataset.txt" # 0.554,0.438,0.597
    # filename = "../uci datasets/zoo.data"  # 0.787,0.640,0.719(r1 = 1.6,r2 = 0.85,r = 1,t1 = 3,t2 = 2)
    # filename = "../uci datasets/parkinsons.data"#0.212,0.070,0.573（r1 = 1,r2 = 0.95,r = 0.5,t1 = 3,t2 = 2）
    # filename = "../uci datasets/optdigits.data"  # 0.641,0.434,0.489(r1 = 1,r2 = 0.8,r = 0.6,t1 = 6,t2 = 4)
    # filename = "../uci datasets/ecoli.data"  # 0.659,0.715,0.801(r1 = 0.4，r2 = 0.25，r = 0.15，t1 = 3，t2 = 2)
    # filename = "../uci datasets/wifi"#0.432,0.329,0.591（r1 = 0.78,r2 = 0.45,r = 0.18,t1 = 3,t2 = 2）
    # filename = "../uci datasets/coloumn"#0.328,0.212,0.536(r1 = 0.3,r2 = 0.18,r = 0.1,t1 = 3,t2 = 2)
    # filename = "../uci datasets/yeast.data"#0.295,0.131,0.292(r1 = 0.15,r2 = 0.1,r = 0.08,t1 = 3,t2 = 2)
    filename = "../uci datasets/Page"#(0.288,0.448,0.908)(r1= 0.25 r2 = 0.15 R = 0.2 T1 = 10 Tn = 2)
    # filename = "../uci datasets/divorce.csv"#0.628,0.648,0.814(r1 = 1.18，r2 = 0.55，r = 0.5，t1 = 3，t2 = 2)
    def RunAlgorithm(self):
    # def RunAlgorithm(self):
        # filename = self.fileurl + "dataset.csv"
        # points,label = self.fo.readDatawithLabel(self.filename)
        # points,label = self.fo.readSpiral(self.filename)
        # points = self.fo.readDatawithoutLabel(self.filename)

        # points = self.fo.readDatawithoutLabel_t8(self.filename)
        # points,label = self.fo.readDatawithLabel2(self.filename)  # Flame和R15
        # points,label = self.fo.readThreeCircle(self.filename)

        # data = scio.loadmat('../datasets/points.mat')
        # points = (data['points'])
        # points, label = self.fo.readPoint(self.filename)
        # data = scio.loadmat(data_path)

        # points, label = self.fouci.readSegmentation(self.filename)
        # points, label = self.fouci.readDermetology(self.filename)
        # points,label = self.fouci.readSeed(self.filename)
        # points, label = self.fouci.readZoo(self.filename)
        # points, label = self.fouci.readNewthyroid(self.filename)
        # points, label = self.fouci.readDigits(self.filename)
        # points, label = self.fouci.readParkinsons(self.filename)
        # points, label = self.fouci.readEcoli(self.filename)
        # points, label = self.fouci.readWifi(self.filename)
        # points, label = self.fouci.readColoumn(self.filename)
        # points, label = self.fouci.readYeast(self.filename)
        points, label = self.fouci.readPage(self.filename)
        # points, label = self.fouci.readAba(self.filename)
        # points, label = self.fouci.readLym(self.filename)
        # points, label = self.fouci.readDigits(self.filename)
        # points, label = self.fouci.readDivorce(self.filename)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        points = min_max_scaler.fit_transform(points)
        length,dim = points.shape
        #绘制原始图
        # self.pf.printScatter_Color(points,label)
        # self.pf.printScatter(points)
        #获取距离矩阵
        start = time.time()
        dis,dist = self.getDistance(points)
        sort_dist = np.sort(dist)
        #获取r1距离
        # r1_position = math.floor(length*(length-1)/2*self.r1/100)
        # r1_dc = sort_dist[r1_position]
        #获取r2距离
        # r2_position = math.floor(length*(length-1)/2*self.r2/100)
        # r2_dc = sort_dist[r2_position]
        r1_dc = self.r1
        r2_dc = self.r2
        #执行算法1，找到每个点的近邻和密度和中心点和汇聚点（CPV是中心点，CP是汇聚点)
        r1_neighbor,r2_neighbor,density_r1,density_r2,CPV,CP = self.Find_center_and_calculate_density(points,dis,r1_dc,r2_dc)
        # r_position = math.floor(length*(length-1)/2*self.r/100)
        # self.r = sort_dist[r_position]
        #执行算法2，识别出局部密度峰值点和假峰值点
        FLP,LPS = self.Identify_true_and_false_peaks(points,CP, density_r1, density_r2, self.t1, self.t2,r2_dc,r1_dc)
        RCP = list(set(LPS).difference(set(FLP)))#真峰值点
        #执行算法3，获取每个点的初始聚类信息（不是真峰值点的为0，是真峰值点的话形成一个聚类中心群）
        cl = self.NNC(dis, self.r, length, RCP,flag=0)
        # self.pf.printScatter_Color(points,cl)
        #处理假峰值点
        cl,CP = self.Deal_with_false_peaks(FLP,density_r1,density_r2,dis,length,cl,CP,RCP)
        # #处理不属于峰值点的点
        # cl = self.final_cluster(cl)
        cl = self.Deal_with_other_point(points, RCP, cl, CP)
        cl = self.Merge_outliers(cl,dis)
        end = time.time()
        print(end-start)

        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(label, cl), ari(label, cl), fmi(label, cl)))
        # print(NMI(cl,label))

        # self.pf.printScatter_Color(points, cl)


        return
    #算法1.找到中心点和计算每个点的密度
    def Find_center_and_calculate_density(self,points,dis,r1_dc,r2_dc):
        r1_neighbor,r2_neighbor,density_r1,density_r2 = self.searchNeighbor(points,dis,r1_dc,r2_dc)#计算每个点的密度和近邻
        CPV = self.searchCenterpoint(points, r1_neighbor,density_r1)#计算每个点的中心点
        CP = self.searchCoveragepoint(points,CPV)#找到每个点的汇聚点
        return r1_neighbor,r2_neighbor,density_r1,density_r2,CPV,CP

    #1.获取距离矩阵(dis是距离矩阵,dist是所有的距离列表)
    def getDistance(self,points):
        length = points.shape[0]
        dis = np.zeros((length,length))
        dist = []
        for i in range(length):
            for j in range(i+1,length):
                dd = np.linalg.norm(points[i]-points[j])
                dis[i,j] = dd
                dis[j,i] = dd
                dist.append(dd)
        dist = np.array(dist)
        return dis,dist

    #2.寻找每个点在r1_dc和r2_dc里的局部近邻点，和在r1_dc和r2_dc的密度
    def searchNeighbor(self,points,dis,r1_dc,r2_dc):
        r1_neighbor = []
        r2_neighbor = []
        length = len(points)
        density_r1 = np.zeros(length)
        density_r2 = np.zeros(length)
        for i in range(length):
            l1 = (np.where(dis[i] <= r1_dc)[0]).tolist()
            l1.remove(i)
            l2 = (np.where(dis[i] <= r2_dc)[0]).tolist()
            l2.remove(i)
            r1_neighbor.append(l1)
            r2_neighbor.append(l2)
            density_r1[i] = len(l1)
            density_r2[i] = len(l2)
        return r1_neighbor,r2_neighbor,density_r1,density_r2

    #3.寻找每个点的中心点
    def searchCenterpoint(self,points,r1_neighbor,density_r1):
        length = len(points)
        CPV = np.zeros(length)
        for i in range(length):#若没近邻，则看成自己
            if density_r1[i] == 0:
                CPV[i] = i
            else:
                center_point = np.sum(points[r1_neighbor[i]],axis=0)
                center_point = center_point + points[i]
                center_point = center_point / (density_r1[i]+1)
                mindist = np.linalg.norm(center_point-points[i])
                CPV[i] = i  # 先设置为自己
                for j in r1_neighbor[i]:#在近邻中找到最接近中心点的点（可能出现两个一样距离的点！）
                    distance = np.linalg.norm(center_point-points[j])
                    if distance < mindist:
                        mindist = distance
                        CPV[i] = j

        return CPV

    #4.均值平移，找到每个点的汇聚点
    def searchCoveragepoint(self,points,CPV):
        length = len(points)
        CP = np.zeros(length).astype(int)
        for i in range(length):
            path = []
            k = i
            path.append(k)
            while k != int(CPV[k]):
                if CPV[k] in path:
                    print("...")
                    k = int(CPV[k])
                    break
                else:
                    k = int(CPV[k])
                    path.append(k)
            CP[i] = k
        return CP

    #算法2.识别局部密度峰值和假峰值
    def Identify_true_and_false_peaks(self,points,CP,density_r1,density_r2,t1,t2,r2_dc,r1_dc):
        LPS = []#所有的密度峰值点（包括真的和假的）
        FLP = []#假密度峰值点
        length,dim = points.shape
        for i in range(length):
            if i == CP[i] and density_r1[i] > t1:
                LPS.append(i)
                if density_r2[i] < t2:
                    FLP.append(i)
        return FLP,LPS

    #算法3.最近邻聚类法
    def NNC(self,dis,r,length,cluster_point,flag,cl=None):
        if flag == 0:
            cl = np.zeros(length)
            cl.astype(int)
        h = max(cl)
        for i in cluster_point:#看看
            if cl[i] == 0:
                neighbor = []
                h += 1
                cl[i] = h
                for j in cluster_point:
                    if j == i:
                        continue
                    else:
                        if dis[i,j] < r and cl[j] == 0:
                            neighbor.append(j)
                n_len = len(neighbor)
                while n_len!=0:
                    for j in neighbor:
                        cl[j] = h
                        for k in cluster_point:
                            if dis[j,k] < r and cl[k] == 0 and k not in neighbor:
                                neighbor.append(k)
                        neighbor.remove(j)
                    n_len = len(neighbor)
        return cl

    #算法4.处理假峰值点,把（假峰值点以及将他们看做是汇聚点的点）的汇聚点设置为离假峰值最近的真峰值点
    def Deal_with_false_peaks(self,FLP,density_r1,density_r2,dis,length,cl,CP,RCP):
        #策略1的东西
        falsePeaksForNewCluster = []  # 自己成为新类
        falsePeaksAssignedToNearestCluster = []  # 把他分配给最近的类
        #策略2的东西
        if self.STRATEGY == 1:
            for i in FLP:
                if density_r2[i] / density_r1[i] >= self.s1:
                    falsePeaksForNewCluster.append(i)
                else:
                    if density_r2[i] / density_r1[i] >=self.s2:
                        falsePeaksAssignedToNearestCluster.append(i)
                    else:
                        cl[i] = 0#cl为0是噪声点
            cl = self.NNC(dis, self.r, length, falsePeaksForNewCluster,1,cl)#对于要单独聚类的噪声，用NNC赋予他们初始聚类


        if self.STRATEGY == 2:
            for i in FLP:
                falsePeaksAssignedToNearestCluster.append(i)
        max_cl = int(np.max(cl))#总共有几个初始聚类
        c = []#每个初始聚类里有的点
        for j in range(1,max_cl+1):
            temp = []
            for i in range(length):
                if cl[i] == j:
                    temp.append(i)
            c.append(temp)
        for i in falsePeaksAssignedToNearestCluster:
            minst = np.Inf
            index = 0
            for j in RCP:
                if dis[i,j] < minst:
                    minst = dis[i,j]
                    index = j
            CP[i] = index
            for k in range(length):
                if CP[k] == i:
                    CP[k] = index



        return cl,CP

    #算法5.处理剩余点
    def Deal_with_other_point(self,points,LPS,cl,CP):
        length = len(points)
        for i in range(length):
            if i not in LPS:
                cl[i] = cl[CP[i]]
        return cl

    #算法6.若得出的聚类数小于输入的预期聚类数，则不变，若大于输入的预期聚类数，则把聚类点数少的聚类删掉，作为离群点
    def final_cluster(self,cl):
        cluster_num = int(np.max(cl))#0-8
        cluster_final_num = self.predefined_cluster_num#7
        class_contain_zero = np.zeros(cluster_num+1)#包含了0的点
        length = len(cl)
        class_num = []
        for i in range(length):
            class_contain_zero[int(cl[i])] += 1
        for i in range(1,cluster_num+1):
            class_num.append(class_contain_zero[i])
        class_num = np.array(class_num)
        sort_index = np.argsort(class_num)
        delete_class = sort_index[0:cluster_num - cluster_final_num]
        for i in range(length):
            if cl[i] in delete_class:
                cl[i] = 0
        return cl

    def Merge_outliers(self,cl,dis):
        size = len(dis)
        outliers = np.where(cl==0)[0]
        for i in outliers:
            mindst = np.inf
            for j in range(size):
                if cl[j] != 0 and mindst > dis[i,j]:
                    mindst = dis[i,j]
                    class_number = cl[j]
            cl[i] = class_number
        return cl






def main():
    dcore = Dcore()
    dcore.RunAlgorithm()
if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    main()
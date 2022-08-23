import numpy as np
import sys

from sklearn import preprocessing
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, rand_score, fowlkes_mallows_score

sys.path.append("..")
from tools.PrintFigures import PrintFigures  # 绘图操作类
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci  # 文件操作类
from sklearn.metrics import adjusted_rand_score
import tools.mintree_RNN_DBSCAN as mt #最小生成树类
import time
class RNN_DBSCAN:
    def __init__(self):
        self.fo = FileOperator()
        self.fouci = FileOperatoruci()
        self.pf = PrintFigures()
        self.k = 15
        # filename = '../datasets/VDD2.txt'#k=15
        # filename = '../datasets/ED_Hexagon/dataset.csv'#k = 10
        # filename = '../datasets/VDD_Heartshaped/dataset.csv'#k = 16
        # filename = '../datasets/Aggregation/dataset.csv'#k = 14
        # filename = '../datasets/Compound/dataset.csv'#k=10
        # filename = '../datasets/G50/dataset.csv'
        # filename = "../datasets/jain.txt"#k=15
        # filename = "../datasets/E6.txt"#7个类选择70 # k = 70,80最好,虽然只能识别出6个类,50的话也是双螺旋出了问题，但是有13个类,40的话可以识别出双螺旋的大概，但是末梢呗分成几类，总共10类,60有10个类，没40好
        # filename = "../datasets/d6.txt"#4个类选择40 # k = 50,70,有6个类，最好的话是40，60或者80，只有5个类
        # filename = "../datasets/t4.txt"#6个类选择50#40,50,60,70,80都很好，可以检测到6个类
        # filename = "../datasets/t7.txt"选择50  #9个类 # k = 40,50有9个类,70和80只有7个类，60只有8个类
        # filename = "../datasets/spiral.txt" # k = 10
        # filename = "../datasets/ThreeCircles.txt" #k=20
        # filename = "../datasets/Flame.txt"#k = 11 or 18
        # filename = "../datasets/R15.txt"  # k = 11 or 12
        # filename = "../datasets/mypoint.txt"#30
        # filename = "../datasets/point.txt"#30
        # filename = "../datasets/grid.txt"#10
        # filename = "../datasets/csv/dartboard2.csv"#20

        # uci
        # filename = "../uci datasets/iris.data"  # k=10,0.734,0.568,0.771
        # filename = "../uci datasets/wine.data"#k=10 0.000,0.000,0.581
        # filename = "../uci datasets/new-thyroid.data" #k=45 0.330,0.448,0.803
        # filename = "../uci datasets/segmentation.data"#k=40,0.576,0.245,0.500
        # filename = "../uci datasets/dermatology.data"#k=10 0.454,0.207,0.540
        # filename = "../uci datasets/zoo.data"  # k=10,0.777,0.713,0.811
        # filename = "../uci datasets/parkinsons.data"#k=35,0.018,-0.026,0.769
        # filename = "../uci datasets/ecoli.data"  # k=20,0.516,0.429,0.672
        # filename = "../uci datasets/wifi"#k=10,0.019,0.000,0.495
        # filename = "../uci datasets/coloumn"  # k=10,0.004,-0.002,0.608
        # filename = "../uci datasets/yeast.data"  # k=10,0.056,0.012,0.472
        filename = "../uci datasets/Page"#k=15,0.213,0.281,0.893
        # filename = "../uci datasets/divorce.csv"#k=15,0.728,0.789,0.889
        # filename = "../uci datasets/lymphography.data"#k=10,0.023,0.007,0.486
        # filename = "../uci datasets/optdigits.data"#k=15,0.565,0.182,0.431

        # self.points,self.label = self.fo.readDatawithLabel2(filename)  # Flame和R15
        # self.points, self.label = self.fo.readPoint(filename)
        # self.points,self.label = self.fo.readDatawithLabel(filename)
        # self.points = self.fo.readDatawithoutLabel(filename)  # E6 d6 t4 t7
        # self.points,self.label = self.fo.readSpiral(filename)
        # self.points,self.label = self.fo.readThreeCircle(filename)

        # self.points,self.label = self.fo.readDatawithoutLabel(filename)

        # self.points, self.label = self.fouci.readIris(filename)
        # self.points, self.label = self.fouci.readWine(filename)
        # self.points, self.label = self.fouci.readSegmentation(filename)
        # self.points, self.label = self.fouci.readDermetology(filename)
        # self.points, self.label = self.fouci.readSeed(filename)
        # self.points, self.label = self.fouci.readZoo(filename)
        # self.points, self.label = self.fouci.readNewthyroid(filename)
        # self.points, self.label = self.fouci.readControl(filename)
        # self.points, self.label = self.fouci.readDigits(filename)
        # self.points, self.label = self.fouci.readWdbc(filename)
        # self.points, self.label = self.fouci.readParkinsons(filename)
        # self.points, self.label = self.fouci.readEcoli(filename)
        # self.points, self.label = self.fouci.readWifi(filename)
        # self.points, self.label = self.fouci.readColoumn(filename)
        # self.points, self.label = self.fouci.readYeast(filename)
        self.points, self.label = self.fouci.readPage(filename)
        # self.points, self.label = self.fouci.readAba(filename)
        # self.points, self.label = self.fouci.readLym(filename)
        # self.points, self.label = self.fouci.readDigits(filename)
        # self.points, self.label = self.fouci.readDivorce(filename)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.points = min_max_scaler.fit_transform(self.points)
        self.size, self.dim = self.points.shape
        self.CL = np.zeros(self.size)
    def Run_Algorithm(self):
        # self.pf.printScatter_Color(self.points,self.label)
        start = time.time()
        dis = self.Get_Distance(self.points)#获取距离矩阵
        Neighbor_K,RNeighbor_K,RNeighbor_len = self.K_Neighbor(dis)#返回每个点的K近邻和逆K近邻
        cluster = 1
        for i in range(self.size):
            if self.CL[i] == 0:
                if len(RNeighbor_K[i]) < self.k:
                    continue
                else:
                    self.CL[i] = cluster
                    self.Expand_Cluster(i,Neighbor_K,RNeighbor_K,cluster,RNeighbor_len)
                    cluster += 1
        Cluster_points,Cluster_core_points = self.Cluster_Points(RNeighbor_K)
        density = self.Cluster_Density(Cluster_core_points,dis)
        self.ExpandClusters(Neighbor_K,RNeighbor_K,density,dis)
        end = time.time()
        print("时间:",end-start)
        # self.pf.printScatter_Color(self.points,self.CL)
        # DCPS = self.Compute_DSPC(Cluster_core_points,dis)
        # DSC = self.Compute_DSC(Cluster_core_points,dis)
        # VC = self.Compute_VC(DCPS,DSC)
        # DBCV = self.Compute_DBCV(VC)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(self.label, self.CL), ari(self.label, self.CL), fmi(self.label, self.CL)))
        # print(DBCV)


    #1.计算距离矩阵
    def Get_Distance(self,points):
        size = len(points)
        dis = np.zeros((self.size,self.size))
        for i in range(self.size):
            for j in range(i+1,size):
                d = np.linalg.norm(points[i] - points[j])
                dis[i,j] = d
                dis[j,i] = d
        return dis

    #2.计算每个点的K近邻和每个点的逆近邻
    def K_Neighbor(self,dis):
        Neighbor_K = []
        RNeighbor_K = []
        RNeighbor_len = np.zeros(self.size)
        for i in range(self.size):
            RNeighbor_K.append([])
        dis_index = np.argsort(dis,axis=1)
        for i in range(self.size):
            temp = []
            for j in range(self.k+1):
                if dis_index[i][j] == i:#遇到自己则不添加
                    continue
                else:
                    RNeighbor_K[dis_index[i][j]].append(i)
                    temp.append(dis_index[i][j])
            Neighbor_K.append(temp)
        for i in range(self.size):
            RNeighbor_len[i] = len(RNeighbor_K[i])
        return Neighbor_K,RNeighbor_K,RNeighbor_len

    #3.扩展聚类
    def Expand_Cluster(self,x,Neighbor_K,RNeighbor_K,cluster,RNeighbor_len):
        visited = np.zeros(self.size)
        visited[x] = 1
        current_cluster = self.Neighborhood(x,Neighbor_K[x],RNeighbor_K[x],RNeighbor_len)#找到从点x开始要扩散的初始点
        for i in current_cluster:
            self.CL[i] = cluster
            visited[i] = 1
        while current_cluster != []:#后来要扩散的的点
            y = current_cluster[0]#这里没给聚类标签是因为在初始群在while循环上面已经给过了，而后来的群是z在下面给过了
            del current_cluster[0]#获取当前聚类队列头并删除
            if RNeighbor_len[y] >= self.k:#若点y是核心点，则找到他的k近邻和逆k近邻中的核心点
                later_cluster = self.Neighborhood(y,Neighbor_K[y],RNeighbor_K[y],RNeighbor_len)
                for z in later_cluster:#若z之前没看成是离群点，即不可能是核心点，则直接进群，否则把它放到队列里，之后再判断是否是核心点
                    if self.CL[z] == 0:
                        if len(RNeighbor_K[y]) >= self.k and visited[z] == 0:
                            visited[z] = 1
                            current_cluster.append(z)
                        self.CL[z] = cluster

    #4.获取从点x开始要扩散的一轮的点（x的K近邻和X的逆K近邻中的核心点
    def Neighborhood(self,x,Neighbor_k,RNeighbor_K,RNeighbor_len):
        neighbors = []
        for i in Neighbor_k:
            neighbors.append(i)
        for i in RNeighbor_K:
            if RNeighbor_len[i] >= self.k and i not in neighbors:
                neighbors.append(i)
        return neighbors

    #5.计算每个初始聚类里包括的点和包括的核心点
    def Cluster_Points(self,RNeighbor_K):
        Cluster_points = []#存放每个类中包含的点
        Cluster_core_points = []#存放每个类中包含的核心点
        maxcluster = int(np.max(self.CL))
        Cluster_core_points.append([])
        Cluster_points.append([])
        for i in range(1,maxcluster+1):
            temp = []
            temp_core = []
            for j in range(self.size):
                if self.CL[j] == i:
                    temp.append(j)
                    if len(RNeighbor_K[j]) >= self.k:
                        temp_core.append(j)
            Cluster_points.append(temp)
            Cluster_core_points.append(temp_core)
        return Cluster_points,Cluster_core_points

    #6.计算每个初始聚类的density(Cluster_core_points是0开始的)
    def Cluster_Density(self,Cluster_core_points,dis):
        maxcluster = int(np.max(self.CL))
        density = np.zeros(maxcluster+1)
        for i in range(1,maxcluster+1):
            max_dist = 0
            for j in Cluster_core_points[i]:
                max_dist = max(max_dist,np.max(dis[j,Cluster_core_points[i]]))
            density[i] = max_dist
        return density

    #7.处理之前没聚到类的点（由于不对称性，因为你把核心点看成是K近邻，但是核心点未必把你看成是K近邻（人家要的即使是逆近邻也要求别人是核心点）
    def ExpandClusters(self,Neighbor_K,RNeighbor_K,density,dis):
        size = len(self.CL)
        for i in range(size):
            if self.CL[i] == 0:
                neighbors = Neighbor_K[i]
                mindist = np.inf
                for j in neighbors:
                    if len(RNeighbor_K[j]) >= self.k:
                        clusterID = int(self.CL[j])#获取之前被看成是噪声点的点的近邻里的核心点
                        if mindist > dis[i,j] and dis[i,j] <= density[clusterID]:
                            mindist = dis[i,j]
                            self.CL[i] = clusterID

    #8.计算DCS(需要最小生成树)（聚类Ci里的核心点组建的最小生成树的最大边）
    def Compute_DSC(self,Cluster_core_points,dis):
        maxcluster = int(np.max(self.CL))
        DSC = np.zeros(maxcluster)
        for i in range(1,maxcluster+1):
            dist = []
            for j in Cluster_core_points[i]:
                temp = []
                for k in Cluster_core_points[i]:
                    temp.append(dis[j,k])
                dist.append(temp)
            UG = np.triu(dist)
            Tree_list,DSC[i-1] = mt.minTree(UG)
        return DSC



    #9.计算DSPC(聚类Ci里的核心点和其他聚类核心点之间的最小距离
    def Compute_DSPC(self,Cluster_core_points,dis):
        maxcluster = int(np.max(self.CL))
        DSPC = np.zeros(maxcluster)
        if maxcluster == 1:
            DSPC[0] = 0
        else:
            for i in range(1,maxcluster+1):
                mindst = []
                for j in Cluster_core_points[i]:
                    for k in range(1,maxcluster+1):
                        if k == i:
                            continue
                        else:
                            mindst.append(np.min(dis[j,Cluster_core_points[k]]))
                    DSPC[i-1] = np.min(mindst)
        return DSPC

    #10.计算VC
    def Compute_VC(self,DSPC,DSC):
        maxcluster = int(np.max(self.CL))
        VC = np.zeros(maxcluster)
        for i in range(1,maxcluster+1):
            VC[i-1] = (DSPC[i-1] - DSC[i-1]) / max(DSPC[i-1],DSC[i-1])
        return VC
    #11.计算DBCV,对于每个聚类的加权累加和
    def Compute_DBCV(self,VC):
        maxcluster = int(np.max(self.CL))
        size = len(self.CL)
        DBCV = 0
        for i in range(1,maxcluster+1):
            C = np.where(self.CL == i)[0].shape[0]
            DBCV += C * VC[i-1]
        return DBCV / size


def main():
    rnn_db = RNN_DBSCAN()
    rnn_db.Run_Algorithm()
if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    main()
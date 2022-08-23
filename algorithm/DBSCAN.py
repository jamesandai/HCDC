import numpy as np
import sys

from sklearn import preprocessing
import time
sys.path.append("..")
from tools.PrintFigures import PrintFigures  # 绘图操作类
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci
from sklearn.metrics import normalized_mutual_info_score, fowlkes_mallows_score, adjusted_rand_score


class DBSCAN:
    fo = FileOperator()
    po = PrintFigures()
    fouci = FileOperatoruci()
 
    # filename = "../datasets/point.txt"

    EPS = 1
    Minpts = 4
    def Run_Algorithm(self):
        points, label = self.fo.readPoint(self.filename)
        #min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        #points = min_max_scaler.fit_transform(points)
        start = time.time()
        dis = self.getDistance(points)
        density,neighbors = self.Compute_density_neighbors(points,dis)
        CL = self.Assigned_points(points,density,neighbors)
        end = time.time()
        print("时间",end-start)
        print(max(CL))
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(label, CL), ari(label, CL), fmi(label, CL)))
        # self.po.printScatter_Color(points,CL)

    #1.计算距离矩阵
    def getDistance(self, points):
        length = points.shape[0]
        dis = np.zeros((length, length))
        for i in range(length):
            for j in range(i + 1, length):
                dd = np.linalg.norm(points[i] - points[j])
                dis[i, j] = dd
                dis[j, i] = dd
        return dis

    #2.计算每个点在EPS邻域里的密度和邻居
    def Compute_density_neighbors(self,points,dis):
        length = len(points)
        density = np.zeros(length)
        neighbors = []
        for i in range(length):
            temp = list(np.where(dis[i]<self.EPS)[0])
            temp.remove(i)#在点的邻居里删除当前点
            neighbors.append(temp)
            density[i] = len(temp)
        return density,neighbors

    def Assigned_points(self,points,density,neighbors):
        length = len(points)
        CL = np.zeros(length)
        for i in range(length):
            if density[i] >= self.Minpts and CL[i] == 0:#核心点
                neighbor = []
                clusterID = np.max(CL) + 1
                CL[i] = clusterID
                for j in neighbors[i]:
                    if CL[j] == 0:
                        neighbor.append(j)
                while len(neighbor) != 0:
                    current = neighbor.pop(-1)
                    CL[current] = clusterID
                    if density[current] >= self.Minpts:
                        for k in neighbors[current]:
                            if CL[k] == 0 and k not in neighbor:
                                neighbor.append(k)
        return CL



def main():
    dbscan = DBSCAN()
    dbscan.Run_Algorithm()
if __name__ == "__main__":
    np.set_printoptions(threshold=np.Inf)
    main()

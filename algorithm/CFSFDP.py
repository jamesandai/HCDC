'''
The CFSFDP algorithm 
Alex Rodriguez et al. 'Clustering by fast search and find of density peaks', Science. 2014.
Created on 2017-9-27
implementation: Jianguo Chen
'''
import math
import sys

from sklearn import preprocessing

sys.path.append("..")
import numpy as np
from tools.PrintFigures import PrintFigures  # 绘图操作类
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci
from sklearn.metrics import normalized_mutual_info_score, fowlkes_mallows_score, adjusted_rand_score
import time
class CFSFDP:
    MAX = 1000000    
    fo = FileOperator()
    fouci = FileOperatoruci()
    pf = PrintFigures()
    filename = "../datasets/point.txt"


    #1 main function of CFSFDP
    def runAlgorithm(self):
        #1) load input data
        points, label = self.fo.readPoint(self.filename)
    
        self.length = len(points)
        self.neigh = np.zeros(self.length)
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # points = min_max_scaler.fit_transform(points)
        fcl = []
        self.size, self.dim = points.shape
        start = time.time()
        dis, dist = self.Get_distance(points)
        percent = 2
        position = int(len(dist) * percent / 100)  # Number of neighbors
        sortedll = np.sort(dist)  #
        self.dc = sortedll[position]  # Get the minimum distance of the neighbor as the cutoff distance
            # dc = 1.1
            # 2) compute rho and delta
        rho = self.getlocalDensity(dis)
        delta = self.computDelta(rho, dis)
        # compute rho density and delta distance
        # self.pf.printRhoDelta(rho,delta)  # print clustering decision graph
        # self.pf.printPolt2(rho)  # print graph for rho
        #3) identify cluster centers
        centers = self.identifyCenters(rho, delta)
        cores = []
        for i in range(self.length):
            if centers[i] != 0:
                cores.append(i)
        #4) assign the remaining points
        result = self.assignDataPoint(dis, rho, centers)
        end = time.time()
        # print(NMI(result, label))
        # print(result)
        # self.pf.print_CFSFDP_line(points,result,self.neigh)
        #print clustering results
        # self.pf.printScatter_Color(points,result)
        print("运行时间:",end-start)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(label, result), ari(label, result), fmi(label, result)))
  
    #2 compute rho density and delta distance


    #3 compute distances among data points
    def Get_distance(self, points):
        dis = np.zeros((self.size, self.size))
        dist = []
        for i in range(self.size):
            for j in range(i + 1, self.size):
                dd = np.linalg.norm(points[i, :] - points[j, :])
                dis[i, j] = dd
                dis[j, i] = dd
                dist.append(dd)
        return dis, dist##ll是距离列表,dist是距离矩阵
    
    
    #4 compute rho density
    def getlocalDensity(self, dist):
        rho = np.zeros((self.length, 1))
        for i in range(self.length-1):
            for j in range(i+1, self.length):
                k = math.exp(-(dist[i][j]/self.dc) ** 2)  #using RBF Kernel function
                rho[i] = rho[i] + k
                rho[j] = rho[j] + k
                if dist[i][j] <= self.dc:
                    rho[i] = rho[i] + 1
                    rho[j] = rho[j] + 1
                # if dist[i,j] <= dc:
                #     rho[i] += 1
                #     rho[j] += 1
        # self.fo.writeData(rho, self.fileurl +  'DPC-rho.csv')  #save rho density
        return rho      
    
    
    #5 compute Delta distance
    def computDelta(self,rho,dist):
        delta = np.ones((self.length, 1)) * self.MAX
        maxDensity = np.max(rho)#找到最大密度的点
        for i in range(self.length):
            if rho[i] < maxDensity:
                for j in range(self.length):
                    if rho[j] > rho[i] and dist[i][j] < delta[i]:
                        delta[i] = dist[i][j]
            else:
                delta[i] = 0.0
                for j in range(self.length):
                    if dist[i][j] > delta[i]:
                        delta[i] = dist[i][j]
        # self.fo.writeData(delta, self.fileurl +  'DPC-delta.csv') #save Delta distance
        return delta


    #6 identify cluster centers
    def identifyCenters(self, rho, delta):
        rate1 = 0.1
        thRho = rate1 * (np.max(rho) - np.min(rho)) + np.min(rho)  #set the parameter threshold of rho density
        #按照(pmax-pmin)*w1+pmin找到密度的阈值
        rate2 = 0.1
        thDel = rate2 * (np.max(delta) - np.min(delta)) + np.min(delta)  #set the parameter threshold of delta distance
        #按照(xmax-xmin)*w2+xmin
        centers = np.zeros(self.length, dtype=np.int)
        cNum = 1
        for i in range(self.length):
            if rho[i] > thRho and delta[i] > thDel:#只有大于thDel和thRho的点才能看成是中心
                centers[i] = cNum
                cNum = cNum + 1
        print("Number of cluster centers: ", cNum-1)
        #self.fo.writeData(centers, self.fileurl + 'centers.csv') 
        return centers        
 
 
    #7 assign the remaining points to the corresponding cluster center
    def assignDataPoint(self, dist,rho, result):
        for i in range(self.length):
            dist[i][i] = self.MAX

        for i in range(self.length):
            if result[i] == 0:
                result[i] = self.nearestNeighbor(i,dist, rho, result)
            else:
                continue
        return result
      
      
    #8 Get the nearest neighbor with higher rho density for each point
    def nearestNeighbor(self, index, dist, rho, result):
        dd = self.MAX
        neighbor = -1
        for i in range(self.length):
            if dist[index, i] < dd and rho[index] < rho[i]:
                dd = dist[index, i]
                neighbor = i
        self.neigh[index] = neighbor
        if result[neighbor] == 0:#若没找到，则递归寻找
            result[neighbor] = self.nearestNeighbor(neighbor, dist, rho, result)
        return result[neighbor]

        
def main():    
    dpc = CFSFDP()
    dpc.runAlgorithm()   #run the main function of CFSFDP 
    
       
if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    main()
# -*- coding:utf-8 -*-

from CURE import *
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score,rand_score,fowlkes_mallows_score,adjusted_rand_score,normalized_mutual_info_score
from scipy.sparse.csgraph._min_spanning_tree import minimum_spanning_tree
import matplotlib.pyplot as plt
sys.path.append("../..")
from tools.PrintFigures import PrintFigures  # 绘图操作类
from tools.FileOperator import FileOperator  # 文件操作类
from tools.FileOperatoruci import FileOperatoruci

fo = FileOperator()
po = PrintFigures()
fouci = FileOperatoruci()
# filename = '../../datasets/ED_Hexagon/dataset.csv'
# filename = '../../datasets/Compound/dataset.csv'#
# filename = "../../datasets/ThreeCircles.txt" #
# filename = "../../datasets/mypoint.txt"#14#
# filename = "../../datasets/point.txt"
# filename = '../../datasets/VDD2.txt'
# filename = '../../datasets/jain.txt'
# filename = "../../datasets/csv/dartboard2.csv"
# filename = "../../datasets/grid.txt"

# filename = "../../uci datasets/zoo.data"  # 0.446,0.044,0.273(0.8 2)
# filename = "../../uci datasets/optdigits.data"#0.769,0.632,0.672(1.22 3)
# filename = "../../uci datasets/wifi"#0.617,0.510,0.627（0.14 11）
# filename = "../../uci datasets/coloumn"#k=3,0.226,0.228,0.536(0.1 3)
filename = "../../uci datasets/yeast.data"#0.074,0.032,0.352(0.1 8)
# filename = "../../uci datasets/Page"#0.002,0.003,0.900（1 4）
# filename = "../../uci datasets/divorce.csv"#0.829,0.886,0.942(2.5 2)
# points, label = fo.readDatawithLabel2(filename)#Flame和R12
# points,label = fo.readDatawithLabel(filename)
# points, label = fo.readPoint(filename)
# points = fo.readDatawithoutLabel(filename)#E6 d6 t4 t7
# points, label = self.fo.readSpiral(filename)
# points = fo.readDatawithoutLabel_t8(filename)
# points, label = fo.readThreeCircle(filename)

# points, label = fouci.readZoo(filename)
# points, label = fouci.readDigits(filename)
# points, label = fouci.readWifi(filename)
# points, label = fouci.readColoumn(filename)
points, label = fouci.readYeast(filename)
# points, label = fouci.readPage(filename)
# points, label = fouci.readDigits(filename)
# points, label = fouci.readDivorce(filename)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
points = min_max_scaler.fit_transform(points)
numDesCluster = 10
for numRepPoints in range(10,45,5):
    for alpha in np.arange(0.2,0.7,0.1):
        start = time.time()
        CL = runCURE(points, numRepPoints, alpha, numDesCluster)
        end = time.time()
        print(numRepPoints,alpha)
        print("The time of CURE algorithm is", end - start, "\n")
        # Compute the NMI
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(label, CL), ari(label, CL), fmi(label, CL)))
        print("\n\n")


import time

from sklearn.datasets import make_moons
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score, \
    adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.metrics import normalized_mutual_info_score,rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import sys
sys.path.append("../..")
import algorithm.LDP_MST.LDPMST_OPT as LPO
from tools.FileOperator import FileOperator  # 文件操作类
from tools.PrintFigures import PrintFigures
from tools.FileOperatoruci import FileOperatoruci
class LDPMST:#要指定聚类数目
    def RunAlgorithm(self,points,label):
        size,dim = points.shape
        # pf.printScatter(points)
        plt.show()
        # pf.printScatter_Color_order(points, label)
        np.set_printoptions(threshold=np.Inf)
        result = LPO.LDPMST_opt(points,clu_num=18)
        # pf.printScatter_Color(points, result)
        # pf.printScatter_Color_order(points, label)
        ami = lambda x, y: adjusted_mutual_info_score(x, y)
        ri = lambda x, y: rand_score(x, y)
        fmi = lambda x, y: fowlkes_mallows_score(x, y)
        nmi = lambda x, y: normalized_mutual_info_score(x, y)
        ari = lambda x, y: adjusted_rand_score(x, y)
        print("%.3f,%.3f,%.3f" % (nmi(label, result), ari(label, result), fmi(label, result)))
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
import LDPMST_OPT as LPO
from tools.FileOperator import FileOperator  # 文件操作类
from tools.PrintFigures import PrintFigures
from tools.FileOperatoruci import FileOperatoruci
if __name__ == "__main__":#要指定聚类数目
    # filename = '../../datasets/ED_Hexagon/dataset.csv'#clu_num=2  nmi = 1
    # filename = '../../datasets/MDDM_D31/dataset.csv'  #clu_num = 31 nmi = 0.963413242344491
    # filename = '../../datasets/MDDM_G2/dataset.csv'#
    # filename = '../../datasets/Aggregation/dataset.csv'#可以
    # filename = '../../datasets/Compound/dataset.csv'# 密度差太大的那个类不行，nmi = 0.9032856744169491
    # filename = '../../datasets/G50/dataset.csv'
    # filename = "../../datasets/E6.txt"#clu_num=7 可以
    # filename = "../../datasets/d6.txt"#clu_num=4 可以
    # filename = "../../datasets/t4.txt"#clu_num=6 可以，但是不能区别噪声
    # filename = "../../datasets/t7.txt"  #clu_num=9 可以
    # filename = "../../datasets/t8.dat"
    # filename = "../../datasets/spiral.txt"#可以，nmi = 1
    # filename = "../../datasets/ThreeCircles.txt" # nmi = 1
    # filename = "../../datasets/Flame.txt"#nmi = 0.8752165990296402
    # filename = "../../datasets/R15.txt"  #nmi = 0.9913351232214995
    # filename = "../../datasets/jain.txt"
    # filename = "../../datasets/mypoint.txt"
    filename = "../../datasets/point.txt"
    # filename = '../../datasets/VDD2.txt'
    # filename = "../../datasets/grid.txt"
    # filename = "../../datasets/csv/dartboard2.csv"##4
    # filename = "../../datasets/csv/disk1.txt"#2

    # filename = "../../uci datasets/iris.data"#k=3,0.784,0.720,0.816
    # filename = "../../uci datasets/wine.data"#k=3 0.575,0.438,0.698
    # filename = "../../uci datasets/new-thyroid.data" #k=3 0.411,0.367,0.791
    # filename = "../../uci datasets/segmentation.data"#k=7,0.559,0.212,0.459
    # filename = "../../uci datasets/dermatology.data"#k=6 0.826,0.721,0.782
    # filename = "../../uci datasets/seeds_dataset.txt" # k = 3,0.528,0.404,0.655
    # filename = "../../uci datasets/zoo.data"#k=7,0.755,0.797,0.852
    # filename = "../../uci datasets/optdigits.data"#k=10,0.831,0.746,0.776
    # filename = "../../uci datasets/new-thyroid.data"#k=3,0.411,0.367,0.791
    # filename = "../../uci datasets/parkinsons.data"#k=2,0.039,-0.056,0.737
    # filename = "../../uci datasets/ecoli.data"  # k=8,0.534,0.331,0.481
    # filename = "../../uci datasets/wifi"#k=4,0.802,0.695,0.794
    # filename = "../../uci datasets/coloumn"#k=3,0.274,0.334,0.616
    # filename = "../../uci datasets/yeast.data"#k=10,0.305,0.211,0.441
    # filename = "../../uci datasets/Page"#k=5,0.027,0.072,0.887
    # filename = "../../uci datasets/divorce.csv"#k=2,0.862,0.908,0.954
    # filename = "../../uci datasets/lymphography.data"  # k=7,0.150,0.070,0.352

    fo = FileOperator()
    pf = PrintFigures()
    fouci = FileOperatoruci()
    # points, label = fo.readDatawithLabel2(filename)  # Flame和R15
    # points, label = fo.readDatawithLabel(filename)
    points, label = fo.readPoint(filename)
    # points = fo.readS1(filename)
    # points = fo.readDatawithoutLabel(filename)  # E6 d6 t4 t7
    # points = fo.readDatawithoutLabel_t8(filename)
    # points, label = fo.readSpiral(filename)
    # points, label = fo.readThreeCircle(filename)


    # points, label = fouci.readIris(filename)
    # points, label = fouci.readWine(filename)
    # points, label = fouci.readSegmentation(filename)
    # points, label = fouci.readDermetology(filename)
    # points,label = fouci.readSeed(filename)
    # points, label = fouci.readZoo(filename)
    # points, label = fouci.readNewthyroid(filename)
    # points,label = fouci.readControl(filename)
    # points, label = fouci.readDigits(filename)
    # points, label = fouci.readWdbc(filename)
    # points, label = fouci.readParkinsons(filename)
    # points, label = fouci.readWifi(filename)
    # points, label = fouci.readColoumn(filename)
    # points, label = fouci.readEcoli(filename)
    # points, label = fouci.readYeast(filename)
    # points, label = fouci.readPage(filename)
    # points, label = fouci.readDivorce(filename)
    # points, label = fouci.readAba(filename)
    # points, label = fouci.readLym(filename)
    size,dim = points.shape
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # points = min_max_scaler.fit_transform(points)
    # pf.printScatter(points)
    plt.show()
    # pf.printScatter_Color_order(points, label)
    np.set_printoptions(threshold=np.Inf)
    start = time.time()
    result = LPO.LDPMST_opt(points,clu_num=5)
    end = time.time()
    print("用时:",end-start)
    # pf.printScatter_Color(points, result)
    # pf.printScatter_Color_order(points, label)
    ami = lambda x, y: adjusted_mutual_info_score(x, y)
    ri = lambda x, y: rand_score(x, y)
    fmi = lambda x, y: fowlkes_mallows_score(x, y)
    nmi = lambda x, y: normalized_mutual_info_score(x, y)
    ari = lambda x, y: adjusted_rand_score(x, y)
    print("%.3f,%.3f,%.3f" % (nmi(label, result), ari(label, result), fmi(label, result)))



# if __name__ == "__main__":
#     print("iris")
#     np.set_printoptions(threshold=np.inf)
#     filename = '../../uci datasets/iris.data'
#     data = []
#     label = []
#     name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#     with open(filename, 'r') as f:
#         while True:
#             m = 0
#             data_temp = []
#             lines = f.readline().strip('\n')
#             if not lines:
#                 f.close()
#                 break
#             else:
#                 for i in lines.split(','):
#                     m = m + 1
#                     if m == 5:
#                         label.append(name.index(i))
#                     else:
#                         data_temp.append(float(i))
#             data.append(data_temp)
#     data = np.array(data)
#     label = np.array(label)
#     size,dim = data.shape
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data1 = min_max_scaler.fit_transform(data)
#     # data2 = PCA(n_components=3,svd_solver='full').fit_transform(data)
#     C = lambda x, y: normalized_mutual_info_score(x, y, average_method='geometric')
#     RI = lambda x, y: rand_score(x, y)
#     result = LPO.LDPMST_opt(data,clu_num=3)
#     print(result)
#     print("NMI指标为:%f" % C(label, result))
#     sum_right = 0
#     for i in range(size):
#         if label[i]+1 == result[i]:
#             sum_right += 1
#     print("准确率为" ,(sum_right / size))
# if __name__ == "__main__":
#     print("wine")
#     start = time.time()
#     np.set_printoptions(threshold=np.inf)
#     # data = datasets.load_wine()['data']
#     # data = np.array(data)
#     # label = datasets.load_wine()['target']
#     filename = '../../uci datasets/wine.data'
#     data = []
#     label = []
#     with open(filename, 'r') as f:
#         while True:
#             m = 0
#             data_temp = []
#             lines = f.readline().strip('\n')
#             if not lines:
#                 f.close()
#                 break
#             else:
#                 for i in lines.split(','):
#                     m = m + 1
#                     if m == 1:
#                         label.append(float(i))
#                     else:
#                         data_temp.append(float(i))
#             data.append(data_temp)
#     data = np.array(data)
#     label = np.array(label)
#     size,dim = data.shape
#     X = StandardScaler().fit(data).transform(data)
#     pca = PCA(n_components=3)
#     data_new = pca.fit_transform(X)
#     plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
#     plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
#     plt.ylabel('Explained variance ratio')
#     plt.xlabel('Principal components')
#     plt.show()
#
#     C = lambda x, y: normalized_mutual_info_score(x, y, average_method='geometric')
#     size = data.shape[0]
#     label = np.array(label).astype('float')
#     result = LPO.LDPMST_opt(data,clu_num=3)
#     print(label)
#     print(result)
#     sum_right = 0
#     for i in range(size):
#         if label[i] == result[i]:
#             sum_right += 1
#     print("准确率为", (sum_right / size))
#     end = time.time()
#     print("NMI指标为:%f" % C(label,result))
# if __name__ == "__main__":
#     np.set_printoptions(threshold=np.inf)
#     df_wine = pd.read_csv('./uci datasets/data/wine.data', header=None)
#     wine = df_wine.values
#     data = wine[:,1:14]
#     label = wine[:,0]
#     sc = StandardScaler()
#     x = sc.fit_transform(data)
#     cov_matrix = np.cov(x.T)
#     eigen_val,eigen_vec = np.linalg.eig(cov_matrix)
#     tol = sum(eigen_val)
#     var_exp = [(i/tol) for i in sorted(eigen_val,reverse=True)]
#     cum_var_exp = np.cumsum(var_exp)
#     cum_var_exp = np.cumsum(var_exp)  # 累加方差比率
#     # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
#     # plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='独立解释方差')  # 柱状 Individual_explained_variance
#     # plt.step(range(1, 14), cum_var_exp, where='mid', label='累加解释方差')  # Cumulative_explained_variance
#     # plt.ylabel("解释方差率")
#     # plt.xlabel("主成分索引")
#     # plt.legend(loc='right')
#     # plt.show()
#     # print(x)
#     eigen_pairs = [(np.abs(eigen_val[i]), eigen_vec[:, i]) for i in range(len(eigen_val))]
#     eigen_pairs.sort(key=lambda k: k[0], reverse=True)
#     w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis],eigen_pairs[2][1][:, np.newaxis]))
#     data_new = x.dot(w)
#     C = lambda x, y: normalized_mutual_info_score(x, y)
#     label_result = LPO.LDPMST_opt(data_new,3)
#     print(label_result)
#     sum_right = 0
#     size,dim = data_new.shape
#     for i in range(size):
#         if label[i] == label_result[i]:
#             sum_right += 1
#     print("准确率为", (sum_right / size))
#     print("NMI指标为:%f"%C(label, label_result))
# if __name__ == "__main__":#14
#     print("seg")
#     np.set_printoptions(threshold=np.inf)
#     filename = './uci datasets/data/segmentation.data'
#     data = []
#     label = []
#     name = ['BRICKFACE','SKY','FOLIAGE','CEMENT','WINDOW','PATH','GRASS']
#     with open(filename,'r') as f:
#         while True:
#             m = 0
#             data_temp = []
#             lines = f.readline()
#             if not lines:
#                 f.close()
#                 break
#             else:
#                 for i in lines.split(','):
#                     m = m + 1
#                     if m == 1:
#                         label.append(name.index(i))
#                     else:
#                         data_temp.append(float(i))
#             data.append(data_temp)
#     data = np.array(data)
#     size,dim = data.shape
#     X = StandardScaler().fit(data).transform(data)
#     pca = PCA(n_components=7)
#     data_new = pca.fit_transform(X)

    # plt.bar(range(19), pca.explained_variance_ratio_, alpha=0.5, align='center')
    # plt.step(range(19), np.cumsum(pca.explained_variance_ratio_), where='mid')
    # plt.ylabel('Explained variance ratio')
    # plt.xlabel('Principal components')
    # plt.show()

    # C = lambda x, y: normalized_mutual_info_score(x, y, average_method='geometric')
    # size = data.shape[0]
    # label = np.array(label).astype('float')
    # result = LPO.LDPMST_opt(data_new,clu_num=7)
    # print(label)
    # print(result)
    # sum_right = 0
    # for i in range(size):
    #     if label[i]+1 == result[i]:
    #         sum_right += 1
    # print("准确率为", (sum_right / size))
    # end = time.time()
    # print("NMI指标为:%f" % C(label,result))

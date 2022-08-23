import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append("..")
from algorithm.Dcore import Dcore
from algorithm.DPC_face import CFSFDP
from algorithm.HORC.my4 import my4 as HORC
from algorithm.LDP_MST.LDPMST import LDPMST
from algorithm.RNN_face import RNN_DBSCAN
from algorithm.KDPC_face import KDPC
from algorithm.SNNDPC.SNNDPC_face import SNNDPC_Face
def plt_show(img):
    plt.imshow(img,cmap='gray')
    plt_show()

def read_directory(directory_name):#返回那个路径的所有文件,每一个文件夹下的pm文件
    faces_addr = []
    for file_name in os.listdir(directory_name):
        faces_addr.append(directory_name + "/" + file_name)
    return faces_addr

def load_faces_data(path):
    faces = []
    for i in range(1,41):#获取每个文件夹下的图片的路径，总共有400个路径放在faces里
        file_addr = read_directory(path + "/s" + str(i))
        for addr in file_addr:
            faces.append(addr)
    images = []
    label = []
    for index,face in enumerate(faces):
        image = cv2.imread(face,0)
        images.append(image)
        label.append(int(index/10+1))
    return images,label
def plt_show(img):
    plt.imshow(img,cmap='gray')
    plt.show()
def plot_image(images):
    fig,axes = plt.subplots(10,10,figsize=(30,30),subplot_kw={"xticks":[],"yticks":[]})
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[int(i/10)*10+i%10],cmap="gray")
    plt.show()
def PCA_images(images,label):
    image_data = []
    label_ = []
    #95%
    #100,60是0.954,0.868,0.886,LDP是0.871,0.741,0.773,RNN是0.876,0.716,0.758(5)，DPC是0.630,0.051,0.111,KDPC是0.940,0.836,0.855,SNNDPC是0.871,0.737,0.766(5),
    # Dcore是0.888,0.759,0.784( r1=0.9,r2=0.81,r=0.92,t1=1,t2=0)
    #200，111，HORC是0.909,0.746,0.762，LDP是0.845,0.568,0.612(只有18类),RNN是0.720,0.217,0.383(5)，DPC是0.685,0.073,0.108，KDPC是0.895,0.636,0.676，SNNDPC是0.822,0.504,0.557(5)
    # Dcore是0.858,0.636,0.657( r1=0.9,r2=0.81,r=0.88,t1=1,t2=0)
    for i in range(200):
        data = images[int(i/10)*10+i%10].flatten()#把100*112*92的三维数组变成100*10304的二维数组
        label_.append(label[int(i/10)*10+i%10])
        image_data.append(data)
    X = np.array(image_data)#每个图像数据降到一维后的列表
    data = pd.DataFrame(X)#打印的话，X可以显示列和行号的
    # pca = PCA(n_components=111)
    pca = PCA(n_components=60)
    pca.fit(X)
    PCA_data = pca.transform(X)
    expained_variance_ratio = pca.explained_variance_ratio_.sum()#计算保留原始数据的多少

    # 看降到i维的保留原始数据的曲线图
    # expained_variance_ratio = []
    # for i in range(1,200):
    #     pca = PCA(n_components=i).fit(X)#构建pca降维器
    #     expained_variance_ratio.append(pca.explained_variance_ratio_.sum())#计算每次降维，所带的数据是原始数据的多少
    #     print(i,pca.explained_variance_ratio_.sum())
    # plt.plot(range(1,160),expained_variance_ratio)
    # plt.show()
    # V = pca.components_#pca中间的转换矩阵，让10304*100转换成100*98的矩阵

    return PCA_data,label_

def main():
    path ="../datasets/olivetti"
    images,label = load_faces_data(path)
    print(images)
    cv2.imshow('change_image',images)
    cv2.waitkey(0)
    # data,label_ = PCA_images(images,label)
    # dcore = Dcore()
    # dcore.RunAlgorithm(data,label_)
    # horc = HORC()
    # horc.RunAlgorithm(data,label_)
    # ldpmst = LDPMST()
    # ldpmst.RunAlgorithm(data,label_)
    # dpc = CFSFDP()
    # dpc.runAlgorithm(data,label_)
    # rnn = RNN_DBSCAN(5,data,label_)
    # rnn.Run_Algorithm()
    # kdpc = KDPC()
    # kdpc.Run_Algorithm(data,label_)
    # snndpc= SNNDPC_Face()
    # snndpc.Run_Algorithm(data,label_,20)
if __name__ == "__main__":
    main()
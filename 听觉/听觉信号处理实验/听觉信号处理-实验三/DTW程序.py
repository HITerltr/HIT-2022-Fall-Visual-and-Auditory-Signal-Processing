# -*- coding: gbk -*-

import numpy as np
from struct import unpack
from itertools import product
import matplotlib.pylab as plt

def get_MFCC():
    MFCC = []
    for i in range(5):
        MFCC_row = []
        for j in range(10):
            f = open("语料\\" + str(i + 1) + "_" + str(j + 1) + ".mfc", "rb")
            nframes = unpack(">i", f.read(4))[0]# 帧数
            nbytes = unpack(">h", f.read(2))[0]# MFCC特征的字节数
            data = []
            ndim = nbytes / 4# MFCC特征的维数，每个特征大小为4字节
            print(nbytes, ndim, nframes)
            for m in range(nframes):
                data_frame = []
                for n in range(int(ndim)):
                    data_frame.append(unpack(">f", f.read(4))[0])
                data.append(data_frame)
            f.close()
            MFCC_row.append(data)
        MFCC.append(MFCC_row)
    return MFCC

def get_MFCC_model_test(MFCC):
    model = []
    test = []
    for i in range(5):
        test_row = []
        model.append(MFCC[i][0])
        for j in range(1, 10):
            test_row.append(MFCC[i][j])
        test.append(test_row)
    return model, test

def voice_Recognition(model, test):
    row = len(test)
    col = len(test[0])
    cnt = 0# 正确的语料文件个数
    flag_r = np.zeros((5, 9))# 待测试集的标签矩阵
    for i in range(row):
        flag_l = i + 1
        for j in range(col):
            print("现在正在处理第" + str(i * col + 1 + j) + "个语料音频文件；")
            flag_r[i][j] = 1
            min_dis = DTW(test[i][j], model[0])
            for m in range(len(model)):
                dis = DTW(test[i][j], model[m])
                if dis < min_dis:
                    flag_r[i][j] = m + 1
                    min_dis = dis
            if flag_r[i][j] == flag_l:
                cnt += 1
    print("待测试集的标签各为：")
    print(flag_r)
    print("正确率为：")
    print(cnt / (row * col) * 100,"%")

def DTW(x_1, x_2):
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    row = len(x_1)
    col = len(x_2)
    ndim = len(x_1[0])
    D = np.zeros((row, col))
    for i in range(row):# 初始化到点（i，j）的路径代价
        for j in range(col):
            D[i][j] = 0
            D[i][j] = np.linalg.norm(x_1[i] - x_2[j])# 计算欧式距离
    for m in range(ndim):
        D[i][j] += abs(x_1[i][m] - x_2[j][m])# 计算每一个差值的绝对值，最后求和
    for n in range(1, row):# 计算每一列代价路径的累计距离
        D[n][0] += D[n - 1][0]
    for k in range(1, col):# 计算每一行代价路径的累计距离
        D[0][k] += D[0][k - 1]
    for a in range(1, row):# 开始搜索最短代价的路径
        for b in range(1, col):
            D[a][b] = min(D[a - 1][b] + D[a][b], D[a - 1][b - 1] + 2 * D[a][b], D[a][b - 1] + D[a][b])
    return D[row - 1][col - 1] / (row + col - 2)# 计算结果用系数和来进行归正
    return D[row - 1][col - 1]

MFCC = get_MFCC()
model, test = get_MFCC_model_test(MFCC)
voice_Recognition(model, test)


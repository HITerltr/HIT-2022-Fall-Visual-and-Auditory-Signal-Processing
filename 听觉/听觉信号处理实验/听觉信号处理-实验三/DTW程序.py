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
            f = open("����\\" + str(i + 1) + "_" + str(j + 1) + ".mfc", "rb")
            nframes = unpack(">i", f.read(4))[0]# ֡��
            nbytes = unpack(">h", f.read(2))[0]# MFCC�������ֽ���
            data = []
            ndim = nbytes / 4# MFCC������ά����ÿ��������СΪ4�ֽ�
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
    cnt = 0# ��ȷ�������ļ�����
    flag_r = np.zeros((5, 9))# �����Լ��ı�ǩ����
    for i in range(row):
        flag_l = i + 1
        for j in range(col):
            print("�������ڴ����" + str(i * col + 1 + j) + "��������Ƶ�ļ���")
            flag_r[i][j] = 1
            min_dis = DTW(test[i][j], model[0])
            for m in range(len(model)):
                dis = DTW(test[i][j], model[m])
                if dis < min_dis:
                    flag_r[i][j] = m + 1
                    min_dis = dis
            if flag_r[i][j] == flag_l:
                cnt += 1
    print("�����Լ��ı�ǩ��Ϊ��")
    print(flag_r)
    print("��ȷ��Ϊ��")
    print(cnt / (row * col) * 100,"%")

def DTW(x_1, x_2):
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    row = len(x_1)
    col = len(x_2)
    ndim = len(x_1[0])
    D = np.zeros((row, col))
    for i in range(row):# ��ʼ�����㣨i��j����·������
        for j in range(col):
            D[i][j] = 0
            D[i][j] = np.linalg.norm(x_1[i] - x_2[j])# ����ŷʽ����
    for m in range(ndim):
        D[i][j] += abs(x_1[i][m] - x_2[j][m])# ����ÿһ����ֵ�ľ���ֵ��������
    for n in range(1, row):# ����ÿһ�д���·�����ۼƾ���
        D[n][0] += D[n - 1][0]
    for k in range(1, col):# ����ÿһ�д���·�����ۼƾ���
        D[0][k] += D[0][k - 1]
    for a in range(1, row):# ��ʼ������̴��۵�·��
        for b in range(1, col):
            D[a][b] = min(D[a - 1][b] + D[a][b], D[a - 1][b - 1] + 2 * D[a][b], D[a][b - 1] + D[a][b])
    return D[row - 1][col - 1] / (row + col - 2)# ��������ϵ���������й���
    return D[row - 1][col - 1]

MFCC = get_MFCC()
model, test = get_MFCC_model_test(MFCC)
voice_Recognition(model, test)


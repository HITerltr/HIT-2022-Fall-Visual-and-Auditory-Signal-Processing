import numpy as np
from numpy import ndarray
import random

# 添加高斯噪声
def gauss_noise(img: ndarray) -> ndarray:
    img_noise = np.random.normal(0, 0.1, img.shape) #使用随机函数来产生0到1范围内的高斯分布
    img_out = img + img_noise #将产生的噪声和原始图像叠加
    #进行底值限制
    if img_out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    img_out = np.clip(img_out, low_clip, 1.0)
    return img_out


# 添加椒盐噪声
def sp_noise(img: ndarray, sp_number = 0.05) -> ndarray: #固定一个椒盐噪声比例系数将其设置为0.05
    img_out = img
    #对每个像素的值从高度到宽度均选择生成一个0到1范围内的随机值
    h = img_out.shape[0] #图像高度
    w = img_out.shape[1] #图像宽度
    num = int(h * w * sp_number)
    for i in range(num):
        #返回一个随机整型数进行黑白点噪声的添加
        w1 = random.randint(0, w - 1)  
        h1 = random.randint(0, h - 1)
        if random.randint(0,1) == 0:
            img_out[h1,w1] = 0         
        else:
            img_out[h1,w1] = 1   
    return img_out


# 实现中值滤波
def median_filter(img: ndarray) -> ndarray:
    #需要自己设置窗口位
    h = img.shape[0] #图像高度
    w = img.shape[1] #图像宽度
    medianfilter = []
    #选择一个3*3的窗口位
    for i in range(h - 2):
        for j in range(w - 2):
            for k in range(i, i + 3):
                for l in range(j, j + 3):
                    medianfilter.append(img[k, l])
            MedNum = np.median(medianfilter) #调用中值函数求取中间像素点的中值
            img[i + 1, j + 1] = MedNum
            medianfilter.clear() #清除编译脚本，释放内存
    return img


# 实现均值滤波
def means_filter(img: ndarray) -> ndarray:
    #需要自己设置窗口位
    h = img.shape[0] #图像高度
    w = img.shape[1] #图像宽度
    meansfilter = []
    #选择一个3*3的窗口位
    for i in range(h - 2):
        for j in range(w - 2):
            for k in range(i, i + 3):
                for l in range(j, j + 3):
                    meansfilter.append(img[k, l])
            MeaNum = np.mean(meansfilter) #调用均值函数求取中间像素点的均值
            img[i + 1, j + 1] = MeaNum
            meansfilter.clear() #清除编译脚本，释放内存
    return img


# 实现直方图均衡化
def equalize(img: ndarray) -> ndarray:
    img_out = img.copy()
    img_out *= 250.0
    img_out = img_out.astype(np.uint8) #所输入图像值并非整型数，将其强制转化为unit8格式分布
    a = np.max(img_out) #求取最大值
    b = np.min(img_out) #求取最小值
    h = img_out.shape[0] #图像高度
    w = img_out.shape[1] #图像宽度
    hist = np.zeros(256) #返回一个给定形状和类型的用0填充的数组
    #统计各个像素点出现频率
    for i in range(h):
        for j in range(w):
            hist[img_out[i, j]] += 1
    hist = hist/ (h * w)
    hist = np.cumsum(hist) #计算累计分布函数，即积分
    #计算映射函数，最后得到256维数组
    for i in range(h):
        for j in range(w):
            img_out[i, j] = hist[img_out[i, j]] * (a - b) + b
    return img_out


# 实现直方图归一化
def normalize(img: ndarray) -> ndarray:
    img_out = img.copy()
    img_out *= 250.0
    img_out = img_out.astype(np.uint8) #所输入图像值并非整型数，将其强制转化为unit8格式分布
    a = np.max(img_out) #求取最大值
    b = np.min(img_out) #求取最小值
    h = img_out.shape[0] #图像高度
    w = img_out.shape[1] #图像宽度
    #统计各个像素点出现频率
    for i in range(h):
        for j in range(w):
            img_out[i, j] = (255 / (a - b)) * (img_out[i, j] - b) #实现线性对应关系
    return img_out


# 实现膨胀运算
def dilate(img: ndarray) -> ndarray:
    img_out = img.copy()
    a = 0
    h = img.shape[0] #图像高度
    w = img.shape[1] #图像宽度
    #将3*3的窗口位中的每一个像素点进行检查
    for i in range (1, h - 1):
        for j in range(1, w - 1):
            #将输入的图像像素点矩阵进行二值化处理
            if img[i, j] <= 0.5:
                img[i, j] = 0
            else:
                img[i, j] = 1
    for i in range (1, h - 1):
        for j in range(1, w - 1):
            for k in range(-1, 2):
                for l in range(-1, 2):
                    #若有白色像素点，便将其中心像素设置为白色
                    if img[i + k, j + l] == 1:
                        a += 1
            if a != 0:
                img_out[i,j] = 1
                a = 0
    return img_out


# 实现腐蚀运算
def erode(img: ndarray) -> ndarray:
    img_out = img.copy()
    a = 0
    h = img.shape[0] #图像高度
    w = img.shape[1] #图像宽度
    #将3*3的窗口位中的每一个像素点进行检查
    for i in range (1, h - 1):
        for j in range(1, w - 1):
            #将输入的图像像素点矩阵进行二值化处理
            if img[i, j] <= 0.5:
                img[i, j] = 0
            else:
                img[i, j] = 1
    for i in range (1, h - 1):
        for j in range(1, w - 1):
            for k in range(-1, 2):
                for l in range(-1, 2):
                     #若有黑色像素点，便将其中心像素设置为黑色
                    if img[i + k, j + l] == 0:
                        a += 1
            if a != 0:
                img_out[i, j] = 0
                a = 0
    return img_out
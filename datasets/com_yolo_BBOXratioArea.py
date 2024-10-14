# -*- coding: utf-8 -*-
"""
Compute SAR-SHIPdataset bbox
"""


# 1、统计数据集中小、中、大 GT的个数
# 2、统计某个类别小、中、大 GT的个数
# 3、统计数据集中ss、sm、sl GT的个数
import os
from pathlib import Path
import matplotlib.pyplot as plt

import xml.etree.cElementTree as et
import numpy as np

from tqdm import tqdm
# 设置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体为黑体
plt.rcParams['font.family'] = 'sans-serif'  # 设置字体样式
plt.rcParams['figure.figsize'] = (10.0, 10.0)  # 设置字体大小


def getGtAreaAndRatio(label_dir):
    """
    得到不同尺度的gt框个数
    :params label_dir: label文件地址
    :return data_dict: {dict: 3}  3 x {'类别':{’area':[...]}, {'ratio':[...]}}
    """
    data_dict = {}
    bbox_width_list = []
    bbox_height_list = []
    assert Path(label_dir).is_dir(), "label_dir is not exist"

    txts = os.listdir(label_dir)  # 得到label_dir目录下的所有txt GT文件

    for txt in txts:  # 遍历每一个txt文件
        with open(os.path.join(label_dir, txt), 'r') as f:  # 打开当前txt文件 并读取所有行的数据
            lines = f.readlines()

        for line in lines:  # 遍历当前txt文件中每一行的数据
            temp = line.split()  # str to list{5}
            coor_list = list(map(lambda x: x, temp[1:]))  # [x, y, w, h]


            bbox_width = float(coor_list[2]) * 256
            bbox_height = float(coor_list[3]) * 256
            bbox_width_list.append(bbox_width)
            bbox_height_list.append(bbox_height)


            area = float(coor_list[2]) * float(coor_list[3]) *256*256 # 计算出当前txt文件中每一个gt的面积
            # center = (int(coor_list[0] + 0.5*coor_list[2]),
            #           int(coor_list[1] + 0.5*coor_list[3]))
            ratio = round(float(coor_list[2]) / float(coor_list[3]), 2)  # 计算出当前txt文件中每一个gt的 w/h

            if temp[0] not in data_dict:
                data_dict[temp[0]] = {}
                data_dict[temp[0]]['area'] = []
                data_dict[temp[0]]['ratio'] = []

            data_dict[temp[0]]['area'].append(area)
            data_dict[temp[0]]['ratio'].append(ratio)

    return data_dict,bbox_width_list,bbox_height_list


def getSMLGtNumByClass(data_dict, class_num):
    """
    计算某个类别的小物体、中物体、大物体的个数
    params data_dict: {dict: 3}  3 x {'类别':{’area':[...]}, {'ratio':[...]}}
    params class_num: 类别  0, 1, 2
    return s: 该类别小物体的个数  0 < area <= 0.5%
           m: 该类别中物体的个数  0.5% < area <= 1%
           l: 该类别大物体的个数  area > 1%
    """
    s, m, l = 0, 0, 0
    # 图片的尺寸大小 注意修改!!!
    h = 960
    w = 540
    for item in data_dict['{}'.format(class_num)]['area']:
        if item * h * w <= h * w * 0.005:
            s += 1
        elif item * h * w <= h * w * 0.010:
            m += 1
        else:
            l += 1
    return s, m, l


def getAllSMLGtNum(data_dict, isEachClass=False):
    """
    数据集所有类别小、中、大GT分布情况
    isEachClass 控制是否按每个类别输出结构
    """
    S, M, L = 0, 0, 0
    # 需要手动初始化下，有多少个类别就需要写多个
    classDict = {'0': {'S': 0, 'M': 0, 'L': 0}, '1': {'S': 0, 'M': 0, 'L': 0}, '2': {'S': 0, 'M': 0, 'L': 0},
                 '3': {'S': 0, 'M': 0, 'L': 0}}

    print(classDict['0']['S'])
    # range(class_num)类别数 注意修改!!!
    if isEachClass == False:
        for i in range(4):
            s, m, l = getSMLGtNumByClass(data_dict, i)
            S += s
            M += m
            L += l
        return [S, M, L]
    else:
        for i in range(4):
            S = 0
            M = 0
            L = 0
            s, m, l = getSMLGtNumByClass(data_dict, i)
            S += s
            M += m
            L += l
            classDict[str(i)]['S'] = S
            classDict[str(i)]['M'] = M
            classDict[str(i)]['L'] = L
        return classDict


# 画图函数
def plotAllSML(SML):
    x = ['S:[0, 32x32]', 'M:[32x32, 96x96]', 'L:[96*96, 640x640]']
    fig = plt.figure(figsize=(10, 8))  # 画布大小和像素密度
    plt.bar(x, SML, width=0.5, align="center", color=['skyblue', 'orange', 'green'])
    for a, b, i in zip(x, SML, range(len(x))):  # zip 函数
        plt.text(a, b + 0.01, "%d" % int(SML[i]), ha='center', fontsize=15, color="r")  # plt.text 函数
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('gt大小', fontsize=16)
    plt.ylabel('数量', fontsize=16)
    plt.title('广佛手病虫害训练集小、中、大GT分布情况(640x640)', fontsize=16)
    plt.show()
    # 保存到本地
    # plt.savefig("")


if __name__ == '__main__':
    labeldir = r'D:\softapp\Anaconda\envs\pytorch_gpu\datasets\SARship_dataset_v0_label'
    data_dict,bbox_width_list,bbox_height_list = getGtAreaAndRatio(labeldir)

    #
    # ratio_list = data_dict['0']['ratio']
    # ratio_array = np.array(ratio_list)
    # ratio_max = np.max(ratio_array)
    # ratio_min = np.min(ratio_array)
    # ratio_mean = np.mean(ratio_array)
    # ratio_var = np.var(ratio_array)
    # plt.figure(1)
    # plt.hist(ratio_array, 20)
    # plt.xlabel('Ratio of width/height')
    # plt.ylabel('Frequency of ratio')
    # plt.title('Ratio\n' \
    #           + 'max=' + str(round(ratio_max, 2)) + ', min=' + str(round(ratio_min, 2)) + '\n' \
    #           + 'mean=' + str(round(ratio_mean, 2)) + ', var=' + str(round(ratio_var, 2))
    #           )
    #
    # plt.show()
    #
    bboxInimgArea_list = data_dict['0']['ratio']
    bboxInimgArea_array = np.array(bboxInimgArea_list)
    bboxInimgArea_max = np.max(bboxInimgArea_array)
    bboxInimgArea_min = np.min(bboxInimgArea_array)
    bboxInimgArea_mean = np.mean(bboxInimgArea_array)
    bboxInimgArea_var = np.var(bboxInimgArea_array)
    plt.figure(2)
    plt.hist(bboxInimgArea_array, 20)
    plt.xlabel('Ratio of bbox')
    plt.ylabel('Frequency of ratio')
    plt.title('Ratio\n' \
              + 'max=' + str(round(bboxInimgArea_max, 2)) + ', min=' + str(round(bboxInimgArea_min, 2)) + '\n' \
              + 'mean=' + str(round(bboxInimgArea_mean, 2)) + ', var=' + str(round(bboxInimgArea_var, 2))
              )

    plt.show()

    bbox_width_array = np.array(bbox_width_list)
    bbox_height_array = np.array(bbox_height_list)
    plt.figure(1)
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.scatter(bbox_width_array, bbox_height_array,
                c='blue',
                label='BBOX size')

    plt.tick_params(labelsize=30)
    plt.xlabel('bbox_width/m', fontsize=30)
    plt.ylabel('bbox_height/m', fontsize=30)
    plt.title('SAR_SHIPdataset_BBOX_size', fontsize=25
              )
    plt.show()

    # 画出bbox/img
    square_array = np.array( data_dict['0']['area'])
    square_max = np.max(square_array)
    square_min = np.min(square_array)
    square_mean = np.mean(square_array)
    square_var = np.var(square_array)
    plt.figure(1)
    plt.hist(square_array, 20)
    plt.xlabel('Area in pixel')
    plt.ylabel('Frequency of area')
    plt.title('Area\n' \
              + 'max=' + str(square_max) + ', min=' + str(square_min) + '\n' \
              + 'mean=' + str(int(square_mean)) + ', var=' + str(int(square_var))
              )
    plt.show()

    # 1、数据集所有类别小、中、大GT分布情况
    # 控制是否按每个类别输出结构
    # isEachClass = False
    # SML = getAllSMLGtNum(data_dict, isEachClass)
    # print(SML)
    # if not isEachClass:
    #     plotAllSML(SML)


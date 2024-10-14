# -*- coding: utf-8 -*-
"""
Compute SAR-AIRcarft1.0 bbox
"""

import os
import xml.etree.cElementTree as et
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体为黑体
plt.rcParams['font.family'] = 'sans-serif'  # 设置字体样式
plt.rcParams['figure.figsize'] = (10.0, 10.0)  # 设置字体大小


path = "D:/softapp/Anaconda/envs/pytorch_gpu/datasets/SARAIRcraft_Annotations"
files = os.listdir(path)

area_list = []
ratio_list = []
bboxInimgArea_list = []
bbox_width_list = []
bbox_height_list = []


def file_extension(path):
    return os.path.splitext(path)[1]


for xmlFile in tqdm(files, desc='Processing'):
    if not os.path.isdir(xmlFile):
        if file_extension(xmlFile) == '.xml':
            tree = et.parse(os.path.join(path, xmlFile))
            root = tree.getroot()
            # filename = root.find('filename').text
            # print("--Filename is", xmlFile)
            for Size in root.findall('size'):
                width = Size.find('width').text
                height= Size.find('height').text
                imgArea = int(width) * int(height)
                # imgArea_list.append(imgArea)

            for Object in root.findall('object'):
                bndbox = Object.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text

                bbox_width = int(xmax) - int(xmin)
                bbox_height = int(ymax) - int(ymin)
                bbox_width_list.append(bbox_width)
                bbox_height_list.append(bbox_height)

                area = (int(ymax) - int(ymin)) * (int(xmax) - int(xmin))
                bboxInimgArea = area / imgArea
                area_list.append(area)
                bboxInimgArea_list.append(bboxInimgArea)
                # print("Area is", area)

                ratio = (int(xmax) - int(xmin)) / (int(ymax) - int(ymin))
                ratio_list.append(ratio)
                # print("Ratio is", round(ratio,2))

bbox_width_array = np.array(bbox_width_list)
bbox_height_array = np.array(bbox_height_list)
plt.figure(1)
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.scatter(bbox_width_array,bbox_height_array,
            c = 'blue',
            label = 'BBOX size')
plt.tick_params(labelsize=30)
plt.xlabel('bbox_width/m',fontsize=30)
plt.ylabel('bbox_height/m',fontsize=30)
plt.title('SAR-AIRcraft1.0_BBOX_size',fontsize=25
          )
plt.show()


square_array = np.array(area_list)
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

bboxInimgArea_array = np.array(bboxInimgArea_list)
bboxInimgArea_max = np.max(bboxInimgArea_array)
bboxInimgArea_min = np.min(bboxInimgArea_array)
bboxInimgArea_mean = np.mean(bboxInimgArea_array)
bboxInimgArea_var = np.var(bboxInimgArea_array)
plt.figure(2)
plt.hist(bboxInimgArea_array, 20)
plt.xlabel('Ratio of bbox / img')
plt.ylabel('Frequency of ratio')
plt.title('SAR-AIRcraft1.0 Ratio\n' \
          + 'max=' + str(round(bboxInimgArea_max, 2)) + ', min=' + str(round(bboxInimgArea_min, 2)) + '\n' \
          + 'mean=' + str(round(bboxInimgArea_mean, 2)) + ', var=' + str(round(bboxInimgArea_var, 2))
          )

plt.show()
#
#
# ratio_array = np.array(ratio_list)
# ratio_max = np.max(ratio_array)
# ratio_min = np.min(ratio_array)
# ratio_mean = np.mean(ratio_array)
# ratio_var = np.var(ratio_array)
# plt.figure(3)
# plt.hist(ratio_array, 20)
# plt.xlabel('Ratio of  width/height')
# plt.ylabel('Frequency of ratio')
# plt.title('Ratio\n' \
#           + 'max=' + str(round(ratio_max, 2)) + ', min=' + str(round(ratio_min, 2)) + '\n' \
#           + 'mean=' + str(round(ratio_mean, 2)) + ', var=' + str(round(ratio_var, 2))
#           )
#
# plt.show()
#

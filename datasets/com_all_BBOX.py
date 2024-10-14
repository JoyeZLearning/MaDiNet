# -*- coding: utf-8 -*-
"""
Compute all three datasets bbox
"""

import os
import xml.etree.cElementTree as et
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from pathlib import Path


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

bbox_width_array_air = np.array(bbox_width_list)
bbox_height_array_air = np.array(bbox_height_list)


# 读取数据
ann_json_path_train = r"D:\softapp\Anaconda\envs\pytorch_gpu\DiffDet_original\datasets_ssdd\annotations\instances_train.json"
with open(ann_json_path_train) as f:
    ann_train = json.load(f)

# 创建{1: 'multi_signs', 2: 'window_shielding', 3: 'non_traffic_sign'}
# 创建{'multi_signs': 0, 'window_shielding': 0, 'non_traffic_sign': 0}
categorys_dic_train = dict([(i['id'], i['name']) for i in ann_train['categories']])
categorys_num_train = dict([i['name'], 0] for i in ann_train['categories'])

# 统计每个类别的数量
for i in ann_train['annotations']:
    categorys_num_train[categorys_dic_train[i['category_id']]] += 1

# 统计bbox的w、h、wh
bbox_w_train = []
bbox_h_train = []
bbox_wh_train = []
area_list_train = []
for i in ann_train['annotations']:
    bbox_w_train.append(round(i['bbox'][2], 2))
    bbox_h_train.append(round(i['bbox'][3], 2))
    area_list_train.append(round(i['area'],2))
    wh_train = round(i['bbox'][2] / i['bbox'][3], 0)
    # if (wh < 1):
    #     wh = round(i['bbox'][3] / i['bbox'][2], 0)
    bbox_wh_train.append(wh_train)



# 读取数据
ann_json_path_val = r"D:\softapp\Anaconda\envs\pytorch_gpu\DiffDet_original\datasets_ssdd\annotations\instances_val.json"
with open(ann_json_path_val) as f:
    ann_val = json.load(f)

# 创建{1: 'multi_signs', 2: 'window_shielding', 3: 'non_traffic_sign'}
# 创建{'multi_signs': 0, 'window_shielding': 0, 'non_traffic_sign': 0}
categorys_dic_val = dict([(i['id'], i['name']) for i in ann_val['categories']])
categorys_num_val = dict([i['name'], 0] for i in ann_val['categories'])

# 统计每个类别的数量
for i in ann_val['annotations']:
    categorys_num_val[categorys_dic_val[i['category_id']]] += 1

# 统计bbox的w、h、wh
bbox_w_val = []
bbox_h_val = []
bbox_wh_val = []
area_list_val = []
for i in ann_val['annotations']:
    bbox_w_val.append(round(i['bbox'][2], 2))
    bbox_h_val.append(round(i['bbox'][3], 2))
    area_list_val.append(round(i['area'],2))
    wh_val = round(i['bbox'][2] / i['bbox'][3], 0)
    # if (wh < 1):
    #     wh = round(i['bbox'][3] / i['bbox'][2], 0)
    bbox_wh_val.append(wh_val)



bbox_w = bbox_w_train +bbox_w_val
bbox_h = bbox_h_train +bbox_h_val
bbox_wh = bbox_wh_train +bbox_wh_val
area_list = area_list_train +area_list_val



bbox_width_array_ssdd = np.array(bbox_w)
bbox_height_array_ssdd = np.array(bbox_h)




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


            area = float(coor_list[2]) * float(coor_list[3])  # 计算出当前txt文件中每一个gt的面积
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

labeldir = r'D:\softapp\Anaconda\envs\pytorch_gpu\datasets\SARship_dataset_v0_label'
data_dict,bbox_width_list,bbox_height_list = getGtAreaAndRatio(labeldir)
bbox_width_array_ship = np.array(bbox_width_list)
bbox_height_array_ship = np.array(bbox_height_list)

plt.figure(1)
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.scatter(bbox_width_array_ship,bbox_height_array_ship,
            c = 'g',
            # c = 'orange',
            label = 'SAR-ShipDataset')
plt.scatter(bbox_width_array_air,bbox_height_array_air,
            c = 'blue',
            # c = 'orange',
            label = 'SAR-AIRcraft1.0')
plt.scatter(bbox_width_array_ssdd,bbox_height_array_ssdd,
            c = 'r',
            label = 'SSDD')



plt.tick_params(labelsize=30)
plt.xlabel('bbox_width/m',fontsize=30)
plt.ylabel('bbox_height/m',fontsize=30)
# plt.title('BBOX_size',fontsize=25
#           )
plt.legend(loc="best",fontsize=22)
plt.savefig('./bboxSize.eps')
plt.show()
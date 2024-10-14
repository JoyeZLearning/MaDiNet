# -*- coding: utf-8 -*-
"""
Compute HRSID bbox
"""



import pandas as pd
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体为黑体
plt.rcParams['font.family'] = 'sans-serif'  # 设置字体样式
plt.rcParams['figure.figsize'] = (10.0, 10.0)  # 设置字体大小

# 读取数据
ann_json_path_train = r"D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\HRSID_JPG\annotations\train2017.json"
with open(ann_json_path_train) as f:
    ann_train = json.load(f)

# 创建{1: 'multi_signs', 2: 'window_shielding', 3: 'non_traffic_sign'}
# 创建{'multi_signs': 0, 'window_shielding': 0, 'non_traffic_sign': 0}
categorys_dic_train = dict([(i['id'], i['name']) for i in ann_train['categories']])
categorys_num_train = dict([i['name'], 0] for i in ann_train['categories'])

img_w_train = []
img_h_train = []

for j in ann_train['images']:
    img_w_train.append(int(j['width']))
    img_h_train.append(int(j['height']))

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
ann_json_path_val = r"D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\HRSID_JPG\annotations\test2017.json"
with open(ann_json_path_val) as f:
    ann_val = json.load(f)

# 创建{1: 'multi_signs', 2: 'window_shielding', 3: 'non_traffic_sign'}
# 创建{'multi_signs': 0, 'window_shielding': 0, 'non_traffic_sign': 0}
categorys_dic_val = dict([(i['id'], i['name']) for i in ann_val['categories']])
categorys_num_val = dict([i['name'], 0] for i in ann_val['categories'])


img_w_val = []
img_h_val = []

for j in ann_val['images']:
    img_w_val.append(int(j['width']))
    img_h_val.append(int(j['height']))


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



bbox_width_array = np.array(bbox_w)
bbox_height_array = np.array(bbox_h)
plt.figure(1)
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.scatter(bbox_width_array,bbox_height_array,
            c = 'blue',
            label = 'BBOX size')

plt.tick_params(labelsize=30)
plt.xlabel('bbox_width/m',fontsize=30)
plt.ylabel('bbox_height/m',fontsize=30)
plt.title('HRSID_BBOX_size',fontsize=25
          )
plt.show()

img_w = img_w_train +img_w_val
img_h = img_h_train +img_h_val

img_w_array = np.array(img_w)
img_h_array = np.array(img_h)
plt.figure(1)
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.scatter(img_w_array,img_h_array,
            c = 'blue',
            label = 'img size')

plt.tick_params(labelsize=30)
plt.xlabel('img_width/m',fontsize=30)
plt.ylabel('img_height/m',fontsize=30)
plt.title('HRSID_IMG_size',fontsize=25
          )
plt.show()


# 画出bbox_ratio  w/h
ratio_array = np.array(bbox_wh)
ratio_max = np.max(ratio_array)
ratio_min = np.min(ratio_array)
ratio_mean = np.mean(ratio_array)
ratio_var = np.var(ratio_array)
plt.figure(3)
plt.hist(ratio_array, 20)
plt.xlabel('Ratio of  width/height')
plt.ylabel('Frequency of ratio')
plt.title('HRSID Ratio\n' \
          + 'max=' + str(round(ratio_max, 2)) + ', min=' + str(round(ratio_min, 2)) + '\n' \
          + 'mean=' + str(round(ratio_mean, 2)) + ', var=' + str(round(ratio_var, 2))
          )

plt.show()


# 画出bbox/img
square_array = np.array(area_list)
square_max = np.max(square_array)
square_min = np.min(square_array)
square_mean = np.mean(square_array)
square_var = np.var(square_array)
plt.figure(1)
plt.hist(square_array, 20)
plt.xlabel('Area in pixel')
plt.ylabel('Frequency of area')
plt.title('HRSID Area\n' \
          + 'max=' + str(square_max) + ', min=' + str(square_min) + '\n' \
          + 'mean=' + str(int(square_mean)) + ', var=' + str(int(square_var))
          )
plt.show()


# # 统计所有的宽高比
# bbox_wh_unique = set(bbox_wh)  # set挑选出不重复的元素，即挑选出有多少种比例的anchors
# # print(bbox_wh_unique)
# bbox_count_unique = [bbox_wh.count(i) for i in bbox_wh_unique]  # 统计宽高比数量
# # print(bbox_count_unique)
#
# # 画图
# wh_dataframe = pd.DataFrame(bbox_count_unique, index=bbox_wh_unique, columns=['宽高比数量'])
# wh_dataframe.plot(kind='bar', color="#55aacc")
# plt.show

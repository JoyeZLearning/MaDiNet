import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = './BSDS300/syn_train'
path = './BSDS300/syn_train/'
img_ids    =   sorted(os.listdir(data_dir))
print(len(img_ids))

for img_id in img_ids:
    dataFile =  path + img_id  # 单个的mat文件
    data = scio.loadmat(dataFile)
    # print(type(data))
    # print (data['data'])
    # 由于导入的mat文件是structure类型的，所以需要取出需要的数据矩阵
    # a1=data['inst_map']
    a1=data['clean']
    a2=data['noisy']
    # 取出需要的数据矩阵

    # 数据矩阵转图片的函数
    def MatrixToImage(data):
        data = data*255
        new_im = Image.fromarray(data.astype(np.uint8))
        return new_im

    new_im_clean = MatrixToImage(a1)
    new_im_noisy = MatrixToImage(a2)
    # plt.imshow(a, cmap=plt.cm.gray, interpolation='nearest')
    # new_im.show()
    # new_im_clean.save(img_id[:-4] + '.png') # 保存图片
    new_im_noisy.save(img_id[:-4] + '_noisy.png') # 保存图片

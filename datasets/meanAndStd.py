import os
from PIL import Image
import numpy as np
import tqdm

def main(path):
    # 数据集通道数
    img_channels = 3
    # img_names = os.listdir('/root/autodl-tmp/project/DiffusionDet-main/datasets/SAR_Ship_dataset/ship_train')
    img_names = os.listdir(path)
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for img_name in tqdm.tqdm(img_names, total=len(img_names)):
        img_path = os.path.join(path, img_name)
        img = np.array(Image.open(img_path)) / 255.
        # 对每个维度进行统计，Image.open打开的是HWC格式，最后一维是通道数
        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(img_names)
    std = cumulative_std / len(img_names)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__== '__main__':
    main('/root/autodl-tmp/project/DiffusionDet-main/datasets/SAR_Ship_dataset/ship_train')
    # main(r"D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\datasets\AIRcraft_VOC\images")

# import numpy as np
# import cv2
# import os
#
# # img_h, img_w = 32, 32
# # img_h, img_w = 32, 48  # 根据自己数据集适当调整，影响不大
# means, stdevs = [], []
# img_list = []
#
# imgs_path = r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\datasets\coco\val2017'
# imgs_path_list = os.listdir(imgs_path)
#
# len_ = len(imgs_path_list)
# i = 0
# for item in imgs_path_list:
#     img = cv2.imread(os.path.join(imgs_path, item))
#     # img = cv2.resize(img, (img_w, img_h))
#     img = img[:, :, :, np.newaxis]
#     img_list.append(img)
#     i += 1
#     # print(i, '/', len_)
#
# imgs = np.concatenate(img_list, axis=3)
# imgs = imgs.astype(np.float32) / 255.
#
# for i in range(3):
#     pixels = imgs[:, :, i, :].ravel()  # 拉成一行
#     means.append(np.mean(pixels))
#     stdevs.append(np.std(pixels))
#
# # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
# means.reverse()
# stdevs.reverse()
#
# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))

# import torchvision
# from torchvision.datasets import ImageFolder
# import torch
# from torchvision import transforms
#
# def getStat(train_data):
#     '''
#     Compute mean and variance for training data
#     :param train_data: 自定义类Dataset(或ImageFolder即可)
#     :return: (mean, std)
#     '''
#     print('Compute mean and variance for training data.')
#     print(len(train_data))
#     train_loader = torch.utils.data.DataLoader(
#         train_data, batch_size=1, shuffle=False, num_workers=0,
#         pin_memory=True)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     for X, _ in train_loader:
#         for d in range(3):
#             mean[d] += X[:, d, :, :].mean()
#             std[d] += X[:, d, :, :].std()
#     mean.div_(len(train_data))
#     std.div_(len(train_data))
#     return list(mean.numpy()), list(std.numpy())
#
#
# if __name__ == '__main__':
#     train_dataset = ImageFolder(root=r'/root/autodl-tmp/project/DiffDet_original/datasets_ssdd', transform=torchvision.transforms.ToTensor())
#     print(getStat(train_dataset))
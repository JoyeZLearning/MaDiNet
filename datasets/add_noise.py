import os
import cv2
import numpy as np
import random

#添加椒盐噪声

def sp_noise(noise_img,proportion):
    '''
    添加椒盐噪声
    proportion表示加入的噪声的量，自行调整
    return: img_noise
    '''
    height, width = noise_img.shape[0],noise_img.shape[1] #获取高宽像素值
    num = int(height * weight * proportion) # 一个准备加入多少噪声小点
    for i in range(num):
        w  = random.randint(0,width -1)
        h  = random.randint(0,height -1)
        if random.randint(0,1) == 0:
            noise_img[h,w] = 0
        else:
            noise_img[h,w] = 255
    return noise_img


#添加高斯噪声
def gaussian_noise(img,mean,sigma):
    '''
    此函数将产生的高斯噪声加到图片上
    传入
      img 原图
      mean 均值
      sigma 标准差
    返回
      gaussian_out 噪声处理后的图片
    '''

    #将图片灰度标准化
    img = img / 255
    #产生Gaussian 噪声
    noise = np.random.normal(mean,sigma,img.shape)

    gaussian_out = img + noise

    gaussian_out = np.clip(gaussian_out,0,1)

    # 将图片恢复为255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围恢复到0-255
    # noise = np.unit8(noise*255)

    return gaussian_out



# 添加随机噪声
def random_noise(image,noise_num):
    '''
    添加随机噪点（实际上就是随机在图像上将像素点的灰度值变成255即白色）
    param image: 需要加噪的图片
    param noise_num: 添加的噪音点数目
    return: img_noise
    '''
    img_noise = image
    rows, cols, chn = img_noise.shape

    for i in range(noise_num):
        x = np.random.randint(0,rows)
        y = np.random.randint(0,cols)
        img_noise[x,y,:] = 255
    return img_noise


# 读取图片并保存
def convert(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        path = input_dir + '/' + filename
        print('doing...',path)
        #读取图片
        noise_img = cv2.imread(path)
        img_noise = gaussian_noise(noise_img,0,0.005)
        # img_noise = sp_noise(noise_num,0,0.25) # 椒盐噪声
        # img_noise = random_noise(noise_img,500) # 随机噪声
        cv2.imwrite(output_dir+'/'+filename,img_noise)


if __name__ == '__main__':
    # input_dir = r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\datasets\coco\val2017'
    # output_dir = r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\datasets\coco\val2017_noise'
    input_dir = '/root/autodl-tmp/project/DiffusionDet-main/datasets/coco/val2017'
    output_dir = '/root/autodl-tmp/project/DiffusionDet-main/datasets/coco/val2017_noise_0_0.005'

    convert(input_dir,output_dir)
















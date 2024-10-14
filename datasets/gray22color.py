import cv2
import numpy as np

# # 读取灰度图像
# gray_image = cv2.imread(r'D:\softapp\Anaconda\envs\pytorch_gpu\datasets\scatter_airplane\a220_1.png', cv2.IMREAD_GRAYSCALE)
#
# # 检查图像是否成功读取
# if gray_image is None:
#     print("图像读取失败，请检查文件路径")
# else:
#     # 创建一个与灰度图像大小相同的彩色图像
#     color_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
#
#     # 将灰度值复制到彩色图像的每个通道
#     color_image[:, :, 0] = gray_image
#     color_image[:, :, 1] = gray_image
#     color_image[:, :, 2] = gray_image
#
#     # 显示原始灰度图像和转换后的彩色图像
#     cv2.imshow('Gray Image', gray_image)
#     cv2.imshow('Color Image', color_image)
#
#     # 保存转换后的彩色图像
#     cv2.imwrite('./scatter_rgb_airplane/a220_1.png', color_image)


# # 创建映射表
# def create_colormap():
#     colormap = np.zeros((256, 3), dtype=np.uint8)
#     for i in range(256):
#         colormap[i] = [255 - i, i // 2, i]  # 定制映射规则
#     return colormap
#
#
# def apply_custom_colormap(gray_image, colormap):
#     color_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
#
#     rows, cols = gray_image.shape
#     for row in range(rows):
#         for col in range(cols):
#             color_image[row, col] = colormap[gray_image[row, col]]
#     return color_image
#
#
# # 使用自定义映射
# colormap = create_colormap()
# gray_image = cv2.imread(r'D:\softapp\Anaconda\envs\pytorch_gpu\datasets\scatter_airplane\a220_1.png', cv2.IMREAD_GRAYSCALE)
# color_image = apply_custom_colormap(gray_image, colormap)
#
# # 显示图像
# cv2.imshow('Color Image', color_image)
# cv2.imwrite('./scatter_rgb_airplane/a220_1.png', color_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.imread(r'D:\softapp\Anaconda\envs\pytorch_gpu\datasets\scatter_airplane\airplane1.png')


dst = cv2.applyColorMap(img, 12)
# cv2.imwrite('poisson_noise.jpg', dst)
cv2.imshow('map', dst)
cv2.imwrite('./scatter_rgb_airplane/airplane1.png',dst,[cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.waitKey()
# cv2.destroyAllWindows()
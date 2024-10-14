import os
from osgeo import gdal

file_folder = 'val/images_tif'  # 储存tif格式的文件夹
output_folder = 'val/images_png'  # 储存png格式的文件夹

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(file_folder):
    file_path = os.path.join(file_folder, file_name)
    print(file_path)

    # 检查是否为 TIFF 文件
    if file_path.endswith('.tif'):
        ds = gdal.Open(file_path)
        if ds is None:
            print(f"Failed to open {file_path}")
            continue

        driver = gdal.GetDriverByName('PNG')
        # driver = gdal.GetDriverByName('JPEG')  # 使用JPEG驱动
        # 创建输出文件的路径
        output_file_path = os.path.join(output_folder, file_name.replace('.tif', '.png'))
        # output_file_path = os.path.join(output_folder, file_name.replace('.tif', '.jpg'))
        dst_ds = driver.CreateCopy(output_file_path, ds)
        del dst_ds  # 释放资源
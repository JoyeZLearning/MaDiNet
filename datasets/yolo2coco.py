import os
import json
from PIL import Image

# 设置数据集路径
dataset_path = "D:/softapp/Anaconda/envs/pytorch_gpu/DiffusionDet-main/datasets/SAR_Ship_dataset"
images_path = os.path.join(dataset_path, "ship_dataset_v0_data")
labels_path = os.path.join(dataset_path, "ship_dataset_v0_labels")

# 类别映射
categories = [
    {"id": 1, "name": "ship"},
    # 添加更多类别
]


# YOLO格式转COCO格式的函数
def convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height):
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]


# 初始化COCO数据结构
def init_coco_format():
    return {
        "images": [],
        "annotations": [],
        "categories": categories
    }


# 处理每个数据集分区
for split in ['train', 'test', 'val']:
    coco_format = init_coco_format()
    annotation_id = 1

    for img_name in os.listdir(os.path.join(images_path, split)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_path, split, img_name)
            label_path = os.path.join(labels_path, split, img_name.replace("jpg", "txt"))

            img = Image.open(img_path)
            img_width, img_height = img.size
            image_info = {
                "file_name": img_name,
                "id": len(coco_format["images"]) + 1,
                "width": img_width,
                "height": img_height
            }
            coco_format["images"].append(image_info)

            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    for line in file:
                        category_id, x_center, y_center, width, height = map(float, line.split())
                        bbox = convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height)
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_info["id"],
                            "category_id": int(category_id) + 1,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        }
                        coco_format["annotations"].append(annotation)
                        annotation_id += 1

    # 为每个分区保存JSON文件
    with open(f"path/to/output/{split}_coco_format.json", "w") as json_file:
        json.dump(coco_format, json_file, indent=4)
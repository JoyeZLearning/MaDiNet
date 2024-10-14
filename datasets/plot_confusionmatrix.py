import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json,os
import pycocotools.coco as coco
import cv2

import numpy as np


def show_cls(category_id):
    id = category_id
    if int(id) == 1:
        id = 'A220'
    elif int(id) == 2:
        id = 'A320/321'
    elif int(id) ==3:
        id = 'A330'
    elif int(id) == 4:
        id = 'ARJ21'
    elif int(id) == 5:
        id = 'Boeing737'
    elif int(id) == 6:
        id = 'Boeing787'
    elif int(id) == 7:
        id = 'other'
    else:
        Exception('not a valid id')
    return id

CONFIDENCE_thres = 0.5


sns.set()

air_classes =['A220','A320/321','A330','ARJ21','Boeing737','Boeing787','other']
gt_label = []
pre_label = []

def confusionmatrix_plot(detections,groundtruth):
    for i in range(groundtruth.shape[0]):
        pt_gt = groundtruth[i, :]
        pt_gt_cls = str(show_cls(pt_gt[4]))
        gt_label.append(pt_gt_cls)
        if detections[i, 4] >= CONFIDENCE_thres:
             pt = detections[i, :]
             pt_cls = str(show_cls(pt[5]))
             pre_label.append(pt_cls)
        if len(gt_label) >= len(pre_label):
            pre_label.append(' ')
        elif len(gt_label) <= len(pre_label):
            gt_label.append(' ')

    # f, (ax1, ax2) = plt.subplots(figsize=(10, 8), nrows=7)
    # C2 = confusion_matrix(gt_label, pre_label, labels=['A220','A320_321','A330','ARJ21','Boeing737','Boeing787','other'])
    # print(C2)
    # print(C2.ravel())
    # sns.heatmap(C2, annot=True)
    #
    # ax2.set_title('sns_heatmap_confusion_matrix')
    # ax2.set_xlabel('pd')
    # ax2.set_ylabel('gt')
    # f.savefig('sns_heatmap_confusion_matrix.jpg', bbox_inches='tight')


datasets_upper = 'D:/softapp/Anaconda/envs/pytorch_gpu/DiffusionDet-main/datasets/coco'

mode = 'val'

if mode == 'train':
    anno_path = 'annotations/instances_train2017.json'
    img_path = 'train2017'
elif mode == 'val':
    anno_path = 'annotations/instances_val2017.json'
    img_path = 'val2017'
elif mode == 'test':
    anno_path = 'annotations/image_info_test2017.json'
    img_path = 'test2017'
else:
    raise Exception('Not a valid mode')

#full path
anno_full_path = os.path.join(datasets_upper,anno_path)
img_full_path = os.path.join(datasets_upper,img_path)

#get info
coco_sar = coco.COCO(anno_full_path)
images_ids = coco_sar.getImgIds()
num_images = len(images_ids)


# 加了r就不用改变斜杠的方向
# det_json_path =  r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\datasets\coco\annotations\instances_val2017.json'
det_json_path =  r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\output_RE\inference\coco_eval_instances_results.json'
# det_json_path =  r'D:\softapp\Anaconda\envs\pytorch_gpu\ConsistencyDet-main\output_gamma_aircraft1.0\inference\coco_eval_instances_results.json'
# det_json_path =  r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\output_bbox500_CPDCfusionp5_scale1.0\inference\coco_eval_instances_results.json'
coco_dets = coco_sar.loadRes(det_json_path)
# coco_dets = coco_sar.loadAnns(det_json_path)

#iterative process for each image:
for index in range(num_images):
    img_id = images_ids[index]
    file_name = coco_sar.loadImgs(ids=[img_id])[0]['file_name']

    singele_img_path = os.path.join(img_full_path,file_name)
    img = cv2.imread(singele_img_path)

    ann_ids = coco_sar.getAnnIds(imgIds=[img_id])
    anns = coco_sar.loadAnns(ids=ann_ids)
    num_objs = len(anns)
    anno_coord = [i['bbox']+[i['category_id']] for i in anns]
    coords = np.array(anno_coord).reshape(-1,5)
    # anno_coord = [i['bbox'] for i in anns]

    # coords = np.array(anno_coord).reshape(-1,4)
    # coords[:,2:4] = coords[:,:2] + coords[:,2:4]

    # # show gt
    # frame = cv2_demo_gt(img.copy(), coords[:, :])
    # cv2.imshow('frame', frame)
    # cv2.waitKey(1000)

    # show detections
    ann_ids_det = coco_dets.getAnnIds(imgIds=[img_id])
    dets = coco_dets.loadAnns(ids=ann_ids_det)
    # det_coords = [i['bbox']+[i['score']] for i in dets]
    # det_coords = np.array(det_coords).reshape([-1,5])
    det_coords = [i['bbox']+[i['score']]+[i['category_id']]for i in dets]
    det_coords = np.array(det_coords).reshape([-1,6])

    # frame_det = cv2_demo(img.copy(),det_coords)
    # cv2.imshow('frame_det',frame_det)
    # cv2.waitKey(1000)


# #show gt
#     file_path = '/\datasets\detection_gt_test_text'
#     file_path = os.path.join(file_path,str(img_id)+'.jpg')
    confusionmatrix_plot(det_coords, coords[:, :])


# f,  ax2 = plt.subplots(figsize=(10, 8), nrows=7)
C2 = confusion_matrix(gt_label, pre_label,
                      labels=['A220', 'A320/321', 'A330', 'ARJ21', 'Boeing737', 'Boeing787', 'other'])
print(C2)
print(C2.ravel())
sns.heatmap(C2, annot=True)
plt.show()
# ax2.set_title('sns_heatmap_confusion_matrix')
# ax2.set_xlabel('pd')
# ax2.set_ylabel('gt')
# f.savefig('sns_heatmap_confusion_matrix.jpg', bbox_inches='tight')
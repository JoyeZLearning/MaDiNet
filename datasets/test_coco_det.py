import cv2
import random
import json,os
import pycocotools.coco as coco
import numpy as np
# import diffusers

CONFIDENCE_thres=0.5
COLORS = [(0,255,0)]
COLORS_GT = [(0,0,255)]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT1 = cv2.FONT_HERSHEY_COMPLEX

def cv2_demo(frame,detections):
    det = []
    for i in range(detections.shape[0]):
        if detections[i,4] >= CONFIDENCE_thres:
            pt = detections[i,:]
            cv2.rectangle(frame,(int(pt[0]),int(pt[1]),int(pt[2]),int(pt[3])),COLORS[0],2)
            cv2.putText(frame,str(pt[4])[:5],(int(pt[0]),(int(pt[1]))),FONT1,1,(0,255,0),1)
            det.append([int(pt[0]),int(pt[1]),int(pt[2]),int(pt[3]),detections[i,4]])
    return frame

def cv2_demo_gt(frame,detections):
    det = []
    for i in range(detections.shape[0]):
            pt_gt = detections[i,:]
            #rectangele(img,left-up coordinate,right-bottom coordinate,colors, the thickness of line )
            cv2.rectangle(frame,(int(pt_gt[0]),int(pt_gt[1]),int(pt_gt[2]),int(pt_gt[3])),COLORS_GT[0],3)
            # cv2.rectangle(frame,(int(pt[0])-4,int(pt[1])-4,int(pt[2])+4,int(pt[3])+4),COLORS[0],1)
            # cv2.putText(frame,str(pt[4]),(int(pt[0]),int(pt[1])),FONT,1,(0,255,0),1)
            # det.append([int(pt[0]),int(pt[1]),int(pt[2]),int(pt[3]),detections[i,4]])
    return frame

#show category
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


def cv2_demo_det_gt(frame,detections,groundtruth):
    det = []
    for i in range(groundtruth.shape[0]):
        pt_gt = groundtruth[i, :]
        pt_gt_cls = show_cls(pt_gt[4])
        cv2.rectangle(frame, (int(pt_gt[0]), int(pt_gt[1]), int(pt_gt[2]), int(pt_gt[3])), COLORS_GT[0], 3)
        cv2.putText(frame, str(pt_gt_cls), (int(pt_gt[0]-50), (int(pt_gt[1])+50)), FONT1, 0.5, COLORS_GT[0], 1)
        if detections[i, 4] >= CONFIDENCE_thres:
            pt = detections[i, :]
            cv2.rectangle(frame, (int(pt[0])-4, int(pt[1])-4, int(pt[2])+4, int(pt[3])+4), COLORS[0], 2)
            pt_cls = show_cls(pt[5])
            cv2.putText(frame, str(str(pt_cls)+str('/')+str(pt[4])[:5]), (int(pt[0]), (int(pt[1]))-10), FONT1, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, str(pt[4])[:5], (int(pt[0]), (int(pt[1]))-10), FONT1, 1, (0, 255, 0), 1)
            det.append([int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]), detections[i, 4]])
    return frame

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
# anno_full_path = r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\datasets\SAR_Ship_dataset\annotations\ship_val.json'
img_full_path = os.path.join(datasets_upper,img_path)
# img_full_path = r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\datasets\SAR_Ship_dataset\ship_val'


#get info
coco_sar = coco.COCO(anno_full_path)
images_ids = coco_sar.getImgIds()
num_images = len(images_ids)


# 加了r就不用改变斜杠的方向
# det_json_path = r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\output_shipdataset_189999_96.931\inference\coco_eval_instances_results.json'
det_json_path = r'D:\softapp\Anaconda\envs\pytorch_gpu\DiffusionDet-main\output_ssdd_29999_gamma_res50\inference\coco_eval_instances_results.json'
coco_dets = coco_sar.loadRes(det_json_path)

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
    #   # det_coords = [i['bbox']+[i['score']] for i in dets]
    #   # det_coords = np.array(det_coords).reshape([-1,5])
    det_coords = [i['bbox']+[i['score']]+[i['category_id']]for i in dets]
    det_coords = np.array(det_coords).reshape([-1,6])

    # frame_det = cv2_demo(img.copy(),det_coords)
    # cv2.imshow('frame_det',frame_det)
    # cv2.waitKey(1000)


#show gt
    file_path = r'G:\thesis_Dr\MDiffdet4SAR_ssdd_det'
    file_path = os.path.join(file_path,str(img_id)+'.jpg')
    frame = cv2_demo_det_gt(img.copy(),det_coords,coords[:,:])
    # frame = cv2_demo_gt(img.copy(),coords[:,:])

    cv2.imwrite(file_path,frame)
    # cv2.imshow('frame',frame)
    # cv2.waitKey(1000)



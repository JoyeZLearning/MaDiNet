import argparse
import pickle,json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import itertools
from terminaltables import AsciiTable


parser = argparse.ArgumentParser(description='Calculating metrics (AR,Recall,F1) in every class')
parser.add_argument('--det_json',default='/root/autodl-tmp/project/DiffusionDet-main/datasets/coco/annotations/instances_val2017.json', type=str,
                    help='inference detection json file path')
parser.add_argument('--gt_json',default='/root/autodl-tmp/project/DiffusionDet-main/output-30000/inference/coco_eval_instances_results.json',type=str,
                    help='ground truth json file path')
parser.add_argument('--classes',default=('A220','A320/321','A330','ARJ21','Boeing737','Boeing787','other'),type=tuple,
                    help='every class name with str type in a tuple')

def read_pickle(pkl):
    with open(pkl,'rb') as f:
        data = pickle.load(f)
    return data


def read_json(json_path):
    with open(json_path,'r') as f:
        data = json.load(f)
    return data


def process(det_json, gt_json, CLASSES):
    cocoGT = COCO(gt_json)

    #获取类别（单类）对应的id
    catIds = cocoGT.getCatIds(catNms=list(CLASSES))


    # 获取多个类别对应的所有图片的Id
    imgid_list = []
    for id_c in catIds:
        imgIds = cocoGT.getImgIds(catIds=id_c)
        imgid_list.extend(imgIds)


    #通过gt的json文件和pred的json文件计算map
    class_num = len(CLASSES)
    cocoGT = COCO(gt_json)
    cocoDt = cocoGT.loadRes(det_json)
    cocoEval = COCOeval(cocoGT,cocoDt,'bbox')
    cocoEval.params.iouThrs = np.linspace(0.5,0.95,int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    cocoEval.params.maxDets = list((100,300,1000))
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


    # 计算F1
    # precision_f1 = round(cocoEval.stats[1] * 100, 3)  # Precision at IoU
    # recall_f1 = round(cocoEval.stats[8] * 100, 3)  # Recall at IoU
    # ap_75_f1 = round(cocoEval.stats[2] * 100, 3)  # AP at IoU
    # # 计算F1-Score
    # f1_score = round(2 * (precision_f1 * recall_f1) / (precision_f1 + recall_f1), 3)
    # print('f1_score:', f1_score)
    # print('ap_75_f1:', ap_75_f1)


    # 看下cocoEval.eval的内容，应该是字典
    # 通过gt 和pred计算precision和recall
    # 101是计算PR曲线求AP时候，积分，分成101小块，计算的更加精确
    precisions = cocoEval.eval['precision']
    recalls = cocoEval.eval['recall']
    # f1_score_2 =2 * (precisions[2] * recalls[1]) / (precisions[2] + recalls[1])
    # print(f1_score_2)
    a = precisions[0, :, 0, 0, 2]

    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[0], np.mean(precisions[0, :, :, 0, -1]),
                                                     np.mean(recalls[0, :, 0, -1])))

    print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[5], np.mean(precisions[5, :, :, 0, -1]),
                                                     np.mean(recalls[5, :, 0, -1])))


    # compute per-category AP
    # ptrcision: (iou, recall, cls, area, range, max dets)
    results_per_category = []
    results_per_category_iou50 = []
    results_per_category_iou75 = []
    res_item = []

    for idx,catIds in enumerate(range(class_num)):
        name = CLASSES[idx]

        # 这里的precision_50/75有101维，对应求解AP时候的计算公式
        precision = precisions[:, :, idx, 0, -1]
        precision_50 = precisions[0, :, idx, 0, -1]
        precision_75 = precisions[5, :, idx, 0, -1]
        # 这个输出看一下是啥意思
        precision = precision[precision > -1]

        recall = recalls[:, idx, 0, -1]
        recall_50 = recalls[0, idx, 0, -1]
        recall_75 = recalls[5, idx, 0, -1]
        # 这个输出看一下是啥意思
        recall = recall[recall > -1]

        if precision.size:
            ap = np.mean(precision)
            ap_50 = np.mean(precision_50)
            ap_75 = np.mean(precision_75)
            rec = np.mean(recall)
            rec_50 = np.mean(recall_50)
            rec_75 = np.mean(recall_75)
            #不太知道F1是不是这样计算的
            # F1 = (2*ap*rec)/(ap+rec)
        else:
            ap = float('nan')
            ap_50 = float('nan')
            ap_75 = float('nan')
            rec = float('nan')
            rec_50 = float('nan')
            rec_75 = float('nan')

        res_item = [f'{name}', f'{float(ap):0.3f}',f'{float(rec):0.3f}']
        results_per_category.append(res_item)
        res_item_50 = [f'{name}', f'{float(ap_50):0.3f}',f'{float(rec_50):0.3f}']
        results_per_category_iou50.append(res_item_50)
        res_item_75 = [f'{name}', f'{float(ap_75):0.3f}',f'{float(rec_75):0.3f}']
        results_per_category_iou75.append(res_item_75)


    item_num = len(res_item)
    # 这个输出也看一下，这一堆要搞清楚是在干啥，好像是在设置表格的显示方式
    num_columns = min(6, len(results_per_category) * item_num)
    results_flatten = list(
        itertools.chain(*results_per_category))
    headers = ['caterogy', 'AP', 'Recall'] * (num_columns // item_num)
    results_2d = itertools.zip_longest(*[
        results_flatten[i::num_columns]
        for i in range(num_columns)
    ])

    table_data = [headers]
    table_data += [result for result in results_2d]
    table = AsciiTable(table_data)
    print('\n' + table.table)


    # 显示AP50
    num_columns_50 = min(6, len(results_per_category_iou50) * item_num)
    results_flatten_50 = list(
        itertools.chain(*results_per_category_iou50))
    iou_ = cocoEval.params.iouThrs[0]
    headers_50 = ['caterogy', 'AP{}'.format(iou_), 'Recall{}'.format(iou_)] * (num_columns_50 // item_num)
    results_2d_50 = itertools.zip_longest(*[
        results_flatten_50[i::num_columns]
        for i in range(num_columns_50)
    ])

    table_data_50 = [headers_50]
    table_data_50 += [result for result in results_2d_50]
    table_50 = AsciiTable(table_data_50)
    print('\n' + table_50.table)


    # 显示AP75
    num_columns_75 = min(6, len(results_per_category_iou75) * item_num)
    results_flatten_75 = list(
        itertools.chain(*results_per_category_iou75))
    iou_5 = cocoEval.params.iouThrs[5]
    headers_75 = ['caterogy', 'AP{}'.format(iou_5), 'Recall{}'.format(iou_5)] * (num_columns_75 // item_num)
    results_2d_75 = itertools.zip_longest(*[
        results_flatten_75[i::num_columns]
        for i in range(num_columns_75)
    ])

    table_data_75 = [headers_75]
    table_data_75 += [result for result in results_2d_75]
    table_75 = AsciiTable(table_data_75)
    print('\n' + table_75.table)

if __name__ =='__main__':
    args = parser.parse_args()
    process(det_json=args.det_json, gt_json=args.gt_json, CLASSES=args.classes)
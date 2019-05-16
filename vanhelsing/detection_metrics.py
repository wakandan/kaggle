import numpy as np
import pandas as pd
from logger import *
from configurator import *

parser = MyArgumentParser(description='Calc precision recall for object detection')

parser.add_argument('--prediction',
                    help="prediction path, headers should be in ['fn', 'pr', 'xmin', 'ymin', 'xmax', 'ymax']")
parser.add_argument('--groundtruth', help='ground truth path, headers should be in [filename, bbox] format')
parser.add_argument('--threshold', type=float, default=0.5)


def iou_np(bboxes1, bboxes2):
    """
    boxes are in [x1, y1, x2, y2] format
    :param bboxes1:
    :param bboxes2:
    :return:
    """
    if bboxes1.size == 0 or bboxes2.size == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def detection_eval(gt_bboxes, pr_bboxes, iou_threshold=.5):
    """
    :param gt_bboxes:
    :param pr_bboxes:
    """
    frame_result = {"FP": 0, "TP": 0, "FN": 0, "predict": len(pr_bboxes),
                    "groundtruth": len(gt_bboxes)}
    if len(gt_bboxes) < 1:
        frame_result["FP"] = len(pr_bboxes)
        return frame_result
    if len(pr_bboxes) < 1:
        frame_result["FN"] = len(gt_bboxes)
        return frame_result
    # Start Evaluation
    result = iou_np(gt_bboxes, pr_bboxes)
    # Status of ground truth and prediction boxes
    gt_stat = len(gt_bboxes) * [True]
    pr_stat = len(pr_bboxes) * [True]
    # Distance matrix
    dist_map = np.array(result)
    # Sort the Most matched pairs
    dist_arg = np.argsort(dist_map.flatten("C"))
    # Get the sorted list
    dist_list = [np.unravel_index(x, dist_map.shape, "C") for x in dist_arg]
    cur_dst = iou_threshold + 1
    idx = len(dist_list) - 1
    while cur_dst >= iou_threshold and idx >= 0:
        gt_idx, pr_idx = dist_list[idx]
        cur_dst = dist_map[gt_idx][pr_idx]
        idx -= 1
        # pr and gt boxes must be not claimed by any box. and their iou must not smaller than iou thres.
        if (not gt_stat[gt_idx]) or (cur_dst < iou_threshold) or (not pr_stat[pr_idx]):
            continue

        gt_stat[gt_idx] = False
        pr_stat[pr_idx] = False
        frame_result["TP"] += 1

    for idx in range(len(gt_stat)):
        if not gt_stat[idx]:
            continue
        frame_result["FN"] += 1
    for idx in range(len(pr_stat)):
        if not pr_stat[idx]:
            continue
        frame_result["FP"] += 1
    return frame_result


if __name__ == '__main__':
    args = parser.parse_args()
    gt_bb_input = pd.read_csv(args.groundtruth, names=['fn', 'xmin', 'ymin', 'xmax', 'ymax', 'class'], delimiter=',')
    pr_input = pd.read_csv(args.prediction, names=['fn', 'pr', 'xmin', 'ymin', 'xmax', 'ymax'], delimiter=',')
    # gt_bb_input = pd.read_csv('/data/dataset/id-detection-data/validation/annotations_bbox.csv')
    # pr_input = pd.read_csv('/kdang/pytorch-ssd/eval_results/det_test_ID.txt', header=None,
    #                        names=['fn', 'pr', 'xmin', 'ymin', 'xmax', 'ymax'], delimiter=',')
    pr_input = pr_input[pr_input.pr > args.threshold]
    print(pr_input.head())
    gt = {}
    pr = {}
    total = 0
    for r in gt_bb_input.iterrows():
        r = r[1]
        bb = (r['xmin'], r['ymin'], r['xmax'], r['ymax'])

        print(r['fn'])
        filepath = r['fn'].split("/")
        filename = filepath[len(filepath)-1]
        if filename not in r:
            gt[filename] = []
        gt[filename].append(bb)
        total += 1
    gt = {k: np.array(v) for k, v in gt.items()}
    pr_input = pr_input.dropna()
    for r in pr_input.iterrows():
        r = r[1]
        # bb = list(map(lambda x: int(float(x)), r['bb'].split()))
        filename = r['fn']
        bb = (r.xmin, r.ymin, r.xmax, r.ymax)
        if filename not in pr:
            pr[filename] = []
        pr[filename].append(bb)
    pr = {k: np.array(v) for k, v in pr.items()}
    gt_fns = gt.keys()
    tp = 0
    fp = 0
    result = pd.DataFrame(columns=['fn', 'TP', 'FP', 'is_empty'])
    for fn in gt_fns:
        gt_bbs = gt[fn]
        data = {'fn': fn}

        if fn in pr:
            pr_bbs = pr[fn]
            evalu = detection_eval(gt_bbs, pr_bbs)
            print(f'fn = {fn}, evalu = {evalu}')
            tp += evalu['TP']
            fp += evalu['FP']
            data['TP'] = evalu['TP']
            data['FP'] = evalu['FP']
            data['is_empty'] = False
        else:
            print('empty result = {}'.format(fn))
            data['is_empty'] = True
        row = pd.Series(data)
        result = result.append(row, ignore_index=True)
    print(result.head())
    logging.debug(f'tp = {tp}, fp = {fp}')
    prec = tp * 100 / (tp + fp)
    reca = tp * 100 / total
    f1 = 2 * prec * reca / (prec + reca)
    logging.info('precision = {}'.format(prec))
    logging.info('recall = {}'.format(reca))
    logging.info('f1 = {}'.format(f1))
    result.to_csv('result.csv', index=False, header=True)

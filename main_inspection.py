from matplotlib.patches import Patch

from panopticapi.evaluation import PQStat
import os, sys
import numpy as np
import json
import time
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback, rgb2id

### CONFIGURATION ###
# Folder configuration
output_segmentation_path = '../COCO/output/segmentation'
output_detection_path =  '../COCO/output/detection'
output_detection_matterport_path = '../COCO/output/detection_matterport'
output_panoptic_path = '../COCO/output/panoptic'
### CONFIGURATION ###

OFFSET = 256 * 256 * 256
VOID = 0



# Find false positives among images and display
def find_fp(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            #raise Exception('no prediction for the image with id: {}'.format(image_id))
            continue ###########################################################################################################################ANDREA
        matched_annotations_list.append((gt_ann, pred_annotations[image_id], image_id)) #Ground truth annotation, prediction, image id

    with open('./classes/panoptic_coco_categories.json', 'r') as f:
        color_js = json.load(f)
        colors = {}
        for col in color_js:
            colors[col['id']]=col['color']

    # Find false positives for each image
    for g_truth, pred, img_id in matched_annotations_list:
        fp_img = pq_inspection(g_truth, pred, img_id, gt_folder, pred_folder, categories)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred['file_name'])), dtype=np.uint32)
        out = np.zeros(pan_pred.shape, dtype=np.uint8)

        if len(fp_img['fp'])==0:
            continue

        pan_pred = rgb2id(pan_pred)

        legend=[]
        for seg in fp_img['fp']:

            for s2 in pred['segments_info']:
                if s2['id']==seg:
                    out[pan_pred == seg, :] = colors[s2['category_id']]  #########
                    legend.append((categories[s2['category_id']]['name'],colors[s2['category_id']] ))
                    break


        #plot_fp(fp_img, )
        import matplotlib.pyplot as plt
        #plt.imshow(out)

        elems=[]
        for class_name, color in legend:
            elems.append(Patch(facecolor=(color[0]/255,color[1]/255,color[2]/255),label=class_name))
        plt.imshow(out)
        plt.legend(handles=elems)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_panoptic_path, str(img_id) + '_fp.png'), bbox_inches='tight')


    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

def pq_inspection(gt_ann, pred_ann, img_id, gt_folder, pred_folder, categories):
    pq_stat = PQStat()


    pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
    pan_gt = rgb2id(pan_gt)
    pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
    pan_pred = rgb2id(pan_pred)

    gt_segms = {el['id']: el for el in gt_ann['segments_info']}
    pred_segms = {el['id']: el for el in pred_ann['segments_info']}

    # predicted segments area calculation + prediction sanity checks
    pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
    labels, labels_cnt = np.unique(pan_pred, return_counts=True)
    for label, label_cnt in zip(labels, labels_cnt):
        if label not in pred_segms:
            if label == VOID:
                continue
            raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
        pred_segms[label]['area'] = label_cnt
        pred_labels_set.remove(label)
        if pred_segms[label]['category_id'] not in categories:
            raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
    if len(pred_labels_set) != 0:
        raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

    # confusion matrix calculation
    pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
    gt_pred_map = {}
    labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
    for label, intersection in zip(labels, labels_cnt):
        gt_id = label // OFFSET
        pred_id = label % OFFSET
        gt_pred_map[(gt_id, pred_id)] = intersection

    # count all matched pairs
    gt_matched = set()
    pred_matched = set()
    for label_tuple, intersection in gt_pred_map.items():
        gt_label, pred_label = label_tuple
        if gt_label not in gt_segms:
            continue
        if pred_label not in pred_segms:
            continue
        if gt_segms[gt_label]['iscrowd'] == 1:
            continue
        if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
            continue

        union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
        iou = intersection / union
        if iou > 0.5:
            pq_stat[gt_segms[gt_label]['category_id']].tp += 1
            pq_stat[gt_segms[gt_label]['category_id']].iou += iou
            gt_matched.add(gt_label)
            pred_matched.add(pred_label)

    # count false negatives
    crowd_labels_dict = {}
    for gt_label, gt_info in gt_segms.items():
        if gt_label in gt_matched:
            continue
        # crowd segments are ignored
        if gt_info['iscrowd'] == 1:
            crowd_labels_dict[gt_info['category_id']] = gt_label
            continue
        pq_stat[gt_info['category_id']].fn += 1

    # count false positives
    fp_list = []
    for pred_label, pred_info in pred_segms.items():
        if pred_label in pred_matched:
            continue
        # intersection of the segment with VOID
        intersection = gt_pred_map.get((VOID, pred_label), 0)
        # plus intersection with corresponding CROWD region if it exists
        if pred_info['category_id'] in crowd_labels_dict:
            intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
        # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
        if intersection / pred_info['area'] > 0.5:
            continue
        #Found a false positive
        pq_stat[pred_info['category_id']].fp += 1
        fp_list.append(pred_label) #Store id of the false positive

    res = {'img': img_id, 'fp':fp_list}
    return res


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time.strftime("Start date: %Y-%m-%d %H:%M:%S"))
    num_processes = 10   # number of processes where scheduling tasks
    input_images = '../COCO/images/val2017/'

    #Run inspection
    find_fp('../COCO/annotations/panoptic_val2017.json', output_panoptic_path + "/panoptic.json", '../COCO/annotations/panoptic_val2017', output_panoptic_path)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))


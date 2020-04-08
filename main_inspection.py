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





















def pq_inspection(gt_ann, pred_ann, gt_map, pred_map):
    """

    :param gt_ann: ground truth annotation (panoptic format, loaded from json)
    :param pred_ann: predicted annotation (panoptic format, loaded from json)
    :param gt_map: ground truth label map (from PNG annotation)
    :param pred_map: predicted truth label map (from PNG annotation)
    :return:
    """

    pq_stat = PQStat()

    gt_segms = {el['id']: el for el in gt_ann['segments_info']}
    pred_segms = {el['id']: el for el in pred_ann['segments_info']}

    # Predicted segments area calculation + prediction sanity checks
    pred_ids_set = set(pred_segms.keys())
    pred_map_ids, pred_map_ids_cnt = np.unique(pred_map, return_counts=True)
    for s_id, id_cnt in zip(pred_map_ids, pred_map_ids_cnt):
        if s_id not in pred_segms:
            if s_id == VOID:
                continue
            raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], s_id))
        pred_segms[s_id]['area'] = id_cnt
        pred_ids_set.remove(s_id)
    if len(pred_ids_set) != 0:
        raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_ids_set)))

    # Compute intersection between gt and pred objects
    pan_gt_pred = gt_map.astype(np.uint64) * OFFSET + pred_map.astype(np.uint64)
    gt_pred_intersect = {}    # Dictionary with (gt id, pred_id) as key. Number of pixels of the intersection as value.
    gtpred_ids, gtpred_ids_cnt = np.unique(pan_gt_pred, return_counts=True)
    for gtpred_id, intersection in zip(gtpred_ids, gtpred_ids_cnt):
        gt_id = gtpred_id // OFFSET
        pred_id = gtpred_id % OFFSET
        gt_pred_intersect[(gt_id, pred_id)] = intersection

    # Count all matched pairs (those with IoU>0.5 and same class)
    gt_matched = set()
    pred_matched = set()
    tp_list = []
    for label_tuple, intersection in gt_pred_intersect.items():
        gt_id, pred_id = label_tuple
        if gt_id not in gt_segms:
            continue
        if pred_id not in pred_segms:
            continue
        if gt_segms[gt_id]['iscrowd'] == 1:
            continue

        # Important: check that the two categories are the same
        if gt_segms[gt_id]['category_id'] != pred_segms[pred_id]['category_id']:
            continue

        # Compute IoU between segments
        union = pred_segms[pred_id]['area'] + gt_segms[gt_id]['area'] - intersection - gt_pred_intersect.get((VOID, pred_id), 0)
        iou = intersection / union
        if iou > 0.5:
            pq_stat[gt_segms[gt_id]['category_id']].tp += 1       # Add true positive for the corresponding class
            pq_stat[gt_segms[gt_id]['category_id']].iou += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)
            tp_list.append(int(pred_id)) # Add prediction to true positives

    # Count false negatives (i.e. Ground truth segments that are not matched)
    crowd_labels_dict = {}
    fn_list = []
    for gt_id, gt_info in gt_segms.items():
        if gt_id in gt_matched:
            continue
        # crowd segments are ignored
        if gt_info['iscrowd'] == 1:
            crowd_labels_dict[gt_info['category_id']] = gt_id
            continue
        pq_stat[gt_info['category_id']].fn += 1
        fn_list.append(gt_id)

    # Count false positives (i.e. Predicted samples that are not matched)
    fp_list = []
    for pred_id, pred_info in pred_segms.items():
        if pred_id in pred_matched:
            continue
        # Intersection of the segment with VOID
        intersection = gt_pred_intersect.get((VOID, pred_id), 0)
        # plus intersection with corresponding CROWD region if it exists
        if pred_info['category_id'] in crowd_labels_dict:
            intersection += gt_pred_intersect.get((crowd_labels_dict[pred_info['category_id']], pred_id), 0)
        # Predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
        if intersection / pred_info['area'] > 0.5:
            continue
        #Found a false positive
        pq_stat[pred_info['category_id']].fp += 1
        fp_list.append(pred_id) #Store id of the false positive
    pqs = pq_stat.get_pq()
    res = {'img_id':gt_ann['image_id'], 'fp':fp_list, 'tp':tp_list, 'pq':round(pqs['pq'],4), 'sq':round(pqs['sq'],4), 'rq':round(pqs['rq'],4)}
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


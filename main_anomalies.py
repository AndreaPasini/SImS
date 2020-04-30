import json
import os
from PIL import Image

from config import out_panoptic_json_path, out_panoptic_dir, \
    COCO_val_json_path, \
    COCO_ann_val_dir, kb_pairwise_json_path, anomaly_detection_dir, \
    out_panoptic_val_graphs_json_path, \
    pq_info_path, pairanomaly_kbfilter_json_path, \
    objectanomaly_kbfilter_json_path, objectanomaly_json_path, pairanomaly_json_path
from main_inspection import pq_inspection
import pyximport
pyximport.install(language_level=3)
from tqdm import tqdm
from panopticapi.utils import rgb2id
from semantic_analysis.anomaly_detection import inspect_anomalies, inspect_anomalies2
from semantic_analysis.knowledge_base import filter_kb_histograms, get_sup_ent_lists
from scipy.stats import entropy
import numpy as np




def compute_pq_stats(gt_json, pred_json, output_path):
    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    print("Analyzing predictions...")
    pbar = tqdm(total=len(gt_json['annotations']))

    pq_info = []
    # For each image in ground truth annotations
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        # Check if this ground-truth image has a prediction
        if image_id not in pred_annotations:
            print("Warning. Found an image that is not in predictions.")
            continue

        # Read PNG annotations (ground truth and predictinons)
        pred_ann = pred_annotations[image_id]
        gt_map = np.array(Image.open(os.path.join(COCO_ann_val_dir, gt_ann['file_name'])), dtype=np.uint32)
        gt_map = rgb2id(gt_map)
        pred_map = np.array(Image.open(os.path.join(out_panoptic_dir, pred_ann['file_name'])), dtype=np.uint32)
        pred_map = rgb2id(pred_map)

        # Find TP, FP, FN according to panoptic quality
        pq_info_i = pq_inspection(gt_ann, pred_ann, gt_map, pred_map)
        pq_info.append(pq_info_i)
        pbar.update()

    pbar.close()
    # Save results
    with open(output_path, 'w') as f:
        json.dump(pq_info, f)

def inspect_pairwise_anomalies(gt_json, pq_info_path, panoptic_graphs_json, kb_filtered, output_path):
    with open(pq_info_path, 'r') as f:
        pq_info_json = json.load(f)
    pq_info = {}
    for img in pq_info_json:
        pq_info[img['img_id']] = img

    # Dictionary to analyze the different configurations
    # both_fp : pairs of objects that are both false positives
    # tp_fp : pairs of objects that contain a false positive and a true positive
    # both_tp : pairs of objects that contain both true positives
    # fp_ignored : pairs of objects that contain a false positive and an ignored segment
    # ignored : pairs of objects with both ignored objects or a tp with an ignored object
    anomaly_stat = {}
    # Same keys, counts the pairs that are not associated to an histogram
    no_histogram = {}
    for k in ['both_fp', 'tp_fp', 'both_tp', 'fp_ignored', 'ignored']:
        anomaly_stat[k] = {'l': [], 'sup': [], 'entropy': []}
        no_histogram[k] = 0

    final_stats = {'with_histogram': anomaly_stat, 'no_histogram': no_histogram}
    panoptic_graphs = {g['graph']['name']: g for g in panoptic_graphs_json}
    print("Analyzing predictions...")
    pbar = tqdm(total=len(gt_json['annotations']))

    # For each image in ground truth annotations
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']

        # Read PNG annotations (ground truth and predictinons)
        pq_stat = pq_info[image_id]

        # Analyze image graph and compare with knowledge base. Fill anomaly_stat
        inspect_anomalies(panoptic_graphs[image_id], kb_filtered, pq_stat, anomaly_stat, no_histogram)

        pbar.update()

    pbar.close()

    if not os.path.isdir(anomaly_detection_dir):
        os.makedirs(anomaly_detection_dir)
    with open(output_path, "w") as f:
        json.dump(final_stats, f)

def analyze_object_anomalies(gt_json, pq_info_path, panoptic_graphs_json, kb_filtered, thr, output_path):
    with open(pq_info_path, 'r') as f:
        pq_info_json = json.load(f)
    pq_info = {}
    for img in pq_info_json:
        pq_info[img['img_id']] = img

    anomaly_stat = {}
    # Same keys, counts the pairs that are not associated to an histogram
    for k in ['tp', 'fp']:
        anomaly_stat[k] = {'n_anom': [], 'perc_anom': [], 'n_links': [], 'avg_l' : []}

    panoptic_graphs = {g['graph']['name']: g for g in panoptic_graphs_json}
    print("Analyzing predictions...")
    pbar = tqdm(total=len(gt_json['annotations']))

    # For each image in ground truth annotations
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']

        # Read PNG annotations (ground truth and predictinons)
        pq_stat = pq_info[image_id]

        # Analyze image graph and compare with knowledge base. Fill anomaly_stat
        inspect_anomalies2(panoptic_graphs[image_id], kb_filtered, pq_stat, anomaly_stat, thr)

        pbar.update()

    pbar.close()

    with open(output_path, "w") as f:
        json.dump(anomaly_stat, f)


if __name__ == "__main__":
    ### Choose methods to be run ###
    class RUN_CONFIG:
        compute_pq_stats = False
        analyze_pairwise_anomalies = False
        analyze_object_anomalies = True
        filter_kb = False   #Whether to filter KB with max entropy for detecting anomalies
        obj_anom_thr = 0.01 # Threshold for detecting an anomalous link on object anomalies
    # Load KB
    with open(kb_pairwise_json_path, 'r') as f:
        kb = json.load(f)
    # Get support and entropy
    sup, entr = get_sup_ent_lists(kb)
    max_entropy = entropy([1/3,1/3,1/3])
    med = np.median(np.log10(sup))
    min_sup = int(round(10**med))
    # Filter KB
    if RUN_CONFIG.filter_kb:
        kb_filtered = filter_kb_histograms(kb, min_sup, max_entropy)
    else:
        kb_filtered = filter_kb_histograms(kb, min_sup, 100)# No filter

    # Load ground truth (segmentations)
    with open(COCO_val_json_path, 'r') as f:
        gt_json = json.load(f)
    # Load predictions (panoptic CNN output)
    with open(out_panoptic_json_path, 'r') as f:
        pred_json = json.load(f)
    # Load graphs (predictions)
    with open(out_panoptic_val_graphs_json_path, 'r') as f:
        panoptic_graphs_json = json.load(f)

    if RUN_CONFIG.compute_pq_stats:
        # Panoptic quality of all objects in predicted graphs from the CNN
        # Create necessary file for 'analyze_pairwise_anomalies' and 'analyze_object_anomalies'
        compute_pq_stats(gt_json, pred_json, pq_info_path)
    if RUN_CONFIG.analyze_pairwise_anomalies:
        # Pairwise anomalies (find anomalous links in graph)
        # Stores in a json file 'likelihood', 'sup', 'entropy' of all the graphs links (of test CNN images)
        # l, sup, entropy are stored separately for both_fp, both_tp, tp_fp links, based on whether the objects of a link
        # are tp or fp. (both_tp should present higher likelihoods).
        if RUN_CONFIG.filter_kb:
            out_path = pairanomaly_kbfilter_json_path
        else:
            out_path = pairanomaly_json_path
        inspect_pairwise_anomalies(gt_json, pq_info_path, panoptic_graphs_json, kb_filtered, out_path)
    if RUN_CONFIG.analyze_object_anomalies:
        # Object-wise anomalies (for each object compute average likelihoods of its graph links)
        # Stores the json results in two lists: 'fp' and 'tp' objects (fp objects should present lower likelihoods)
        if RUN_CONFIG.filter_kb:
            out_path = f"{objectanomaly_kbfilter_json_path[:-5]}_{str(RUN_CONFIG.obj_anom_thr)[2:]}.json"
        else:
            out_path = f"{objectanomaly_json_path[:-5]}_{str(RUN_CONFIG.obj_anom_thr)[2:]}.json"
        analyze_object_anomalies(gt_json, pq_info_path, panoptic_graphs_json, kb_filtered, RUN_CONFIG.obj_anom_thr, out_path)
"""
 Author: Andrea Pasini
 This file provides the code for merging detection (MaskRCNN) and segmentation (Deeplab) to obtain panoptic segmentation

 Usage:
 - configure folders and models to be used (search for ### CONFIGURATION ### in this file)
 - run this file (main)

"""

import matplotlib

from config import COCO_panoptic_cat_info_path
from panopticapi.evaluation import pq_compute_pr
matplotlib.use('Agg')
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import mxnet as mx
import PIL.Image as Image
from os import listdir
import json
from multiprocessing import Pool
from maskrcnn.utils import extract_mask_bool
from panopticapi.utils import IdGenerator, id2rgb, load_panoptic_category_info

### CONFIGURATION ###
# Folder configuration
output_segmentation_path = '../COCO/output/segmentation'
output_detection_path =  '../COCO/output/detection'
output_detection_matterport_path = '../COCO/output/detection_matterport'
output_panoptic_dir = '../COCO/output/panoptic'
### CONFIGURATION ###

def build_panoptic_area(img_id, output_path, detection_path, segmentation_path):
    """
    Build panoptic segmentation of the specified image.
    Sort segments by area: write first smaller objects to avoid overlapping.
    :param img_id: image identifier (for retrieving file name)
    :param output_path: path to store panoptic segmentation results (png)
    :param detection_path: input path for detection
    :param segmentation_path: input path for semantic segmentation
    :return: the json annotation with class information for each panoptic segment.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #Read categories and create IdGenerator (official panopticapi repository)
    categories = load_panoptic_category_info()
    id_generator = IdGenerator(categories)

    #Parameters:
    overlap_thr = 0.5
    stuff_area_limit = 64 * 64

    #read segmentation data
    segm_probs = json.load(open(segmentation_path + '/' + img_id + '_prob.json', 'r'))
    segm_labelmap = np.array(Image.open(segmentation_path + '/' + img_id + '_0.png'), np.uint8)   #.labelmap.astype(np.uint8)).save()
    #read detection data
    detection = json.load(open(detection_path + '/' + img_id + '_prob.json','r'))


    pan_segm_id = np.zeros(segm_labelmap.shape, dtype=np.uint32)
    used = np.full(segm_labelmap.shape, False)

    annotation = {}
    try:
        annotation['image_id'] = int(img_id)
    except Exception:
        annotation['image_id'] = img_id

    annotation['file_name'] = img_id + '.png'

    segments_info = []


    for obj in detection: #for ann in ...
        obj_mask = extract_mask_bool(obj['mask'])
        obj_area = np.count_nonzero(obj_mask)
        obj['area']=obj_area
        obj['mask']=obj_mask

    detection.sort(key=lambda x: x['area'], reverse=False)##First smaller, than bigger


    for obj in detection: #for ann in ...
        obj_mask = obj['mask']#extract_mask_bool(obj['mask'])
        obj_area = obj['area']#np.count_nonzero(obj_mask)
        if obj_area == 0:
             continue
        #Filter out objects with intersection > 50% with used area
        intersection_mask = used & obj_mask
        intersect_area = np.count_nonzero(intersection_mask)
        if 1.0 * intersect_area / obj_area > overlap_thr:
            continue
        used = used | obj_mask

        segment_id = id_generator.get_id(obj['class'])
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = obj['class']
        if intersect_area>0:
            pan_segm_id[obj_mask & (~intersection_mask)] = segment_id
        else:
            pan_segm_id[obj_mask] = segment_id
        segments_info.append(panoptic_ann)

    #
    #
    for segm_class in np.unique(segm_labelmap):
        segm_class = int(segm_class)
        if segm_class==183: #void class
            continue

        #Check class: exclude non-stuff objects
        category = categories[segm_class]
        if category['isthing'] == 1:
            continue

        segm_mask = (segm_labelmap==segm_class)
        mask_left = segm_mask & (~used)
        # Filter out segments with small area
        if np.count_nonzero(mask_left) < stuff_area_limit:
            continue
        segment_id = id_generator.get_id(segm_class)
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = segm_class
        used = used | mask_left
        pan_segm_id[mask_left] = segment_id
        segments_info.append(panoptic_ann)

    annotation['segments_info'] = segments_info

    # Save annotated image
    Image.fromarray(id2rgb(pan_segm_id)).save(
         os.path.join(output_path, annotation['file_name'])
    )

    ##############
    ##remove segments with zero area
    ids = set(np.unique(pan_segm_id))
    segments_info_cleaned = []
    for seg in segments_info:
        if seg['id'] in ids:
            segments_info_cleaned.append(seg)
    annotation['segments_info'] = segments_info_cleaned
    ##################

    return annotation

def build_panoptic(img_id, output_path, detection_path, segmentation_path):
    """
    Build panoptic segmentation of the specified image.
    Sort segments by confidence: write first objects with higher detection confidence (standard behavior proposed by COCO).
    :param img_id: image identifier (for retrieving file name)
    :param output_path: path to store panoptic segmentation results (png)
    :param detection_path: input path for detection
    :param segmentation_path: input path for semantic segmentation
    :return: the json annotation with class information for each panoptic segment.
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #Read categories and create IdGenerator (official panopticapi repository)
    categories_json_file='./classes/panoptic_coco_categories.json'
    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {el['id']: el for el in categories_list}
    id_generator = IdGenerator(categories)

    #Parameters:
    overlap_thr = 0.5
    stuff_area_limit = 64 * 64

    #read segmentation data
    segm_probs = json.load(open(segmentation_path + '/' + img_id + '_prob.json', 'r'))
    segm_labelmap = np.array(Image.open(segmentation_path + '/' + img_id + '_0.png'), np.uint8)   #.labelmap.astype(np.uint8)).save()
    #read detection data
    detection = json.load(open(detection_path + '/' + img_id + '_prob.json','r'))


    pan_segm_id = np.zeros(segm_labelmap.shape, dtype=np.uint32)
    used = np.full(segm_labelmap.shape, False)

    annotation = {}
    try:
        annotation['image_id'] = int(img_id)
    except Exception:
        annotation['image_id'] = img_id

    annotation['file_name'] = img_id + '.png'

    segments_info = []


    for obj in detection: #for ann in ...
        obj_mask = extract_mask_bool(obj['mask'])
        obj_area = np.count_nonzero(obj_mask)
        if obj_area == 0:
             continue
        #Filter out objects with intersection > 50% with used area
        intersection_mask = used & obj_mask
        intersect_area = np.count_nonzero(intersection_mask)
        if 1.0 * intersect_area / obj_area > overlap_thr:
            continue
        used = used | obj_mask

        segment_id = id_generator.get_id(obj['class'])
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = obj['class']
        if intersect_area>0:
            pan_segm_id[obj_mask & (~intersection_mask)] = segment_id
        else:
            pan_segm_id[obj_mask] = segment_id
        segments_info.append(panoptic_ann)

    #
    #
    for segm_class in np.unique(segm_labelmap):
        segm_class = int(segm_class)
        if segm_class==183: #void class
            continue

        #Check class: exclude non-stuff objects
        category = categories[segm_class]
        if category['isthing'] == 1:
            continue

        segm_mask = (segm_labelmap==segm_class)
        mask_left = segm_mask & (~used)
        # Filter out segments with small area
        if np.count_nonzero(mask_left) < stuff_area_limit:
            continue
        segment_id = id_generator.get_id(segm_class)
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = segm_class
        used = used | mask_left
        pan_segm_id[mask_left] = segment_id
        segments_info.append(panoptic_ann)

    annotation['segments_info'] = segments_info

    # Save annotated image
    Image.fromarray(id2rgb(pan_segm_id)).save(
         os.path.join(output_path, annotation['file_name'])
    )

    return annotation

# Run tasks with multiprocessing
# chunk_size = number of images processed for each task
# input_path = path to input images
def run_tasks(num_processes, input_path, detection_path, segmentation_path):
    def update(x):
        pbar.update()

    files = sorted(listdir(input_path))
    pbar = tqdm(total=len(files))

    print("Number of images: %d" % len(files))
    print("Scheduling tasks...")

    panoptic_json = {}
    ann_list = []

    pool = Pool(num_processes)
    results = []
    for file in files:
        results.append(pool.apply_async(build_panoptic, args=(file.split('.')[0], output_panoptic_dir, detection_path, segmentation_path), callback=update))
    pool.close()
    pool.join()
    pbar.close()

    for res in results:
        ann_list.append(res.get())
    panoptic_json['annotations'] = ann_list
    with open(output_panoptic_dir + "panoptic.json", 'w') as f:
        json.dump(panoptic_json, f)
    print("Done")


if __name__ == "__main__":
    start_time = datetime.now()
    print("Building panoptic segmentation on validation images...")
    print(start_time.strftime("Start date: %Y-%m-%d %H:%M:%S"))
    num_processes = 10   # number of processes where scheduling tasks
    input_images = '../COCO/images/val2017/'

    # Set up results being merged
    detection_path = output_detection_matterport_path
    segmentation_path = output_segmentation_path

    # Execute merging operation
    run_tasks(num_processes, input_images, detection_path, segmentation_path)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

    #Run evaluation (Panoptic Quality: PQ)
    pq_compute_pr('../COCO/annotations/panoptic_val2017.json', output_panoptic_dir + "panoptic.json", '../COCO/annotations/panoptic_val2017', output_panoptic_dir)

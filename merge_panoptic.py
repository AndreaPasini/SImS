
import matplotlib

from panopticapi.evaluation import pq_compute, inspect, pq_compute2

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
from maskrcnn.instance_segmentation import extract_mask_bool
from panopticapi.utils import IdGenerator, id2rgb

'''

 Github repository for segmentation:  https://github.com/kazuto1011/deeplab-pytorch
 For instance segmentation: gluoncv (https://gluon-cv.mxnet.io/model_zoo/segmentation.html)

 Note for high performances (Xeon server):
 - mxnet-mkl does not work
 - using intel-numpy improves performances from 146 to 90 seconds per image

'''

output_segmentation_path = '../COCO/output/segmentation'
output_detection_path =  '../COCO/output/detection'
output_panoptic_path = '../COCO/output/panoptic'

def build_panoptic2(img_id, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #Read categories and create IdGenerator (official panopticapi repository)
    categories_json_file='./classes/panoptic_coco_categories.json'
    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {el['id']: el for el in categories_list}
    id_generator = IdGenerator(categories)

    #Parameters:
    overlap_thr = 0.9###########
    stuff_area_limit = 64 * 64

    #read segmentation data
    segm_probs = json.load(open(output_segmentation_path + '/' + img_id + '_prob.json', 'r'))
    segm_labelmap = np.array(Image.open(output_segmentation_path + '/' + img_id + '_0.png'), np.uint8)   #.labelmap.astype(np.uint8)).save()
    #read detection data
    detection = json.load(open(output_detection_path + '/' + img_id + '_prob.json','r'))


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




def build_panoptic(img_id, output_path):
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
    segm_probs = json.load(open(output_segmentation_path + '/' + img_id + '_prob.json', 'r'))
    segm_labelmap = np.array(Image.open(output_segmentation_path + '/' + img_id + '_0.png'), np.uint8)   #.labelmap.astype(np.uint8)).save()
    #read detection data
    detection = json.load(open(output_detection_path + '/' + img_id + '_prob.json','r'))


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
def run_tasks(input_path, num_processes):
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
        results.append(pool.apply_async(build_panoptic2, args=(file.split('.')[0], output_panoptic_path), callback=update))
    pool.close()
    pool.join()
    pbar.close()

    for res in results:
        ann_list.append(res.get())
    panoptic_json['annotations'] = ann_list
    with open(output_panoptic_path + "/panoptic.json", 'w') as f:
        json.dump(panoptic_json, f)
    print("Done")


if __name__ == "__main__":
    start_time = datetime.now()
    print("Building panoptic segmentation on validation images...")
    print(start_time.strftime("Start date: %Y-%m-%d %H:%M:%S"))
    num_processes = 10   # number of processes where scheduling tasks
    input_images = '../COCO/images/val2017/'



    run_tasks(input_images, num_processes)

    # panoptic_json = {}
    # ann_list = []
    # for file in listdir(output_detection_path):
    #     if (file.endswith('json')):
    #         ann_list.append(build_panoptic(file.split('_')[0], output_panoptic_path))
    # panoptic_json['annotations']=ann_list
    # with open(output_panoptic_path + "/panoptic.json", 'w') as f:
    #     json.dump(panoptic_json, f)


    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

    #Run evaluation
    pq_compute2('../COCO/annotations/panoptic_val2017.json', output_panoptic_path + "/panoptic.json", '../COCO/annotations/panoptic_val2017', output_panoptic_path)
    #inspect('../COCO/annotations/panoptic_val2017.json', output_panoptic_path + "/panoptic.json",
    #           '../COCO/annotations/panoptic_val2017', output_panoptic_path)
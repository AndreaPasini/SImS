
import matplotlib
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





def build_panoptic(img_id):

    #Parameters:
    overlap_thr = 0.5

    #read segmentation data
    segm_probs = json.load(open(output_segmentation_path + '/' + img_id + '_prob.json', 'r'))
    segm_labelmap = np.array(Image.open(output_segmentation_path + '/' + img_id + '_0.png'), np.uint8)   #.labelmap.astype(np.uint8)).save()
    #read detection data
    detection = json.load(open(output_detection_path + '/' + img_id + '_prob.json','r'))


    # pan_segm_id = np.zeros((segm_labelmap['height'], segm_labelmap['width']), dtype=np.uint32)
    used = np.full(segm_labelmap.shape, False)
    # annotation = {}
    # try:
    #     annotation['image_id'] = int(img_id)
    # except Exception:
    #     annotation['image_id'] = img_id
    #
    # annotation['file_name'] = img['file_name'].replace('.jpg', '.png')
    #
    # segments_info = []
    for obj in detection: #for ann in ...
        obj_mask = extract_mask_bool(obj['mask'])
        obj_area = np.count_nonzero(obj_mask)
        if obj_area == 0:
             continue
        #Filter out objects with intersection > 50% with used area
        intersect = np.count_nonzero(used & obj_mask)
        if 1.0 * intersect / obj_area > overlap_thr:
            continue
        used = used | obj_mask
    #
    #     mask = COCOmask.decode(ann['segmentation']) == 1
    #     if intersect != 0:
    #         mask = np.logical_and(pan_segm_id == 0, mask)
    #     segment_id = id_generator.get_id(ann['category_id'])
    #     panoptic_ann = {}
    #     panoptic_ann['id'] = segment_id
    #     panoptic_ann['category_id'] = ann['category_id']
    #     pan_segm_id[mask] = segment_id
    #     segments_info.append(panoptic_ann)
    #
    # for ann in sem_by_image[img_id]:
    #     mask = COCOmask.decode(ann['segmentation']) == 1
    #     mask_left = np.logical_and(pan_segm_id == 0, mask)
    #     if mask_left.sum() < stuff_area_limit:
    #         continue
    #     segment_id = id_generator.get_id(ann['category_id'])
    #     panoptic_ann = {}
    #     panoptic_ann['id'] = segment_id
    #     panoptic_ann['category_id'] = ann['category_id']
    #     pan_segm_id[mask_left] = segment_id
    #     segments_info.append(panoptic_ann)
    #
    # annotation['segments_info'] = segments_info
    # panoptic_json.append(annotation)

    # Image.fromarray(id2rgb(pan_segm_id)).save(
    #     os.path.join(segmentations_folder, annotation['file_name'])
    # )





































#Auxiliary functions for neural networks
def run_maskrcnn(img, outfile, maskrcnn):
    ids, scores, bboxes, masks = maskrcnn.predict(img)
    maskrcnn.save_json_boxes(img, ids, scores, bboxes, masks, outfile + '_prob.json')

def visualize_maskrcnn(img, outfile, maskrcnn):
    data = json.load(open(outfile + '_prob.json','r'))
    maskrcnn.visualize(img, data, outfile)

def run_deeplab(img, outfile, deeplab):
    img = img.asnumpy().astype('uint8')[...,::-1]
    #Compute predictions (keep information of the top-3 predicted classes for each pixel)
    labelmaps, probs = deeplab.predict_topk(img, k=3)
    #Save probabilities
    deeplab.save_json_probs(probs, outfile + '_prob.json')

    #Save result maps, one for each of the top-k predictions, (an image with the predicted class id)
    for i, labelmap in enumerate(labelmaps):
        Image.fromarray(labelmap.astype(np.uint8)).save(outfile+(('_%d.png')%i))

def visualize_deeplab(img, outfile, deeplab):
    img = img.asnumpy().astype('uint8')[..., ::-1]
    probs = json.load(open(outfile + '_prob.json','r'))

    #Save result maps
    i=0
    labelmap = np.array(Image.open(outfile + (('_%d.png') % i)), np.uint8)   #.labelmap.astype(np.uint8)).save()
    deeplab.visualize(img,labelmap,outfile+(('_%d_vis.png')%i), probs[i])

# Apply neural networks for segmentation and detection
def run_model(img_names, i, path):
    """
    :param img_names: list of image files being processed
    :param i: index of the first image being processed (= process images from i to i+len(img_names))
    :param path: input path for reading images
    """
    if not os.path.exists(output_segmentation_path):
        os.makedirs(output_segmentation_path)
    if not os.path.exists(output_detection_path):
        os.makedirs(output_detection_path)
    from deeplab.semantic_segmentation import Semantic_segmentation
    from maskrcnn.instance_segmentation import Instance_segmentation

    deeplab = Semantic_segmentation('./deeplab/configs/cocostuff164k.yaml',
                                    './deeplab/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth',
                                    './classes/deeplabToCoco.csv','./classes/panoptic.csv' )
    maskrcnn = Instance_segmentation('./classes/maskrcnnToCoco.csv')
    for img_name in img_names:
        img = mx.image.imread(path + img_name)
        img_name = img_name.split('.')[0]
        run_maskrcnn(img, output_detection_path + '/' + img_name, maskrcnn)
        run_deeplab(img, output_segmentation_path + '/' + img_name, deeplab)
        i+=1
    return 0

# Visualize segmentation and detection
def visualize_model(img_names, i, path):
    if not os.path.exists(output_segmentation_path):
        os.makedirs(output_segmentation_path)
    if not os.path.exists(output_detection_path):
        os.makedirs(output_detection_path)
    from deeplab.semantic_segmentation import Semantic_segmentation
    from maskrcnn.instance_segmentation import Instance_segmentation

    deeplab = Semantic_segmentation('./deeplab/configs/cocostuff164k.yaml',
                                    './deeplab/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth',
                                    './classes/deeplabToCoco.csv','./classes/panoptic.csv' )
    maskrcnn = Instance_segmentation('./classes/maskrcnnToCoco.csv')

    for img_name in img_names:
        img = mx.image.imread(path + img_name)
        img_name = img_name.split('.')[0]
        visualize_maskrcnn(img, output_detection_path + '/' + img_name, maskrcnn)
        visualize_deeplab(img, output_segmentation_path + '/' + img_name, deeplab)
        i += 1

    return 0

# Run tasks with multiprocessing
# chunk_size = number of images processed for each task
# input_path = path to input images
def run_tasks(chunck_size, input_path, num_processes):
    def update(x):
        pbar.update()

    files = sorted(listdir(input_path))
    chuncks = [files[x:x + chunck_size] for x in range(0, len(files), chunck_size)]
    nchuncks = len(chuncks)
    pbar = tqdm(total=nchuncks)

    print("Number of images: %d" % len(files))
    print("Number of tasks: %d (%d images per task)" % (nchuncks, chunck_size))
    print("Scheduling tasks...")

    pool = Pool(num_processes)
    for i in range(nchuncks):
        pool.apply_async(run_model, args=(chuncks[i], chunck_size*i, input_path), callback=update)
    pool.close()
    pool.join()
    pbar.close()

    print("Done")


if __name__ == "__main__":
    start_time = datetime.now()
    print("Building panoptic segmentation on validation images...")
    print(start_time.strftime("Start date: %Y-%m-%d %H:%M:%S"))
    chunck_size = 10    # number of images processed for each task
    num_processes = 10   # number of processes where scheduling tasks
    input_images = '../COCO/images/val2017/'



    files = set(sorted(listdir('../COCO/images/')))
    donefiles = set()
    for file in listdir('../COCO/output/detection/'):
        if file.endswith("png"):
            donefiles.add(file.split(".")[0]+".jpg")

    todo = files-donefiles

    #run_tasks(chunck_size, input_images, num_processes)
    #build_panoptic('000000183007')
    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))


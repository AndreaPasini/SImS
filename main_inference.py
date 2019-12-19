"""
 Author: Andrea Pasini
 This file provides the code for running inference with detection (MaskRCNN) and segmentation (Deeplab).
 Github repository for segmentation:  https://github.com/kazuto1011/deeplab-pytorch
 For instance segmentation: gluoncv (https://gluon-cv.mxnet.io/model_zoo/segmentation.html)

 Usage:
 - configure folders and models to be used (search for ### CONFIGURATION ### in this file)
 - run this file (main)

 Note for high performances (Xeon server):
 - mxnet-mkl does not work
 - using intel-numpy improves performances from 146 to 90 seconds per image

"""

from datetime import datetime
import matplotlib

from config import COCO_panoptic_cat_list_path

matplotlib.use('Agg')
import os
from tqdm import tqdm
import numpy as np
import mxnet as mx
import PIL.Image as Image
from os import listdir
import json
from multiprocessing import Pool
from deeplab.semantic_segmentation import Semantic_segmentation
from maskrcnn.instance_segmentation import Instance_segmentation
from maskrcnn_matterport.instance_segmentation import Instance_segmentation as Instance_seg_matterport

### CONFIGURATION ###
# Folder configuration
output_segmentation_path = '../COCO/output/segmentation'
output_detection_path =  '../COCO/output/detection'
output_detection_matterport_path = '../COCO/output/detection_matterport'
# Model configuration: select which models you want to run
use_deeplab=False
use_maskrcnn=False
use_maskrcnn_matterport=True
### CONFIGURATION ###


#Auxiliary functions for neural networks
def run_maskrcnn(img, outfile, maskrcnn):
    #Run maskrcnn (gluon implementation) and save result
    ids, scores, bboxes, masks = maskrcnn.predict(img)
    maskrcnn.save_json_boxes(img, ids, scores, bboxes, masks, outfile + '_prob.json')

def run_maskrcnn_matterport(img, outfile, maskrcnn):
    #Run maskrcnn (matterport implementation) and save result
    img = img.asnumpy().astype('uint8')
    prediction = maskrcnn.predict(img)
    maskrcnn.save_json_boxes(prediction, outfile + '_prob.json')
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
def run_model(img_names, path, use_deeplab=True, use_maskrcnn=True, use_maskrcnn_matterport=True):
    """
    :param img_names: list of image files being processed
    :param i: index of the first image being processed (= process images from i to i+len(img_names))
    :param path: input path for reading images
    """
    if not os.path.exists(output_segmentation_path):
        os.makedirs(output_segmentation_path)
    if not os.path.exists(output_detection_path):
        os.makedirs(output_detection_path)
    if not os.path.exists(output_detection_matterport_path):
        os.makedirs(output_detection_matterport_path)

    if use_deeplab:
        deeplab = Semantic_segmentation('./deeplab/configs/cocostuff164k.yaml',
                                        './deeplab/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth',
                                        './classes/deeplabToCoco.csv',COCO_panoptic_cat_list_path)
    if use_maskrcnn:
        maskrcnn = Instance_segmentation('./classes/maskrcnnToCoco.csv')
    if use_maskrcnn_matterport:
        maskrcnn_matterport = Instance_seg_matterport('./classes/maskrcnnToCoco.csv')

    for img_name in img_names:
        img = mx.image.imread(path + img_name)
        img_name = img_name.split('.')[0]
        if use_deeplab:
            run_deeplab(img, output_segmentation_path + '/' + img_name, deeplab)
        if use_maskrcnn:
            run_maskrcnn(img, output_detection_path + '/' + img_name, maskrcnn)
        if use_maskrcnn_matterport:
            try:
                run_maskrcnn_matterport(img, output_detection_matterport_path + '/' + img_name, maskrcnn_matterport)
            except:
                print("ex")
    return 0

# Visualize segmentation and detection
def visualize_model(img_names, i, path):
    if not os.path.exists(output_segmentation_path):
        os.makedirs(output_segmentation_path)
    if not os.path.exists(output_detection_path):
        os.makedirs(output_detection_path)
    if not os.path.exists(output_detection_matterport_path):
        os.makedirs(output_detection_matterport_path)

    from deeplab.semantic_segmentation import Semantic_segmentation
    from maskrcnn.instance_segmentation import Instance_segmentation
    from maskrcnn_matterport.instance_segmentation import Instance_segmentation as Instance_seg_matterport

    # deeplab = Semantic_segmentation('./deeplab/configs/cocostuff164k.yaml',
    #                                 './deeplab/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth',
    #                                 './classes/deeplabToCoco.csv','./classes/panoptic.csv' )
    maskrcnn = Instance_segmentation('./classes/maskrcnnToCoco.csv')

    for img_name in img_names:
        img = mx.image.imread(path + img_name)
        img_name = img_name.split('.')[0]
        visualize_maskrcnn(img, output_detection_path + '/' + img_name, maskrcnn)
        visualize_maskrcnn(img, output_detection_matterport_path + '/' + img_name, maskrcnn)
        visualize_deeplab(img, output_segmentation_path + '/' + img_name, deeplab)

    return 0

# Run tasks with multiprocessing
# chunk_size = number of images processed for each task
# input_path = path to input images
def run_tasks(chunck_size, input_path, num_processes, use_deeplab=True, use_maskrcnn=True, use_maskrcnn_matterport=True):
    def update(x):
        pbar.update()

    files = sorted(listdir(input_path))

    #start with recover... Use this code if the job was interrupted.
    # files=set(files)
    # done = set([f.split('_')[0]+'.jpg' for f in listdir(output_detection_matterport_path)])
    # files = files-done
    # files = list(files)
    # print("Done: %d" % len(done))
    # print("Todo: %d" % len(files))


    chuncks = [files[x:x + chunck_size] for x in range(0, len(files), chunck_size)]
    nchuncks = len(chuncks)
    pbar = tqdm(total=nchuncks)

    print("Number of images: %d" % len(files))
    print("Number of tasks: %d (%d images per task)" % (nchuncks, chunck_size))
    print("Scheduling tasks...")

    pool = Pool(num_processes)
    for i in range(nchuncks):
        pool.apply_async(run_model, args=(chuncks[i], input_path, use_deeplab, use_maskrcnn, use_maskrcnn_matterport), callback=update)
    pool.close()
    pool.join()
    pbar.close()

    print("Done")


if __name__ == "__main__":
    start_time = datetime.now()
    print("Running inference on validation images...")
    print(start_time.strftime("Start date: %Y-%m-%d %H:%M:%S"))
    chunck_size = 10    # number of images processed for each task
    num_processes = 10  # number of processes where scheduling tasks
    input_images = '../COCO/images/val2017/'
    #Run tasks: select which neural networks you'd like to run
    run_tasks(chunck_size, input_images, num_processes, use_deeplab=False, use_maskrcnn=False, use_maskrcnn_matterport=True)
    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

    #Uncomment to visualize results
    #visualize_model(sorted(listdir('../COCO/images/val2017/')), 0, '../COCO/images/val2017/')

from itertools import groupby

import matplotlib
matplotlib.use('Agg')

import os

import time
from tqdm import tqdm
import numpy as np
import argparse
import mxnet as mx
from mxnet import gluon, random
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete, plot_image
from gluoncv.data.transforms.presets.ssd import load_test
from gluoncv.utils.viz import plot_bbox

import threading
import PIL.Image as Image
import matplotlib.pyplot as plt
import json
from multiprocessing import Pool
'''

 Github repository for segmentation:  https://github.com/kazuto1011/deeplab-pytorch
 For instance segmentation: gluoncv (https://gluon-cv.mxnet.io/model_zoo/segmentation.html)

 Note for high performances (Xeon server):
 - mxnet-mkl does not work
 - using intel-numpy improves performances from 146 to 90 seconds per image

'''

output_segmentation_path = '../COCO/output/segmentation'
output_detection_path =  '../COCO/output/detection'

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


def run_model(img_names, i, path):
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
    #print("\n+ New task. Images: %d-%d" % (i,i+len(img_names)-1))
    for img_name in img_names:
        # img = mx.image.imread(path + img_name)
        # img_name = img_name.split('.')[0]
        #run_maskrcnn(img, output_detection_path + '/' + img_name, maskrcnn)
        #run_deeplab(img, output_segmentation_path + '/' + img_name, deeplab)
        i+=1

    for img_name in img_names:
        # img = mx.image.imread(path + img_name)
        # img_name = img_name.split('.')[0]
        #visualize_maskrcnn(img, output_detection_path + '/' + img_name, maskrcnn)
        #visualize_deeplab(img, output_segmentation_path + '/' + img_name, deeplab)
        i += 1

    return 0

class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)








def printResult(result):
    print(result)


def dummy(a, b, c):
    return 1


def run_threads():
    from os import listdir


    # images = ['../COCO/images/train2017/000000000009.jpg', '../COCO/images/train2017/000000000025.jpg',
    #           '../COCO/images/train2017/000000000030.jpg']

    # jobs = []
    # for i,img in enumerate(images):
    #     t = FuncThread(run_model, img, i)
    #     jobs.append(t)
    #
    # # Start the threads (i.e. calculate the random number lists)
    # for j in jobs:
    #     j.start()
    #
    # # Ensure all of the threads have finished
    # for j in jobs:
    #     j.join()

    parallel=True

    if parallel:

        def update(x):
            pbar.update()

        files = listdir('../COCO/images/val2017/')
        chunck_size = 10
        chuncks = [files[x:x + chunck_size] for x in range(0, len(files), chunck_size)]
        nchuncks = len(chuncks)
        pbar = tqdm(total=nchuncks)

        print("Number of images: %d" % len(files))
        print("Number of tasks: %d (%d images per task)" % (nchuncks, chunck_size))
        print("Scheduling tasks...")

        pool = Pool(4)
        for i in range(nchuncks):
            pool.apply_async(run_model, args=(chuncks[i], chunck_size*i, '../COCO/images/val2017/'), callback=update)
        pool.close()
        pool.join()
        pbar.close()

        print("Done")

    else:
        run_model(['3.jpg'], 0, '../COCO/images/')


if __name__ == "__main__":
    start = time.time()
    run_threads()

    end = time.time()
    delta = end - start
    print("time: " + str(delta))

    quit()


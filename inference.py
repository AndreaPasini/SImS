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


    print("Loading models...")
    deeplab = Semantic_segmentation('./deeplab/configs/cocostuff164k.yaml',
                                    './deeplab/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth',
                                    './classes/deeplabToCoco.csv','./classes/panoptic.csv' )
    maskrcnn = Instance_segmentation('./classes/maskrcnnToCoco.csv')

    start = time.time()
    for img_name in img_names:
        img = mx.image.imread(path + img_name)
        img_name = img_name.split('.')[0]
        print("+ Run detection (%d)" % i)
        #run_maskrcnn(img, output_detection_path + '/' + img_name, maskrcnn)
        print("- End detection (%d)" % i)
        print("+ Run segmentation (%d)" % i)
        #run_deeplab(img, output_segmentation_path + '/' + img_name, deeplab)
        print("- End segmentation (%d)" % i)
        end = time.time()
        i+=1
    delta = end - start
    print("time (%d): %d" % (i, delta))

    start = time.time()
    for img_name in img_names:
        img = mx.image.imread(path + img_name)
        img_name = img_name.split('.')[0]
        print("Run detection (%d)" % i)
        #visualize_maskrcnn(img, output_detection_path + '/' + img_name, maskrcnn)
        print("End detection (%d)" % i)
        print("Start segmentation (%d)" % i)
        #visualize_deeplab(img, output_segmentation_path + '/' + img_name, deeplab)
        print("End segmentation (%d)" % i)
        end = time.time()
        i += 1
    delta = end - start

    print("time (%d): %d" % (i, delta))

class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)


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
        # files = listdir('../COCO/images/train2017/')
        # for i in range(0,4):
        #     newpid = os.fork()
        #     if newpid == 0:
        #         print("Fork process: %d to %d" %(2*i,2*i+2))
        #         run_model(files[2*i:2*i+2],2*i, '../COCO/images/train2017/')
        #         break



        ############################

        import time
        import multiprocessing

        # def basic_func(x):
        #     if x == 0:
        #         return 'zero'
        #     elif x % 2 == 0:
        #         return 'even'
        #     else:
        #         return 'odd'

        # def multiprocessing_func(x):
        #     y = x * x
        #     time.sleep(2)
        #     print('{} squared results in a/an {} number'.format(x, basic_func(y)))

        # starttime = time.time()
        # processes = []
        # files = listdir('../COCO/images/train2017/')
        # for i in range(0, 4):
        #     p = multiprocessing.Process(target=run_model, args=(files[2*i:2*i+2],2*i, '../COCO/images/train2017/'))
        #     processes.append(p)
        #     p.start()
        #
        # for process in processes:
        #     process.join()
        #
        # print('That took {} seconds'.format(time.time() - starttime))

        ntasks=8

        pbar = tqdm(total=ntasks)

        def update(*a):
            pbar.update()
            print("done")

        files = listdir('../COCO/images/train2017/')
        pool = Pool(4)
        for i in range(ntasks):
            pool.apply_async(run_model, args=(files[i], i, '../COCO/images/train2017/'), callback=update)
        pool.close()
        pool.join()
        ######################




    else:
        run_model(['3.jpg'], 0, '../COCO/images/')


if __name__ == "__main__":
    start = time.time()
    run_threads()

    end = time.time()
    delta = end - start
    print("time: " + str(delta))

    quit()


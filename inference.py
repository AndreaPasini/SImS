import os


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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


'''

 Github repository for segmentation:  https://github.com/kazuto1011/deeplab-pytorch
 For instance segmentation: gluoncv (https://gluon-cv.mxnet.io/model_zoo/segmentation.html)

 Note for high performances (Xeon server):
 - mxnet-mkl does not work
 - using intel-numpy improves performances from 146 to 90 seconds per image

'''


def test_maskrcnn(img, outfile, maskrcnn):
    ids, scores, bboxes, masks = maskrcnn.predict(img)
    maskrcnn.visualize(img,  ids, scores, bboxes, masks, outfile)

def test_deeplab(img, outfile, deeplab):
    img = img.asnumpy().astype('uint8')[...,::-1]
    labelmap = deeplab.predict(img)
    deeplab.visualize(img,labelmap,outfile)

def run_model(img_name, i):
    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    from deeplab.semantic_segmentation import Semantic_segmentation
    from maskrcnn.instance_segmentation import Instance_segmentation
    deeplab = Semantic_segmentation('./deeplab/configs/cocostuff164k.yaml',
                                    './deeplab/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth')
    maskrcnn = Instance_segmentation()

    # images = ['../COCO/images/1.jpg','../COCO/images/2.jpg','../COCO/images/3.jpg','../COCO/images/4.jpg']
    images = ['../COCO/images/train2017/000000000009.jpg', '../COCO/images/train2017/000000000025.jpg',
              '../COCO/images/train2017/000000000030.jpg']

    import time

    start = time.time()

    img = mx.image.imread(img_name)

    print("start mxnet (%d)"% (i))
    test_maskrcnn(img, outdir + '/mask_segment_%d.jpg' % (i), maskrcnn)
    print("end mxnet (%d), start tensorflow"% (i))
    test_deeplab(img, outdir + '/sem_segment_%d.jpg' % (i), deeplab)
    print("Done, %d" % i)

    end = time.time()
    delta = end - start
    print("time: " + str(delta))


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)













def run_threads():
    images = ['../COCO/images/train2017/000000000009.jpg', '../COCO/images/train2017/000000000025.jpg',
              '../COCO/images/train2017/000000000030.jpg']

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

    for i, img in enumerate(images):
        newpid = os.fork()
        if newpid == 0:
            run_model(img,i)
            break





if __name__ == "__main__":
    # #Execute inference, semantic segmentation on coco
    # #python inference.py --dataset coco --model-zoo psp_resnet101_coco --model-zoo fcn_resnet50_ade --eval
    #
    # # output folder
    # outdir = 'outdir'
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    # from deeplab.semantic_segmentation import Semantic_segmentation
    # from maskrcnn.instance_segmentation import Instance_segmentation
    # deeplab = Semantic_segmentation('./deeplab/configs/cocostuff164k.yaml', './deeplab/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth')
    # maskrcnn = Instance_segmentation()
    #
    # #images = ['../COCO/images/1.jpg','../COCO/images/2.jpg','../COCO/images/3.jpg','../COCO/images/4.jpg']
    # images = ['../COCO/images/train2017/000000000009.jpg','../COCO/images/train2017/000000000025.jpg','../COCO/images/train2017/000000000030.jpg']
    #
    # import time
    #
    #
    # sum =0
    # count=0
    # for i,img_name in enumerate(images):
    #     start = time.time()
    #
    #     img = mx.image.imread(img_name)
    #
    #     test_maskrcnn(img, outdir + '/mask_segment_%d.jpg' % (i))
    #     test_deeplab(img, outdir + '/sem_segment_%d.jpg' % (i))
    #     print("Done, %d" % i)
    #
    #     end = time.time()
    #     delta=end - start
    #     sum+=delta
    #     count+=1
    #     print("time: " + str(delta))
    # print("average per image: " + str(1.0*sum/count)+" s")




    import time

    start = time.time()
    run_threads()

    end = time.time()
    delta = end - start
    print("time: " + str(delta))




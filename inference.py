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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


'''

 Github repository for segmentation:  https://github.com/kazuto1011/deeplab-pytorch
 For instance segmentation: gluoncv (https://gluon-cv.mxnet.io/model_zoo/segmentation.html)


'''


def test_maskrcnn(img, outfile):
    ids, scores, bboxes, masks = maskrcnn.predict(img)
    maskrcnn.visualize(img,  ids, scores, bboxes, masks, outfile)

def test_deeplab(img, outfile):
    img = img.asnumpy().astype('uint8')[...,::-1]
    labelmap = deeplab.predict(img)
    deeplab.visualize(img,labelmap,outfile)


if __name__ == "__main__":
    #Execute inference, semantic segmentation on coco
    #python inference.py --dataset coco --model-zoo psp_resnet101_coco --model-zoo fcn_resnet50_ade --eval

    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    from deeplab.semantic_segmentation import Semantic_segmentation
    from maskrcnn.instance_segmentation import Instance_segmentation
    deeplab = Semantic_segmentation('./deeplab/configs/cocostuff164k.yaml', './deeplab/data/models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth')
    maskrcnn = Instance_segmentation()

    #images = ['../COCO/images/1.jpg','../COCO/images/2.jpg','../COCO/images/3.jpg','../COCO/images/4.jpg']
    images = ['../COCO/images/train2017/000000000009.jpg','../COCO/images/train2017/000000000025.jpg','../COCO/images/train2017/000000000030.jpg']

    for i,img_name in enumerate(images):
        img = mx.image.imread(img_name)

        test_maskrcnn(img, outdir + '/mask_segment_%d.jpg' % (i))
        test_deeplab(img, outdir + '/sem_segment_%d.jpg' % (i))
        print("Done, %d" % i)


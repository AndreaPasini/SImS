#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from addict import Dict

from deeplab.libs.models import *



def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("(Deeplab) Device:", torch.cuda.get_device_name(current_device))
    else:
        print("(Deeplab) Device: CPU")
    return device

from deeplab.demo import get_classtable
from deeplab.demo import setup_postprocessor
from deeplab.demo import preprocessing


class Semantic_segmentation:
    """
    Semantic segmentation model (DeeplabV2). Trained on COCO STUFF dataset.
    https://github.com/kazuto1011/deeplab-pytorch
    """

    __cuda = True   #enable GPU if available
    __crf = True    #False   #use CRF post processing
    __device = None
    __CONFIG = None
    __postprocessor = None
    __model = None
    __classes = None                #map between deeplab classes id and their labels
    __panoptic_classes = None       #id:label for coco panoptic classes
    __deeplab_to_coco = None        #deeplab to coco classes conversion (by id)

    def build_class_file(self, coco_stuff_classes_path, coco_thing_classes_path, coco_merged_path, output_conversion_path, output_panoptic_path):
        """
        Generates a file with the map between deeplab classes and COCO classes
        :param coco_stuff_classes_path: input csv with COCO stuff classes and ids
        :param coco_thing_classes_path: input csv with COCO thing classes and ids
        :param coco_merged_path: input csv with COCO merged classes
        """
        """ ('./classes/cocoStuff.csv','./classes/cocoThing.csv','./classes/cocoMerged.csv','./classes/deeplabToCoco.csv','./classes/panoptic.csv') """

        coco_panoptic = {}

        # read coco thing classes and their ids
        coco_thing_classes = {}
        with open(coco_thing_classes_path) as f:
            for line in f.readlines():
                id, label = line.rstrip('\n').split(":")
                coco_thing_classes[label] = id
                coco_panoptic[label] = id

        # read coco stuff classes and their ids
        coco_stuff_classes = {}  #Map from class name to class id (COCO)
        with open(coco_stuff_classes_path) as f:
            for line in f.readlines():
                id, label = line.rstrip('\n').split(":")
                coco_stuff_classes[label] = id
                coco_panoptic[label]=id

        # read merged classes (stuff)
        with open(coco_merged_path) as f:
            for line in f.readlines():
                merged_label, labels = line.rstrip('\n').split(":")
                merged_id = coco_stuff_classes[merged_label]
                for label in labels.strip().split(','):
                    coco_stuff_classes[label] = merged_id

        # map deeplab classes to coco stuff
        deeplab_to_coco = {}
        for id, label in self.__classes.items():
            try:
                deeplab_to_coco[id] = coco_stuff_classes[label]
            except:
                try:
                    deeplab_to_coco[id] = coco_thing_classes[label]
                except:
                    deeplab_to_coco[id] = coco_stuff_classes['other']
        #Save mappings to file
        with open(output_conversion_path, 'w') as f:
            for k,v in deeplab_to_coco.items():
                f.write('%s:%s\n'%(k,v))

        #Save panoptic classes:
        with open(output_panoptic_path, 'w') as f:
            for k, v in coco_panoptic.items():
                f.write('%s:%s\n' % (v, k))


    def __init__(self, config_path, model_path, deeplab_to_coco_path, panoptic_classes_path):
        """
        Initialize the semantic segmentation model (DeeplabV2). Trained on COCO STUFF dataset.
        :param config_path: input configuration file
        :param model_path: input model weights path
        :param coco_stuff_classes_path: input csv with COCO stuff classes and ids
        """

        self.__CONFIG = Dict(yaml.safe_load(open(config_path)))  ####
        self.__device = get_device(self.__cuda)  # true, enable if available
        torch.set_grad_enabled(False)

        self.__classes = get_classtable(self.__CONFIG)
        self.__postprocessor = setup_postprocessor(self.__CONFIG) if self.__crf else None

        self.__model = eval(self.__CONFIG.MODEL.NAME)(n_classes=self.__CONFIG.DATASET.N_CLASSES)
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)  ########
        self.__model.load_state_dict(state_dict)
        self.__model.eval()
        self.__model.to(self.__device)

        # Read deeplab to coco classes conversion
        self.__deeplab_to_coco = {}
        with open(deeplab_to_coco_path, 'r') as f:
            for line in f.readlines():
                deeplab_id, coco_id = line.rstrip('\n').split(":")
                self.__deeplab_to_coco[int(deeplab_id)] = int(coco_id)

        # Read COCO panoptic classes
        self.__panoptic_classes = {}
        with open(panoptic_classes_path) as f:
            for line in f.readlines():
                id, label = line.rstrip('\n').split(":")
                self.__panoptic_classes[int(id)] = label

        print("(Deeplab) Model:", self.__CONFIG.MODEL.NAME)


    def __inference(self, model, image, raw_image=None, postprocessor=None):
        """
        Run inference on the specified image.
        :return: 2D matrix with labels (panoptic COCO ids)
        """
        _, _, H, W = image.shape

        # Image -> Probability map
        logits = model(image)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0]
        probs = probs.cpu().numpy()

        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            probs = postprocessor(raw_image, probs)

        labelmap = np.argmax(probs, axis=0)
        def f_deeplab_to_coco(c):
            return self.__deeplab_to_coco[c]
        fvectorized = np.vectorize(f_deeplab_to_coco)
        #Convert to coco class ids:
        labelmapCOCO = fvectorized(labelmap)

        return labelmapCOCO

    def __inference_topk(self, model, image, k, raw_image=None, postprocessor=None):
        """
        Run inference on the specified image.
        :return: k 2D matrices with labels (panoptic COCO ids) -> the matrices are sorted by
                decreasing confidence on the top-k classes for each pixel
                and a list of dictionaries (one for each k) with the average probabilities of the objects ([classid]->avgprob)
        """
        _, _, H, W = image.shape

        # Image -> Probability map
        logits = model(image)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0]
        probs = probs.cpu().numpy()

        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            probs = postprocessor(raw_image, probs)



        # Take top-k classes
        labelmaps = probs.argsort(axis=0)[-k:][::-1]
        labelmaps_objects=[]
        for labelmap in labelmaps:
            labelmap_objects = {}
            unique_classes = np.unique(labelmap)
            for class_id in unique_classes:
                avgprob = np.average(probs[class_id][labelmap==class_id])
                labelmap_objects[self.__deeplab_to_coco[class_id]]=avgprob
            labelmaps_objects.append(labelmap_objects)

        def f_deeplab_to_coco(c):
            return self.__deeplab_to_coco[c]
        fvectorized = np.vectorize(f_deeplab_to_coco)
        #Convert to coco class ids:
        labelmapsCOCO = fvectorized(labelmaps)

        return labelmapsCOCO, labelmaps_objects

    def predict(self, image):
        """
        Inference from a single image.
        :param image: must follow the format -> mx.image.imread(path + img_name).asnumpy().astype('uint8')[...,::-1]
        :return: 2D matrix with labels (panoptic COCO ids)
        """

        h,w,d = image.shape

        # Inference
        image, raw_image = preprocessing(image, self.__device, self.__CONFIG)   ## Resize to [h,w=513]
        labelmap = self.__inference(self.__model, image, raw_image, self.__postprocessor) #return labelmap

        ## Resize to original image dimensions
        return cv2.resize(labelmap, (w,h), interpolation=cv2.INTER_NEAREST)

    def predict_topk(self, image, k):
        """
        Inference from a single image.
        :param image: must follow the format -> mx.image.imread(path + img_name).asnumpy().astype('uint8')[...,::-1]
        :return: 2D matrices with labels (panoptic COCO ids). Matrices represent the top-k classes for each pixel. Matrices are sorted by decreasing confidence
        """

        h,w,d = image.shape

        # Inference
        image, raw_image = preprocessing(image, self.__device, self.__CONFIG)   ## Resize to [h,w=513]
        labelmaps, probs = self.__inference_topk(self.__model, image, k, raw_image, self.__postprocessor) #return labelmap

        ## Resize to original image dimensions
        res = []
        for labelmap in labelmaps:
            res.append(cv2.resize(labelmap, (w,h), interpolation=cv2.INTER_NEAREST))
        return res, probs


    def visualize(self, image, labelmap, outfile, probs):

        labels = np.unique(labelmap)

        # Show result for each class
        rows = np.floor(np.sqrt(len(labels) + 1))
        cols = np.ceil((len(labels) + 1) / rows)

        plt.figure(figsize=[10,6])
        ax = plt.subplot(rows, cols, 1)
        ax.set_title("Input image")
        ax.imshow(image[:, :, ::-1])
        ax.axis("off")

        for i, label in enumerate(labels):
            mask = labelmap == label
            ax = plt.subplot(rows, cols, i + 2)
            ax.set_title(self.__panoptic_classes[label]+(" (%.2f)"%probs[label]))
            ax.imshow(image[..., ::-1])
            ax.imshow(mask.astype(np.float32), alpha=0.5)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()







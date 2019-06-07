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

from .demo import get_classtable
from .demo import setup_postprocessor
from .demo import preprocessing


class Semantic_segmentation:
    """
    Semantic segmentation model (DeeplabV2). Trained on COCO STUFF dataset.
    https://github.com/kazuto1011/deeplab-pytorch
    """

    __cuda = True   #enable GPU if available
    __crf = True #False   #use CRF post processing
    __device = None
    __CONFIG = None
    __postprocessor = None
    __model = None
    __classes = None

    def __init__(self, config_path, model_path):
        """
        Initialize the semantic segmentation model (DeeplabV2). Trained on COCO STUFF dataset.
        :param config_path: input configuration file
        :param model_path: input model weights path
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
        print("(Deeplab) Model:", self.__CONFIG.MODEL.NAME)


    def inference(self, model, image, raw_image=None, postprocessor=None):
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

        return labelmap


    def predict(self, image):
        """
        Inference from a single image
        """

        h,w,d = image.shape

        # Inference
        image, raw_image = preprocessing(image, self.__device, self.__CONFIG)   ## Resize to [h,w=513]
        labelmap = self.inference(self.__model, image, raw_image, self.__postprocessor) #return labelmap

        ## Resize to original image dimensions
        return cv2.resize(labelmap, (w,h), interpolation=cv2.INTER_NEAREST)


    def visualize(self, image, labelmap, outfile):
        labels = np.unique(labelmap)

        # Show result for each class
        rows = np.floor(np.sqrt(len(labels) + 1))
        cols = np.ceil((len(labels) + 1) / rows)

        plt.figure()
        ax = plt.subplot(rows, cols, 1)
        ax.set_title("Input image")
        ax.imshow(image[:, :, ::-1])
        ax.axis("off")

        for i, label in enumerate(labels):
            mask = labelmap == label
            ax = plt.subplot(rows, cols, i + 2)
            ax.set_title(self.__classes[label])
            ax.imshow(image[..., ::-1])
            ax.imshow(mask.astype(np.float32), alpha=0.5)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()





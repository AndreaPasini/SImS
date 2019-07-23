#from maskrcnn_matterport.samples.coco.coco import CocoDataset, build_coco_results
import skimage
import numpy as np

from maskrcnn.utils import compress_mask
from maskrcnn_matterport.mrcnn import model as modellib, utils
from maskrcnn_matterport.mrcnn.config import Config
from maskrcnn_matterport.mrcnn.utils import download_trained_weights
import os
import json

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

class Instance_segmentation:
    """
    Instance segmentation model (MaskRCNN). Trained on COCO THINGS dataset.
    Matterport implementation (https://github.com/matterport/Mask_RCNN)
    """

    __model = None
    __maskrcnn_to_coco = None        #maskrcnn to coco classes conversion (by id)

    __model_file = 'mask_rcnn_coco.h5'
    __model_path = './maskrcnn_matterport/data/'

    def __init__(self, maskrcnn_to_coco_path):
        """
        Initialize the instance segmentation model (MaskRCNN). Trained on COCO THING dataset.
        """
        config = InferenceConfig()
        #config.display()

        #Download pretrained weights if not available
        path_to_model = self.__model_path + self.__model_file
        if not os.path.isfile(path_to_model):
            if not os.path.isdir(self.__model_path):
                os.makedirs(self.__model_path)
            download_trained_weights(path_to_model)

        #Load weights
        self.__model = modellib.MaskRCNN(mode="inference", config=config, model_dir=path_to_model)
        self.__model.load_weights(path_to_model, by_name=True)

        # Read maskrcnn to coco classes conversion
        self.__maskrcnn_to_coco = {}
        with open(maskrcnn_to_coco_path, 'r') as f:
            for line in f.readlines():
                maskrcnn_id, coco_id = line.rstrip('\n').split(":")
                self.__maskrcnn_to_coco[int(maskrcnn_id)] = int(coco_id)

    def predict(self, image):
        """
        Inference from a single image.
        :param image: must follow the format -> mx.image.imread(path + img_name).asnumpy().astype('uint8')
        :return: result dictionary with: {rois, class_ids, scores, masks}. Rois: (y1, x1, y2, x2)
        """

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        detection = self.__model.detect([image], verbose=0)[0]

        # Map maskrcnn ids to COCO ids
        def f_maskrcnn_to_coco(c):
            if c >= 0:
                return self.__maskrcnn_to_coco[c-1] #This is different to GLUON implementation (c-1 because class 0  is background)
            else:
                return c

        fvectorized = np.vectorize(f_maskrcnn_to_coco)

        # Convert to coco class ids:
        if detection['class_ids'].size!=0:  #necessary to avoid exceptions
            detection['class_ids'] = fvectorized(detection['class_ids'])

        return detection

    def save_json_boxes(self, prediction, output_path, thresh=0.5):
        """
        Save bounding boxes to json file
        Format:
        """

        rois = prediction['rois']
        class_ids = prediction['class_ids']
        scores = prediction['scores']
        masks = prediction['masks']

        # Loop through detections
        jsonout = []
        for i in range(rois.shape[0]):
            id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            if score < thresh or id < 0:
                continue

            printable_bbox = [int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]  # xmin, ymin, xmax, ymax
            # compress mask with RLE
            cmask = compress_mask(mask)
            jsonout.append({'class': int(id), 'score': ('%.3f' % score), 'bbox': printable_bbox, 'mask': cmask})

        # Save file
        with open(output_path, 'w') as f:
            json.dump(jsonout, f)

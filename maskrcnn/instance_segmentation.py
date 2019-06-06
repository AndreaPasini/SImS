from gluoncv.data.transforms.presets.rcnn import transform_test
from gluoncv.model_zoo import get_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gluoncv.utils.viz import plot_bbox
import numpy as np




class Instance_segmentation:
    """
    Instance segmentation model (MaskRCNN). Trained on COCO THINGS dataset.
    gluoncv (https://gluon-cv.mxnet.io/model_zoo/segmentation.html)
    """

    __model = None

    def __init__(self):
        """
        Initialize the instance segmentation model (MaskRCNN). Trained on COCO THING dataset.
        """
        self.__model = get_model('mask_rcnn_fpn_resnet50_v1b_coco', pretrained=True)

    def predict(self, image):
        #Resize: resize shorter edge to 500 if needed
        h, w, d = image.shape
        short = min(w, h)
        tensor, img = transform_test(image, short=min(short, 500))

        return [xx[0] for xx in self.__model(tensor)] # ids, scores, bboxes, masks

    def visualize(self, img, ids, scores, bboxes, masks, outfile):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot_bbox(img, bboxes, scores, ids, class_names=self.__model.classes, ax=ax)
        plt.savefig(outfile)
        plt.close()

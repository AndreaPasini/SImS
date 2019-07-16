import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from gluoncv.data.transforms.presets.rcnn import transform_test
from gluoncv.model_zoo import get_model
from gluoncv.utils.viz import plot_bbox, expand_mask, plot_mask
import numpy as np
from itertools import groupby

def compress_mask(mask):
    """ Compress mask with RLE """
    cmask = []
    for row in mask:
        compressed_row = [(sum(1 for _ in g)) for k, g in groupby(row)]
        if row[0] == 1:
            compressed_row = [0] + compressed_row
        cmask.append(compressed_row)
    return cmask

def extract_mask(cmask):
    """ Extract compressed mask with RLE, 1/0 values """
    mask = []

    for crow in cmask:
        row = []
        val = 0
        for count in crow:
            if count > 0:
                row+=([val for i in range(count)])
            if val == 0:
                val = 1
            else:
                val = 0
        mask.append(row)
    return np.array(mask)

def extract_mask_bool(cmask):
    """ Extract compressed mask with RLE, True/False values """
    mask = []

    for crow in cmask:
        row = []
        val = False
        for count in crow:
            if count > 0:
                row+=([val for i in range(count)])
            if val == False:
                val = True
            else:
                val = False
        mask.append(row)
    return np.array(mask)

class Instance_segmentation:
    """
    Instance segmentation model (MaskRCNN). Trained on COCO THINGS dataset.
    gluoncv (https://gluon-cv.mxnet.io/model_zoo/segmentation.html)
    Results on COCO val2017
    Box AP (averaged 10 values):  39.2
    Mask AP (averaged 10 values): 35.4
    """

    __model = None
    __maskrcnn_to_coco = None        #maskrcnn to coco classes conversion (by id)

    def __init__(self, maskrcnn_to_coco_path):
        """
        Initialize the instance segmentation model (MaskRCNN). Trained on COCO THING dataset.
        """
        self.__model = get_model('mask_rcnn_fpn_resnet50_v1b_coco', pretrained=True)

        # Read maskrcnn to coco classes conversion
        self.__maskrcnn_to_coco = {}
        with open(maskrcnn_to_coco_path, 'r') as f:
            for line in f.readlines():
                maskrcnn_id, coco_id = line.rstrip('\n').split(":")
                self.__maskrcnn_to_coco[int(maskrcnn_id)] = int(coco_id)

    def build_class_file(self, coco_thing_classes_path, output_conversion_path):
        """
        Generates a file with the map between maskrcnn classes and COCO classes
        """

        # read coco thing classes and their ids
        coco_thing_classes = {}
        with open(coco_thing_classes_path) as f:
            for line in f.readlines():
                id, label = line.rstrip('\n').split(":")
                coco_thing_classes[label] = id

        # map maskrcnn classes to coco thing
        maskrcnn_to_coco = {}
        for id, label in enumerate(self.__model.classes):
            maskrcnn_to_coco[id] = coco_thing_classes[label]

        #Save mappings to file
        with open(output_conversion_path, 'w') as f:
            for k,v in maskrcnn_to_coco.items():
                f.write('%s:%s\n'%(k,v))

    def predict(self, image):
        """
        Perform
        :param image: image being classified
        :return: ids (classes), scores (confidence of predictions), bboxes (1000x4), masks
        """
        #Resize: resize shorter edge to 500 if needed
        h, w, d = image.shape
        short = min(w, h)
        tensor, image1 = transform_test(image, short=min(short, 500))
        h1, w1, d1 = image1.shape
        ids, scores, bboxes, masks = [xx[0] for xx in self.__model(tensor)]
        #Remove objects outside figure
        bboxes = bboxes.asnumpy()
        scores = scores.asnumpy()
        ids = ids.asnumpy()
        outside_sel = (bboxes[:,0]>w1) | (bboxes[:,1]>h1) | (bboxes[:,2]>w1) | (bboxes[:,3]>h1)
        bboxes[outside_sel,:]= -1
        scores[outside_sel, :] = 0

        #Resize bounding boxes to original image size
        selection = bboxes[:,0]>=0 #Do not touch elements that have value -1
        bboxes[selection, 0]*= (1.0*w/w1)   #xmin
        bboxes[selection, 1]*= (1.0*h/h1)   #ymin
        bboxes[selection, 2]*= (1.0*w/w1)   #xmax
        bboxes[selection, 3]*= (1.0*h/h1)   #ymax

        #Map maskrcnn ids to COCO ids
        def f_maskrcnn_to_coco(c):
            if c>=0:
                return self.__maskrcnn_to_coco[c]
            else:
                return c

        fvectorized = np.vectorize(f_maskrcnn_to_coco)
        # Convert to coco class ids:
        idsCOCO = fvectorized(ids)

        return idsCOCO, scores, bboxes, masks

    def save_json_boxes(self, img, ids, scores, bboxes, masks, output_path, thresh=0.5):
        """
        Save bounding boxes to json file
        Format:
        """
        width, height = img.shape[1], img.shape[0]
        masks = expand_mask(masks, bboxes, (width, height), scores, thresh=thresh)
        jsonout = []
        i = 0
        for id, score, bbox in zip(ids, scores, bboxes):
            score = score[0]
            id = id[0]
            if score < thresh or id < 0:
                continue
            printable_bbox = [int(round(x)) for x in bbox]
            #compress mask with RLE
            cmask = compress_mask(masks[i])
            jsonout.append({'class': int(id), 'score': ('%.3f' % score), 'bbox': printable_bbox, 'mask' : cmask})
            i+=1
        with open(output_path, 'w') as f:
            json.dump(jsonout, f)

    def visualize(self, img, json_data, outfile):
        """
        Visualize detections
        :param img: image where masks will be drawn
        :param json_data: json generated by save_json_boxes()
        :param outfile: output png file
        """
        reverse_map = {val:key for (key, val) in self.__maskrcnn_to_coco.items()}
        ids = np.array([reverse_map[x['class']] for x in json_data])
        scores = np.array([float(x['score']) for x in json_data])
        bboxes = [x['bbox'] for x in json_data]
        masks = [extract_mask(x['mask']) for x in json_data]

        #Plot data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        out_img = plot_mask(img, masks)
        plot_bbox(out_img, bboxes, scores, ids, class_names=self.__model.classes, ax=ax)
        plt.savefig(outfile)
        plt.close()

import os

from config import COCO_ann_val_dir, COCO_img_val_dir, COCO_ann_train_dir, COCO_img_train_dir
from sims.scene_graphs.image_processing import mask_baricenter, getImageName
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import cv2
import seaborn as sns
import numpy as np
import pyximport
pyximport.install(language_level=3)
from panopticapi.utils import load_png_annotation
from sims.scene_graphs.feature_extraction import extract_bbox_from_mask

class RelationshipVisualizer:
    """ Class for printing object relationships """

    __MIN_AGGREGATION_DIST = 40     # Minimum distance (pixel) for aggregating relationship end-points
    __OBJECT_ALPHA_COLOR = 0.8      # Opacity (0=transparent) of object patches drawn on the image

    def __init__(self, image_id, ax, is_train_image=True, preprocess_image=True):
        """
        Constructor
        :param image_id: id of the input COCO image
        :param ax: axes object where the output will be printed
        :param is_train_image: True if the graph is from COCO Training, otherwise False for Validation set
        :param preprocess_image: True if you want to change image colors for better visualization
        """
        if is_train_image:
            ann_folder = COCO_ann_train_dir
            img_folder = COCO_img_train_dir
        else:
            ann_folder = COCO_ann_val_dir
            img_folder = COCO_img_val_dir
        image_ann_name = getImageName(image_id, ann_folder, 'png')
        self.__img_annotation = load_png_annotation(image_ann_name)
        image_file_name = getImageName(image_id, img_folder, 'jpg')
        rgb_image = cv2.imread(image_file_name)[:, :, ::-1]
        if preprocess_image:
            self.__output_image = self.__preprocess_rgb_image(rgb_image)
        else:
            self.__output_image = rgb_image
        self.__ax = ax
        self.__object_centers = {}

    def __preprocess_rgb_image(self, rgb_image):
        # Preprocess bgr image for better visualization
        img_print = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        v = img_print[:, :, 2]
        increase = 50
        img_print[:, :, 2] = 0.3 * 255 + 0.7 * np.where(v <= 255 - increase, v + increase, 255)
        img_print[:, :, 1] = 0.3 * img_print[:, :, 1]
        return cv2.cvtColor(img_print, cv2.COLOR_HSV2RGB)

    def __random_color(self):
        pal = sns.color_palette("Set3")
        index = np.random.randint(0, len(pal))
        return 255*np.array(pal[index])*0.8 + 0.2 * np.random.choice(range(256), size=3)

    def __aggregate_endpoints(self, obj_id, computed_x, computed_y):
        if obj_id in self.__object_centers:     # if already have an endpoint on this object
            # Find a close endpoint for obj_id
            for c in self.__object_centers[obj_id]:
                if np.sqrt((c[0] - computed_x) ** 2 + (c[1] - computed_y) ** 2) < self.__MIN_AGGREGATION_DIST:
                    return c[0], c[1]
        # Close endpoint not found. Add a new one
        self.__object_centers[obj_id].append((computed_x, computed_y))
        return computed_x, computed_y

    def draw_relationship(self, pos, sub, ref, mode=1, draw_masks=True):
        """
        Draw to the image a single relationship.
        :param pos: position relationship (e.g. "above")
        :param sub: subject id
        :param ref: reference id
        :param mode: 1 (optimized segment positioning), 2 (base relationhip-line positioning)
        :param draw_masks: True if you want to draw object masks
        """
        colors = {'side-up': 'black', 'side-down': 'black', 'side': 'black'}

        # Get subject and reference masks
        sub_mask = self.__img_annotation == sub
        ref_mask = self.__img_annotation == ref

        # Draw object masks (transparent patch) if not present
        if sub not in self.__object_centers:
            rand_color = self.__OBJECT_ALPHA_COLOR*self.__random_color()
            if draw_masks:
                self.__output_image[sub_mask] = np.floor(self.__output_image[sub_mask] * (1-self.__OBJECT_ALPHA_COLOR) + rand_color)
            self.__object_centers[sub] = []
        if ref not in self.__object_centers:
            rand_color = self.__OBJECT_ALPHA_COLOR*self.__random_color()
            if draw_masks:
                self.__output_image[ref_mask] = np.floor(self.__output_image[ref_mask] * (1-self.__OBJECT_ALPHA_COLOR) + rand_color)
            self.__object_centers[ref] = []

        # Find subject and reference positions
        _, sub_left, _, sub_right = extract_bbox_from_mask(sub_mask)
        _, ref_left, _, ref_right = extract_bbox_from_mask(ref_mask)
        # Get sub and ref baricenters (first attempt for relationship endpoints)
        x_sub, y_sub = mask_baricenter(sub_mask)
        x_ref, y_ref = mask_baricenter(ref_mask)

        # Update endopoints for better visualization (try to use vertical lines)
        if mode == 1:
            # Middle position at the horizontal intersection of sub and ref
            x_middle = int((min(sub_right, ref_right) + max(sub_left, ref_left)) / 2)
            if pos in ['above', 'on', 'below', 'hanging']:
                if self.__img_annotation[y_sub, x_middle] == sub:  # if this is a point of subject (might not be)
                    x_sub = x_middle   # Update x value with middle position
                if self.__img_annotation[y_ref, x_middle] == ref:
                    x_ref = x_middle

            # Update endpoint positions (aggregate endpoints with previous relationships to avoid cluttering)
            x_sub, y_sub = self.__aggregate_endpoints(sub, x_sub, y_sub)
            x_ref, y_ref = self.__aggregate_endpoints(ref, x_ref, y_ref)

        # Draw line relationships
        if pos in colors:
            self.__ax.add_line(mlines.Line2D([x_sub, x_ref], [y_sub, y_ref], color=colors[pos], label=pos))
        else:
            self.__ax.add_line(mlines.Line2D([x_sub, x_ref], [y_sub, y_ref]))
        # Draw endpoints
        self.__ax.add_patch(mpatches.Circle((x_sub, y_sub), color='white'))
        self.__ax.add_patch(mpatches.Circle((x_ref, y_ref), color='white'))
        self.__ax.add_patch(mpatches.Circle((x_sub, y_sub), radius=10, color='black', fill=False))
        self.__ax.add_patch(mpatches.Circle((x_ref, y_ref), radius=10, color='black', fill=False))

    def get_output_image(self):
        """
        :return: RGB output image, with object patches
        """
        return self.__output_image

    def draw_positive_relationships(self, graph, kb, thr, draw_masks):
        """
        Draw on axes: the image with object patches and the set of relationships
        :param graph: graph of this image, with object positions (read from json)
        :param kb: knowledge base (read from json)
        :param thr: confidence threshold for a relationship to be printed
        :param draw_masks: True if you want to draw object masks
        """
        nodes = {}
        for node in graph['nodes']:
            nodes[node['id']] = node['label']
        # Print links
        for link in graph['links']:
            sub = nodes[link['s']]
            ref = nodes[link['r']]
            pos = link['pos']
            pair = f"{sub},{ref}"
            if pair in kb:
                hist = kb[f"{sub},{ref}"]
                pal = sns.color_palette("Set3")
                if hist[pos] > thr: # Threshold for printing a link
                    self.draw_relationship(pos, link['s'], link['r'], mode=1, draw_masks=draw_masks)
        self.__ax.imshow(self.get_output_image())

def filter_graphs_with_local_data(graphs, is_train=True):
    """
    Filter out graphs that do not have the image file in the local folder.
    (=keep only graphs that can be shown with RelationshipVisualizer)
    :param graphs: list of graphs
    :param is_train: True if the graph is from COCO Training, otherwise False for Validation set
    :return: filtered graphs
    """
    filtered_graphs = []
    if is_train:
        img_folder = COCO_img_train_dir
    else:
        img_folder = COCO_img_val_dir
    for g in graphs:
        img_file = getImageName(g['graph']['name'], img_folder, 'jpg')
        if os.path.exists(img_file):
            filtered_graphs.append(g)
    return filtered_graphs
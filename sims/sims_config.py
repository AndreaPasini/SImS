"""
This file contains the class for configuring SImS to a given dataset
and with specific hyperparameter settings.
"""
import json
from scipy.stats import entropy
import numpy as np

from config import COCO_SGS_dir, COCO_PRS_json_path, COCO_train_graphs_json_path, VG_SGS_dir, \
    VG_PRS_json_path, VG_train_graphs_json_path, position_labels_csv_path, VG_objects_json_path, \
    VG_predicates_json_path, COCO_train_graphs_subset_json_path, COCO_ann_train_dir, COCO_train_json_path, \
    VG_train_json_path, COCO_PRS_dir, VG_PRS_dir
from panopticapi.utils import load_panoptic_categ_list

class SImS_config:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == 'COCO':
            self.ann_dir = COCO_ann_train_dir
            self.ann_json_path = COCO_train_json_path
            self.scene_graphs_json_path = COCO_train_graphs_json_path
            self.PRS_dir = COCO_PRS_dir
            self.PRS_json_path = COCO_PRS_json_path
            self.SGS_dir = COCO_SGS_dir
        elif dataset == 'COCO_subset':
            self.ann_dir = COCO_ann_train_dir
            self.ann_json_path = COCO_train_json_path
            self.scene_graphs_json_path = COCO_train_graphs_subset_json_path
            self.PRS_dir = COCO_PRS_dir
            self.PRS_json_path = COCO_PRS_json_path
            self.SGS_dir = COCO_SGS_dir + "/subset/"
        elif dataset == 'VG':
            self.ann_dir = None
            self.ann_json_path = VG_train_json_path
            self.SGS_dir = VG_SGS_dir
            self.PRS_dir = VG_PRS_dir
            self.PRS_json_path = VG_PRS_json_path
            self.scene_graphs_json_path = VG_train_graphs_json_path

        # Default SGS configuration
        self.SGS_params = {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.01}

    def setSGS_params(self, SGS_params):
        """
        Change default SGS configuration
        :param SGS_params: SGS configuration (dict: alg, edge_pruning, node_pruning, min_sup)
        """
        self.SGS_params = SGS_params

    def load_categories(self):
        """
        :return: object categories and relationship categories for the configured dataset
        """
        if self.dataset == 'COCO' or self.dataset == 'COCO_subset':
            obj_categories = load_panoptic_categ_list()
            with open(position_labels_csv_path) as f:
                rel_categories = tuple(s.strip() for s in f.readlines())
        elif self.config['dataset'] == 'VG':
            with open(VG_objects_json_path) as f:
                obj_categories = {i: l for i, l in enumerate(json.load(f))}
            with open(VG_predicates_json_path) as f:
                rel_categories = json.load(f)
        return obj_categories, rel_categories

    def get_PRS_filters(self, sup_list, entr_list):
        """
        Return min_sup and max_entr for filtering the PRS
        :param sup_list: list of PRS supports
        :param entr_list: list of PRS entropies
        :return: min_sup, max_entr
        """
        max_entropy = entropy([1 / 3, 1 / 3, 1 / 3])
        med = np.median(np.log10(sup_list))
        # Set minsup to median, if COCO dataset
        if self.dataset == 'COCO' or self.dataset == 'COCO_subset':
            min_sup = int(round(10 ** med))
        else:
            min_sup = 20
        return min_sup, max_entropy

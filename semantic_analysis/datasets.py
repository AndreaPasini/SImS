import json

from scipy.stats import entropy
import numpy as np

from config import COCO_graph_mining_dir, kb_pairwise_json_path, train_graphs_json_path, VG_graph_mining_dir, \
    VG_kb_pairwise_json_path, VG_train_graphs_json_path, position_labels_csv_path, VG_objects_json_path, \
    VG_predicates_json_path, train_graphs_subset_json_path
from panopticapi.utils import load_panoptic_categ_list


class MiningConf:
    config = None
    graph_mining_dir = None
    kb_json_path = None
    graphs_json_path = None

    def __init__(self, config):
        self.config = config
        if config['dataset'] == 'COCO':
            if config['dataset_info'] == "COCO_subset":
                self.graph_mining_dir = COCO_graph_mining_dir + "/subset/"
                self.graphs_json_path = train_graphs_subset_json_path
            else:
                self.graph_mining_dir = COCO_graph_mining_dir
                self.graphs_json_path = train_graphs_json_path
            self.kb_json_path = kb_pairwise_json_path


        elif config['dataset'] == 'VG':
            self.graph_mining_dir = VG_graph_mining_dir
            self.kb_json_path = VG_kb_pairwise_json_path
            self.graphs_json_path = VG_train_graphs_json_path



    def load_categories(self):
        if self.config['dataset'] == 'COCO':
            obj_categories = load_panoptic_categ_list()
            with open(position_labels_csv_path) as f:
                rel_categories = tuple(s.strip() for s in f.readlines())
        elif self.config['dataset'] == 'VG':
            with open(VG_objects_json_path) as f:
                obj_categories = {i: l for i, l in enumerate(json.load(f))}
            with open(VG_predicates_json_path) as f:
                rel_categories = json.load(f)
        return obj_categories, rel_categories

    def get_filters(self, sup, entr):
        max_entropy = entropy([1 / 3, 1 / 3, 1 / 3])
        med = np.median(np.log10(sup))
        # Set minsup to median, if COCO dataset
        if self.config['dataset'] == 'COCO':
            min_sup = int(round(10 ** med))
        else:
            min_sup = 20
        return min_sup, max_entropy
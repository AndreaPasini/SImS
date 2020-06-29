
import json
import networkx as nx
from config import VG_predicates_json_path, VG_objects_json_path
from sims.graph_utils import nx_to_json


def vgann2scene_graph(ann_id, ann, obj_list, rel_list):
    """
    Build scene graph from visual genome annotation.
    :param ann_id: img id
    :param ann: annotation, json
    :param obj_list: list of object classes
    :param rel_list: list of object relationships
    :return:
    """
    bbox_to_id = {}
    new_id = 0
    g = nx.DiGraph()
    g.name = ann_id
    for el in ann:
        sbbox = tuple(el['subject']['bbox'])
        rbbox = tuple(el['object']['bbox'])
        if sbbox not in bbox_to_id:
            bbox_to_id[sbbox] = new_id
            g.add_node(new_id, label=obj_list[el['subject']['category']])
            new_id += 1
        if rbbox not in bbox_to_id:
            bbox_to_id[rbbox] = new_id
            g.add_node(new_id, label=obj_list[el['object']['category']])
            new_id += 1
        s = bbox_to_id[sbbox]
        r = bbox_to_id[rbbox]
        g.add_edge(s, r, pos=rel_list[el['predicate']])
    return g

def create_scene_graphs_vg(VG_ann_path, out_graphs_json_path):
    """
    Analyze Visual Genome (VG) annotations
    Generate scene graphs
    :param VG_ann_path: VG annotation file (relationships)
    :param out_graphs_json_path: output json file with graphs
    """
    with open(VG_ann_path) as f:
        annotations = json.load(f)
    with open(VG_predicates_json_path) as f:
        rel_list = json.load(f)
    with open(VG_objects_json_path) as f:
        obj_list = json.load(f)

    graphs = []
    for img_id, ann in annotations.items():
        img_id = img_id.split(".")[0]
        g = vgann2scene_graph(img_id, ann, obj_list, rel_list)
        graphs.append(nx_to_json(g))
    with open(out_graphs_json_path, 'w') as f:
        json.dump(graphs, f)
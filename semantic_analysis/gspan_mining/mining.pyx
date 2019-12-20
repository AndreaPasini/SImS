from config import position_labels_csv_path
from panopticapi.utils import load_panoptic_categ_list
import json
import pyximport
pyximport.install(language_level=3)
from semantic_analysis.gspan_mining.graph import nx_to_json, json_graph_to_gspan


def prepare_gspan_graph_data(train_graphs_data_path, train_graphs_json_path):
    """
    Prepare and save graphs in the correct encoding for applying GSpan.
    """
    with open(train_graphs_json_path, 'r') as f:
        graphs = json.load(f)

    # Encode panoptic class ids
    # gspan requests as input label values that start from 2.
    coco_categories = load_panoptic_categ_list()
    conv_coco_category = {c:i+2 for i,c in enumerate(coco_categories.values())}

    # Encode position labels
    position_labels = tuple(s.strip() for s in open(position_labels_csv_path).readlines())
    conv_pos_category = {c: i + 2 for i, c in enumerate(position_labels)}

    # Prepare nodes with the correct format for graph gspan_mining
    with open(train_graphs_data_path, 'w') as f:
        for g_index, g in enumerate(graphs):
            f.write(f"t # {g_index}\n")
            f.write(json_graph_to_gspan(g, conv_coco_category, conv_pos_category))
        f.write("t # -1")

def gspan_to_final_graphs(gspan_graphs):
    """
    Convert the output of GSpan to final knowledge base graphs.

    ----
    :param gspan_graphs: output of Gspan
    :return: list of dictionaries. 'g': FrequentGraph, 'sup': support
    """
    # Encode panoptic class ids
    # gspan requests as input label values that start from 2.
    coco_categories = load_panoptic_categ_list()
    conv_coco_category = {i + 2: c for i, c in enumerate(coco_categories.values())}

    # Encode position labels
    position_labels = tuple(s.strip() for s in open(position_labels_csv_path).readlines())
    conv_pos_category = {i + 2: c for i, c in enumerate(position_labels)}

    final_graphs = []
    for g, sup in gspan_graphs:
        res_graph = nx_to_json(g.to_nx_COCO_graph(conv_coco_category, conv_pos_category))
        final_graphs.append({'g' : res_graph, 'sup' : sup})

    return final_graphs


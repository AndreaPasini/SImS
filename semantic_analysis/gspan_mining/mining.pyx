import networkx as nx

from config import position_labels_csv_path
from panopticapi.utils import load_panoptic_categ_list
import json
import pyximport
pyximport.install(language_level=3)
from semantic_analysis.gspan_mining.graph import nx_to_json, json_graph_to_gspan, Graph


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

def _read_graphs(file_name, ):
    # Encode panoptic class ids
    # gspan requests as input label values that start from 2.
    coco_categories = load_panoptic_categ_list()
    conv_coco_category = {i + 2: c for i, c in enumerate(coco_categories.values())}

    # Encode position labels
    position_labels = tuple(s.strip() for s in open(position_labels_csv_path).readlines())
    conv_pos_category = {i + 2: c for i, c in enumerate(position_labels)}

    graphs = []
    with open(file_name, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        tgraph = None
        for i, line in enumerate(lines):
            cols = line.split(' ')
            if cols[0] == 't':
                if tgraph is not None:
                    graphs.append((tgraph, graph_sup))
                    tgraph = None
                if cols[-1] == '-1':
                    break

                tgraph = nx.DiGraph()
                node_labels = {}
                graph_sup = int(cols[4])
            elif cols[0] == 'v':
                #tgraph.add_vertex(cols[1], cols[2])
                label = conv_coco_category[int(cols[2])]
                node_labels[int(cols[1])] = label
                tgraph.add_node(int(cols[1]), label=label)


            elif cols[0] == 'e':
                #tgraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], cols[3])
                # Edges must be sorted alphabetically to reconstruct directed graph
                if (node_labels[int(cols[1])] <= node_labels[int(cols[2])]):
                    tgraph.add_edge(int(cols[1]), int(cols[2]), label=conv_pos_category[int(cols[3])])
                else:
                    tgraph.add_edge(int(cols[2]), int(cols[1]), label=conv_pos_category[int(cols[3])])


        # adapt to input files that do not end with 't # -1'
        if tgraph is not None:
            graphs.append((tgraph, graph_sup))

        # Convert networkx graphs to json
        final_graphs = []
        for g, sup in graphs:
            res_graph = nx_to_json(g)
            final_graphs.append({'g': res_graph, 'sup': sup})

        return final_graphs



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


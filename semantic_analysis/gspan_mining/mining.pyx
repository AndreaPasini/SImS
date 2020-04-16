import json
import networkx as nx
import os
import pyximport
pyximport.install(language_level=3)
from semantic_analysis.gspan_mining.graph_utils import nx_to_json, json_graph_to_gspan
from config import position_labels_csv_path
from panopticapi.utils import load_panoptic_categ_list

def prepare_gspan_graph_data(freq_graphs_output_path, graphs):
    """
    Prepare and save graphs in the correct encoding for applying GSpan.
    """
    # Encode panoptic class ids
    # gspan requests as input label values that start from 2.
    coco_categories = load_panoptic_categ_list()
    conv_coco_category = {c:i+2 for i,c in enumerate(coco_categories.values())}

    # Encode position labels
    position_labels = tuple(s.strip() for s in open(position_labels_csv_path).readlines())
    conv_pos_category = {c: i + 2 for i, c in enumerate(position_labels)}

    # Prepare nodes with the correct format for graph gspan_mining
    with open(freq_graphs_output_path, 'w') as f:
        for g_index, g in enumerate(graphs):
            f.write(f"t # {g_index}\n")
            f.write(json_graph_to_gspan(g, conv_coco_category, conv_pos_category))
        f.write("t # -1")

def run_gspan_mining(graphs_data_path, minsup, output_path):
    """
    Run gspan mining to extract frequent graphs
    GSpan, c implementation (https://www.researchgate.net/publication/296573357_gSpan_Implementation,
    https://sites.cs.ucsb.edu/~xyan/software/gSpan.htm
    Paper: gSpan: graph-based substructure pattern mining (ICDM 2003),)
    :param graphs_data_path: input file, generated with prepare_gspan_graph_data()
    :param minsup: relative minsup for frequent graphs
    :param output_path: output json file with frequent graphs
    """
    print("Gspan - Mining frequent graphs...")
    os.system(f"./semantic_analysis/gspan_mining/gSpan-64 -f {graphs_data_path} -s {minsup} -o")
    print("Mining complete. Converting graphs...")
    print(graphs_data_path)
    freq_graphs = __read_gspan_output(f'{graphs_data_path}.fp')
    with open(output_path, 'w') as f:
        f.write(json.dumps(freq_graphs))
    print("Done.")
    os.remove(f'{graphs_data_path}.fp')

def __read_gspan_output(file_name):
    """
    Read gspan output graphs.
    :param file_name: gspan output file
    :return: list of dictionaries. 'g': FrequentGraph, 'sup': support
    """

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


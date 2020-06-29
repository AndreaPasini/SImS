import os
import json
import networkx as nx
from config import position_labels_csv_path
from panopticapi.utils import load_panoptic_categ_list
import pyximport
pyximport.install(language_level=3)
from sims.subdue_mining.graph_utils import json_graph_to_subdue
from sims.graph_utils import nx_to_json

def prepare_subdue_graph_data(freq_graphs_output_path, graphs, obj_categories, rel_categories):
    """
    Prepare and save graphs in the correct encoding for applying GSpan.
    :param freq_graphs_output_path: output file path
    :param graphs: list of json graphs to be encoded
    :param obj_categories: dict -> obj_categories[objectId] = textual label (COCO classes or VG classes)
    :param rel_categories: list -> rel_categories[catId] = textual category for relationships
    """
    # Encode object class ids
    # gspan requests as input label values that start from 2.
    conv_obj_category = {c:i+2 for i,c in enumerate(obj_categories.values())}
    # Encode relationship labels
    conv_rel_category = {c: i + 2 for i, c in enumerate(rel_categories)}

    # Prepare nodes with the correct format for graph gspan_mining
    with open(freq_graphs_output_path, 'w') as f:
        for g_index, g in enumerate(graphs):
            if len(g['links'])>0:
                f.write(f"\nXP\n")
                f.write(json_graph_to_subdue(g, conv_obj_category, conv_rel_category))

def run_subdue_mining(graphs_data_path, nsubs, output_path, obj_categories, rel_categories):
    """
    Run subdue mining to extract frequent graphs
    C Implementation: http://ailab.wsu.edu/subdue/
    Paper: http://ailab.eecs.wsu.edu/subdue/papers/KetkarOSDM05.pdf,
    "Subdue: Compression-Based Frequent Pattern Discovery in Graph Data"
    :param graphs_data_path: input file, generated with prepare_gspan_graph_data()
    :param nsubs: number of substructures to be found
    :param output_path: output json file with frequent graphs
    :param obj_categories: dict -> obj_categories[objectId] = textual label (COCO classes or VG classes)
    :param rel_categories: list -> rel_categories[catId] = textual category for relationships
    """
    print("Subdue - Mining best substuctures...")
    os.system(f"./sims/subdue_mining/subdue -minsize 2 -nsubs {nsubs} {graphs_data_path} > {output_path+'.txt'}")
    print("Mining complete. Converting graphs...")
    print(graphs_data_path)
    freq_graphs = __read_subdue_output(output_path+'.txt', obj_categories, rel_categories)
    with open(output_path, 'w') as f:
        f.write(json.dumps(freq_graphs))
    print("Done.")
    #os.remove(output_path+'.txt')

def __read_subdue_output(file_name, obj_categories, rel_categories):
    """
    Read subdue output graphs.
    :param file_name: gspan output file
    :param obj_categories: dict -> obj_categories[objectId] = textual label (COCO classes or VG classes)
    :param rel_categories: list -> rel_categories[catId] = textual category for relationships
    :return: list of dictionaries. 'g': FrequentGraph, 'sup': support
    """

    # Encode object class ids
    # gspan requests as input label values that start from 2.
    conv_obj_category = {i + 2: c for i, c in enumerate(obj_categories.values())}
    # Encode relationship labels
    conv_rel_category = {i + 2: c for i, c in enumerate(rel_categories)}

    graphs = []
    with open(file_name, 'r', encoding="utf-8") as f:
        while not f.readline().startswith("Best "): pass
        f.readline()
        lines = [line.strip() for line in f.readlines()]

        cur_graph = None
        cur_sup = None
        header_count = 0
        for i, line in enumerate(lines):

            if header_count == 0:
                # New graph
                cols = line.split(' ')
                if cols[0].startswith("("):
                    header_count+=1
                    cur_graph = nx.DiGraph()
                    cur_sup = int(cols[8][:-1])  # Get graph support
                    node_labels = {}
                else:
                    if cols[0]=='':
                        break # End of graps

            elif header_count == 1:
                header_count+=1  # Skip header
            else:
                cols = line.split(' ')
                if cols[0] == 'v':
                    label = conv_obj_category[int(cols[2])]
                    node_labels[int(cols[1])] = label
                    cur_graph.add_node(int(cols[1]), label=label)
                elif cols[0] == 'd':
                    # Edges must be sorted alphabetically to reconstruct directed graph
                    if (node_labels[int(cols[1])] <= node_labels[int(cols[2])]):
                        cur_graph.add_edge(int(cols[1]), int(cols[2]), pos=conv_rel_category[int(cols[3])])
                    else:
                        cur_graph.add_edge(int(cols[2]), int(cols[1]), pos=conv_rel_category[int(cols[3])])
                else:
                    # End of graph
                    header_count=0
                    graphs.append((cur_graph, cur_sup))

        # Convert networkx graphs to json
        final_graphs = []
        for g, sup in graphs:
            res_graph = nx_to_json(g)
            final_graphs.append({'g': res_graph, 'sup': sup})

        return final_graphs

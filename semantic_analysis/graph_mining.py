import json
import os
from shutil import copyfile
import numpy as np
from scipy.stats import entropy
from config import graph_mining_dir, kb_pairwise_json_path, train_graphs_json_path
from semantic_analysis.knowledge_base import get_sup_ent_lists, filter_kb_histograms, prune_graph_edges, \
    prune_equivalent_nodes

from semantic_analysis.gspan_mining.mining import prepare_gspan_graph_data, run_gspan_mining
from semantic_analysis.subdue_mining.mining import prepare_subdue_graph_data, run_subdue_mining

from semantic_analysis.graph_utils import json_to_graphviz


def get_exp_name(experiment):
    """
    Get experiment name, given configuration dictionary
    Experiment name can be used for generating its associated files
    :param experiment: experiment configuration (dictionary)
    """
    kb_filter="_kbfilter" if experiment['edge_pruning'] else ""
    node_pruning = "_prune" if experiment['node_pruning'] else ""
    if experiment['alg']=='gspan':
        exp_name = f"train_freqGraph{kb_filter}{node_pruning}_{experiment['alg']}_{str(experiment['minsup'])[2:]}"
    else:
        exp_name = f"train_freqGraph{kb_filter}{node_pruning}_{experiment['alg']}_{experiment['nsubs']}"
    return exp_name


def prepare_graphs_with_KB(experiment):
    """
    Given experiment configuration, return training COCO graphs, pruned and filtered according to KB
    :param experiment: experiment configuration (dictionary)
    """
    kb_filter="_kbfilter" if experiment['edge_pruning'] else ""
    node_pruning = "_prune" if experiment['node_pruning'] else ""
    output_file = os.path.join(graph_mining_dir, f"train_graphs{kb_filter}{node_pruning}.json")
    # Check whether graph data has already been created
    if not os.path.exists(output_file):
        # Read KB
        with open(kb_pairwise_json_path, 'r') as f:
            kb = json.load(f)
        # Get support and entropy
        sup, entr = get_sup_ent_lists(kb)
        max_entropy = entropy([1 / 3, 1 / 3, 1 / 3])
        med = np.median(np.log10(sup))
        min_sup = int(round(10 ** med))
        # Filter KB
        if experiment['edge_pruning']:
            kb_filtered = filter_kb_histograms(kb, min_sup, max_entropy)
        else:
            kb_filtered = filter_kb_histograms(kb, min_sup, 100)  # No filter for entropy
        # Read COCO Train graphs
        with open(train_graphs_json_path, 'r') as f:
            train_graphs = json.load(f)
        # Edge pruning: filter graphs with KB
        print("Filtering graphs with KB...")
        train_graphs_filtered = prune_graph_edges(kb_filtered, train_graphs)
        # Node pruning (prune equivalent nodes to reduce redundancies and to reduce mining time)
        if experiment['node_pruning']:
            print("Pruning nodes...")
            train_graphs_filtered = prune_equivalent_nodes(train_graphs_filtered)
        with open(output_file, "w") as f:
            json.dump(train_graphs_filtered, f)
        return train_graphs_filtered
    else:
        with open(output_file, "r") as f:
            return json.load(f)


def run_graph_mining(experiment):
    """
    Run graph mining experiment
    :param experiment: experiment configuration (dictionary)
    """
    kb_filter="_kbfilter" if experiment['edge_pruning'] else ""
    node_pruning = "_prune" if experiment['node_pruning'] else ""

    sel_train_graphs_data_path = os.path.join(graph_mining_dir, f"train_graphs{kb_filter}{node_pruning}_{experiment['alg']}.data")
    exp_name = get_exp_name(experiment)
    sel_freq_graphs_path = os.path.join(graph_mining_dir, exp_name+'.json')

    if not os.path.exists(graph_mining_dir):
        os.makedirs(graph_mining_dir)

    # Check whether graph data has already been converted for
    if not os.path.exists(sel_train_graphs_data_path):
        print(f"Preparing graphs for {experiment['alg']}...")
        train_graphs_filtered = prepare_graphs_with_KB(experiment)
        if experiment['alg']=='gspan':
            # Convert json graphs to the correct format for gspan mining.
            prepare_gspan_graph_data(sel_train_graphs_data_path, train_graphs_filtered)
        elif experiment['alg']=='subdue':
            prepare_subdue_graph_data(sel_train_graphs_data_path, train_graphs_filtered)

    # Mining of frequent graphs
    if experiment['alg'] == 'gspan':
        # Necessary because gspan program outputs with the same name of the input file
        tmp_input = os.path.join(graph_mining_dir, exp_name+".data")
        copyfile(sel_train_graphs_data_path, tmp_input)
        run_gspan_mining(tmp_input, experiment['minsup'], sel_freq_graphs_path)
        os.remove(tmp_input)
    elif experiment['alg'] == 'subdue':
        run_subdue_mining(sel_train_graphs_data_path, experiment['nsubs'], sel_freq_graphs_path)

def read_freqgraphs(experiment):
    """
    Read Json frequent graphs generated with run_graph_mining()
    :param experiment: experiment configuration for mining graphs
    :return: loaded json frequent graphs
    """
    exp_name = get_exp_name(experiment)
    # Read frequent graphs
    freq_graphs_path = os.path.join(graph_mining_dir, exp_name + '.json')
    with open(freq_graphs_path, 'r') as f:
        freq_graphs = json.load(f)
    return freq_graphs

def analyze_graphs(experiment):
    """
    Compute statistics (avg. nodes, distinct classes, ...) on the extracted frequent subgraphs.
    :param experiment: experiment configuration for mining graphs
    :return: dictionary with statistics
    """
    # Read frequent graphs
    freq_graphs = read_freqgraphs(experiment)
    dist_classes = {}
    dist_sets = {}
    tot_nodes = 0       # Number of nodes
    tot_dist_nodes = 0  # Number of distinct classes in each graph
    max_dist_nodes = 0  # Max n. distinct classes
    nodes_nodes_dist = 0    # Ratio distinct nodes / nodes
    std_nodes = []
    for g in freq_graphs:
        nodes = [n['label'] for n in g['g']['nodes']]   # All node classes
        tot_nodes += len(nodes)                         # Number of nodes
        std_nodes.append(len(nodes))
        nodes_set = set(nodes)
        tot_dist_nodes += len(nodes_set)                # Number of distinct classes
        max_dist_nodes = max(max_dist_nodes, len(nodes_set)) # Max n. distinct classes
        nodes_nodes_dist += len(nodes_set)/len(nodes)
        # Add
        for n in nodes_set:                             # Track distinct classes
            if n in dist_classes:
                dist_classes[n] += 1
            else:
                dist_classes[n] = 1
        nodes_tuple = tuple(sorted(nodes_set))          # Track distinct class sets
        if nodes_tuple in dist_sets:
            dist_sets[nodes_tuple] += 1
        else:
            dist_sets[nodes_tuple] = 1

    res_dict = {"Minsup":experiment['minsup'],
                "Edge pruning": 'Y' if experiment['edge_pruning'] else 'N',
                "Node pruning": 'Y' if experiment['node_pruning'] else 'N',
                "N. graphs": len(freq_graphs), "N. distinct classes": len(dist_classes),
                "N. distinct class sets": len(dist_sets), "Avg. nodes": round(tot_nodes / len(freq_graphs), 2),
                "Avg. distinct classes": round(tot_dist_nodes / len(freq_graphs),2), "Max. distinct classes": max_dist_nodes,
                "Distinct class ratio":round(nodes_nodes_dist/len(freq_graphs),2),
                "Std. nodes": round(np.std(std_nodes),2)}
    return res_dict


def print_graphs(experiment, subsample=True, pdfformat=True, alternate_colors=True):
    """
    Print graphs to files
    :param experiment: experiment configuration for mining graphs
    :param subsample: subsample graphs if >500
    :param pdfformat: True to print pdf, False to print png
    :param alternate_colors: True if you want to alternate different colors for nodes
    """
    exp_name = get_exp_name(experiment)
    # Read
    sel_freq_graphs_path = os.path.join(graph_mining_dir, exp_name + '.json')
    with open(sel_freq_graphs_path, 'r') as f:
        graphs = json.load(f)

    out_path = os.path.join(graph_mining_dir, f"charts/{exp_name}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Subsampling of graphs
    if type(subsample) is list:
        iter_graphs = [(i,g) for i,g in enumerate(graphs) if i in subsample]
    elif len(graphs) > 500 and subsample==True:
        steps = np.round(np.linspace(0,len(graphs)-1,500)).astype(np.int)
        iter_graphs = [(i,g) for i,g in enumerate(graphs) if i in steps]
    else:
        iter_graphs = enumerate(graphs)

    fillcolors = ["#d4eaff", "#baf0a8"]
    format = "pdf" if pdfformat else "png"
    for i, g_dict in iter_graphs:
        sup = g_dict['sup']
        g = json_to_graphviz(g_dict['g'], fillcolor=fillcolors[0])
        if alternate_colors:
            fillcolors.reverse()
        with open(f"{out_path}/g{i}_s_{sup}.{format}", "wb") as f:
           f.write(g.pipe(format=format))

import json
import os
from shutil import copyfile
import numpy as np
from semantic_analysis.knowledge_base import get_sup_ent_lists, filter_kb_histograms, prune_graph_edges, \
    prune_equivalent_nodes

from semantic_analysis.gspan_mining.mining import prepare_gspan_graph_data, run_gspan_mining
from semantic_analysis.subdue_mining.mining import prepare_subdue_graph_data, run_subdue_mining

from semantic_analysis.graph_utils import json_to_graphviz


def get_exp_name(miningConf):
    """
    Get experiment name, given configuration dictionary
    Experiment name can be used for generating its associated files
    :param miningConf: experimental configuration class
    """
    config = miningConf.config
    kb_filter="_kbfilter" if config['edge_pruning'] else ""     # Edge pruning
    node_pruning = "_prune" if config['node_pruning'] else ""   # Node pruning
    if config['alg']=='gspan':
        exp_name = f"train_freqGraph{kb_filter}{node_pruning}_{config['alg']}_{str(config['minsup'])[2:]}"
    else:
        exp_name = f"train_freqGraph{kb_filter}{node_pruning}_{config['alg']}_{config['nsubs']}"
    return exp_name


def prepare_graphs_with_KB(miningConf):
    """
    Given experiment configuration, return graphs, pruned and filtered according to KB
    Experiment may refer to either COCO or VG dataset.
    :param miningConf: experimental configuration class
    """
    config = miningConf.config
    kb_filter="_kbfilter" if config['edge_pruning'] else ""
    node_pruning = "_prune" if config['node_pruning'] else ""
    output_file = os.path.join(miningConf.graph_mining_dir, f"train_graphs{kb_filter}{node_pruning}.json")
    # Check whether graph data has already been created
    if not os.path.exists(output_file):
        # Read KB
        with open(miningConf.kb_json_path, 'r') as f:
            kb = json.load(f)
        # Get support and entropy
        sup, entr = get_sup_ent_lists(kb)
        # Get minsup and maxentr
        min_sup, max_entropy = miningConf.get_filters(sup, entr)

        # Filter KB
        if config['edge_pruning']:
            kb_filtered = filter_kb_histograms(kb, min_sup, max_entropy)
        else:
            kb_filtered = filter_kb_histograms(kb, min_sup, 100)  # No filter for entropy
        # Read COCO Train graphs
        with open(miningConf.graphs_json_path, 'r') as f:
            train_graphs = json.load(f)
        # Edge pruning: filter graphs with KB
        print("Filtering graphs with KB...")
        train_graphs_filtered = prune_graph_edges(kb_filtered, train_graphs)
        # Node pruning (prune equivalent nodes to reduce redundancies and to reduce mining time)
        if config['node_pruning']:
            print("Pruning nodes...")
            train_graphs_filtered = prune_equivalent_nodes(train_graphs_filtered)
        with open(output_file, "w") as f:
            json.dump(train_graphs_filtered, f)
        return train_graphs_filtered
    else:
        with open(output_file, "r") as f:
            return json.load(f)


def run_graph_mining(miningConf):
    """
    Run graph mining experiment
    :param miningConf: experimental configuration class
    """
    config = miningConf.config
    kb_filter="_kbfilter" if config['edge_pruning'] else ""
    node_pruning = "_prune" if config['node_pruning'] else ""

    # Load dataset categories
    obj_categories, rel_categories = miningConf.load_categories()

    sel_train_graphs_data_path = os.path.join(miningConf.graph_mining_dir, f"train_graphs{kb_filter}{node_pruning}_{config['alg']}.data")
    exp_name = get_exp_name(miningConf)
    sel_freq_graphs_path = os.path.join(miningConf.graph_mining_dir, exp_name+'.json')

    if not os.path.exists(miningConf.graph_mining_dir):
        os.makedirs(miningConf.graph_mining_dir)

    # Check whether graph data has already been converted for
    if not os.path.exists(sel_train_graphs_data_path):
        print(f"Preparing graphs for {config['alg']}...")
        train_graphs_filtered = prepare_graphs_with_KB(miningConf)

        if config['alg']=='gspan':
            # Convert json graphs to the correct format for gspan mining.
            prepare_gspan_graph_data(sel_train_graphs_data_path, train_graphs_filtered, obj_categories, rel_categories)
        elif config['alg']=='subdue':
            prepare_subdue_graph_data(sel_train_graphs_data_path, train_graphs_filtered, obj_categories, rel_categories)

    # Mining of frequent graphs
    if config['alg'] == 'gspan':
        # Necessary because gspan program outputs with the same name of the input file
        tmp_input = os.path.join(miningConf.graph_mining_dir, exp_name+".data")
        copyfile(sel_train_graphs_data_path, tmp_input)
        run_gspan_mining(tmp_input, config['minsup'], sel_freq_graphs_path, obj_categories, rel_categories)
        os.remove(tmp_input)
    elif config['alg'] == 'subdue':
        run_subdue_mining(sel_train_graphs_data_path, config['nsubs'], sel_freq_graphs_path, obj_categories, rel_categories)

def read_freqgraphs(miningConf):
    """
    Read Json frequent graphs generated with run_graph_mining()
    :param miningConf: experimental configuration class
    :return: loaded json frequent graphs
    """
    exp_name = get_exp_name(miningConf)

    # Read frequent graphs
    freq_graphs_path = os.path.join(miningConf.graph_mining_dir, exp_name + '.json')
    with open(freq_graphs_path, 'r') as f:
        freq_graphs = json.load(f)
    return freq_graphs

def analyze_graphs(miningConf):
    """
    Compute statistics (avg. nodes, distinct classes, ...) on the extracted frequent subgraphs.
    :param miningConf: experimental configuration class
    :return: dictionary with statistics
    """
    # Read frequent graphs
    freq_graphs = read_freqgraphs(miningConf)
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

    config = miningConf.config
    if 'minsup' not in config:
        config['minsup'] = None
    res_dict = {"Minsup":config['minsup'],
                "Edge pruning": 'Y' if config['edge_pruning'] else 'N',
                "Node pruning": 'Y' if config['node_pruning'] else 'N',
                "N. graphs": len(freq_graphs), "N. distinct classes": len(dist_classes),
                "N. distinct class sets": len(dist_sets),
                "Distinct set ratio": round(len(dist_sets)/len(freq_graphs), 3),
                "Avg. nodes": round(tot_nodes / len(freq_graphs), 2),
                "Avg. distinct classes": round(tot_dist_nodes / len(freq_graphs),2), "Max. distinct classes": max_dist_nodes,
                "Distinct node ratio":round(nodes_nodes_dist/len(freq_graphs),2),
                "Std. nodes": round(np.std(std_nodes),2)}
    return res_dict


def print_graphs(miningConf, subsample=True, pdfformat=True, alternate_colors=True):
    """
    Print graphs to files
    :param miningConf: experimental configuration class
    :param subsample: subsample graphs if >500
    :param pdfformat: True to print pdf, False to print png
    :param alternate_colors: True if you want to alternate different colors for nodes
    """
    exp_name = get_exp_name(miningConf)
    # Read
    sel_freq_graphs_path = os.path.join(miningConf.graph_mining_dir, exp_name + '.json')
    with open(sel_freq_graphs_path, 'r') as f:
        graphs = json.load(f)

    out_path = os.path.join(miningConf.graph_mining_dir, f"charts/{exp_name}")
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

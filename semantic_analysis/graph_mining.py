import json
import os
from shutil import copyfile
import numpy as np
from scipy.stats import entropy
from config import graph_mining_dir, kb_pairwise_json_path, train_graphs_json_path
from semantic_analysis.knowledge_base import get_sup_ent_lists, filter_kb_histograms, filter_graph_edges, \
    prune_equivalent_nodes

from semantic_analysis.gspan_mining.mining import prepare_gspan_graph_data, run_gspan_mining
from semantic_analysis.subdue_mining.mining import prepare_subdue_graph_data, run_subdue_mining

def get_exp_name(experiment):
    """
    Get experiment name, given configuration dictionary
    Experiment name can be used for generating its associated files
    :param experiment: experiment configuration (dictionary)
    """
    kb_filter="_kbfilter" if experiment['filter_kb'] else ""
    prune_nodes = "_prune" if experiment['prune_nodes'] else ""
    if experiment['alg']=='gspan':
        exp_name = f"train_freqGraph{kb_filter}{prune_nodes}_{experiment['alg']}_{str(experiment['minsup'])[2:]}"
    else:
        exp_name = f"train_freqGraph{kb_filter}{prune_nodes}_{experiment['alg']}_{experiment['nsubs']}"
    return exp_name


def prepare_graphs_with_KB(experiment):
    """
    Given experiment configuration, return training COCO graphs, pruned and filtered according to KB
    :param experiment: experiment configuration (dictionary)
    """
    kb_filter="_kbfilter" if experiment['filter_kb'] else ""
    prune_nodes = "_prune" if experiment['prune_nodes'] else ""
    output_file = os.path.join(graph_mining_dir, f"train_graphs{kb_filter}{prune_nodes}.json")
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
        if experiment['filter_kb']:
            kb_filtered = filter_kb_histograms(kb, min_sup, max_entropy)
        else:
            kb_filtered = filter_kb_histograms(kb, min_sup, 100)  # No filter for entropy
        # Read COCO Train graphs
        with open(train_graphs_json_path, 'r') as f:
            train_graphs = json.load(f)
        # Filter graphs with KB
        print("Filtering graphs with KB...")
        train_graphs_filtered = filter_graph_edges(kb_filtered, train_graphs)
        # Node pruning (prune equivalent nodes to reduce redundancies and to reduce mining time)
        if experiment['prune_nodes']:
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
    kb_filter="_kbfilter" if experiment['filter_kb'] else ""
    prune_nodes = "_prune" if experiment['prune_nodes'] else ""

    sel_train_graphs_data_path = os.path.join(graph_mining_dir, f"train_graphs{kb_filter}{prune_nodes}_{experiment['alg']}.data")
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
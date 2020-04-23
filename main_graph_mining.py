from scipy.stats import entropy
import numpy as np
from shutil import copyfile
from datetime import datetime
import pyximport
pyximport.install(language_level=3)

from semantic_analysis.gspan_mining.graph_utils import print_graph_picture
from semantic_analysis.subdue_mining.mining import prepare_subdue_graph_data, run_subdue_mining
from semantic_analysis.knowledge_base import get_sup_ent_lists, filter_kb_histograms, filter_graph_edges, \
    prune_equivalent_nodes
from semantic_analysis.graph_utils import json_to_nx
from semantic_analysis.gspan_mining.mining import run_gspan_mining, prepare_gspan_graph_data
from config import train_graphs_json_path, kb_pairwise_json_path, graph_mining_dir
import json
import os
import sys


def graph_mining(experiment):
    kb_filter="_kbfilter" if experiment['filter_kb'] else ""
    prune_nodes = "_prune" if experiment['prune_nodes'] else ""

    sel_train_graphs_data_path = os.path.join(graph_mining_dir, f"train_graphs{kb_filter}{prune_nodes}_{experiment['alg']}.data")
    if experiment['alg']=='gspan':
        exp_name = f"train_freqGraph{kb_filter}{prune_nodes}_{experiment['alg']}_{str(experiment['minsup'])[2:]}"
    else:
        exp_name = f"train_freqGraph{kb_filter}{prune_nodes}_{experiment['alg']}_{experiment['nsubs']}"
    sel_freq_graphs_path = os.path.join(graph_mining_dir, exp_name+'.json')

    if not os.path.exists(graph_mining_dir):
        os.makedirs(graph_mining_dir)

    # Check whether graph data has already been converted for
    if not os.path.exists(sel_train_graphs_data_path):
        print(f"Preparing graphs for {experiment['alg']}...")

        # Read KB
        with open(kb_pairwise_json_path, 'r') as f:
            kb = json.load(f)
        # Get support and entropy
        sup, entr = get_sup_ent_lists(kb)
        max_entropy = entropy([1 / 3, 1 / 3, 1 / 3])
        med = np.median(np.log10(sup))
        min_sup = int(round(10 ** med))

        if experiment['filter_kb']:
            kb_filtered = filter_kb_histograms(kb, min_sup, max_entropy)
        else:
            kb_filtered = filter_kb_histograms(kb, min_sup, 100)  # No filter for entropy

        with open(train_graphs_json_path, 'r') as f:
            train_graphs = json.load(f)
        train_graphs_filtered = filter_graph_edges(kb_filtered, train_graphs)

        if experiment['prune_nodes']:
            train_graphs_filtered = prune_equivalent_nodes(train_graphs_filtered)

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


def main():
    ### Choose methods to be run ###
    class RUN_CONFIG:
        graph_mining = True
        print_graphs = False
        experiment = 5 # Index of the experiment configuration to be run (if not specified as command-line argument)

    # Experiment configuration
    experiments = [{'alg':'gspan', 'filter_kb':True, 'prune_nodes':False, 'minsup':0.1},  #5s
                   {'alg':'gspan', 'filter_kb':True, 'prune_nodes':False, 'minsup':0.01},  #4h,30m
                   {'alg': 'subdue', 'filter_kb': True, 'prune_nodes':False, 'nsubs': 10},  #12h
                   {'alg': 'subdue', 'filter_kb': True, 'prune_nodes':False, 'nsubs': 100},  #12h
                   {'alg': 'subdue', 'filter_kb': True, 'prune_nodes':False, 'nsubs': 10000},  #12h

                   {'alg': 'gspan', 'filter_kb': True, 'prune_nodes': True, 'minsup': 0.1},  # 1s
                   {'alg': 'gspan', 'filter_kb': True, 'prune_nodes': True, 'minsup': 0.01},  # 2s
                   {'alg': 'gspan', 'filter_kb': True, 'prune_nodes': True, 'minsup': 0.005},  # 3s
                   {'alg': 'gspan', 'filter_kb': True, 'prune_nodes': True, 'minsup': 0.001},  # 7s
                   {'alg': 'subdue', 'filter_kb': True, 'prune_nodes': True, 'nsubs': 10000},  # 17m

                   {'alg': 'gspan', 'filter_kb': False, 'prune_nodes': True, 'minsup': 0.01},  # 12h 36m
                   ]

    # Experiment selection
    if len(sys.argv) < 2:
        exp = RUN_CONFIG.experiment
    else:
        exp = int(sys.argv[1])
    print(f"Selected experiment: {experiments[exp]}")

    if RUN_CONFIG.graph_mining:
        start_time = datetime.now()
        graph_mining(experiments[exp])
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))
    if RUN_CONFIG.print_graphs:
        experiment = experiments[exp]
        kb_filter = "_kbfilter" if experiment['filter_kb'] else ""
        prune_nodes = "_prune" if experiment['prune_nodes'] else ""
        if experiment['alg'] == 'gspan':
            exp_name = f"train_freqGraph{kb_filter}{prune_nodes}_{experiment['alg']}_{str(experiment['minsup'])[2:]}"
        else:
            exp_name = f"train_freqGraph{kb_filter}{prune_nodes}_{experiment['alg']}_{experiment['nsubs']}"
        sel_freq_graphs_path = os.path.join(graph_mining_dir, exp_name + '.json')
        with open(sel_freq_graphs_path, 'r') as f:
            out_path = f"../COCO/gmining/charts/{exp_name}"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            graphs = json.load(f)
            i = 0
            for g_dict in graphs:
                sup = g_dict['sup']
                g = json_to_nx(g_dict['g'])
                print_graph_picture(f"{out_path}/g{i}_s_{sup}.png", g)
                i+=1

    ################################## TODO: random walk multiple sullo stesso grafo, poi fare sequence mining #############################



if __name__ == '__main__':
    main()



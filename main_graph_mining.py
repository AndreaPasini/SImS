from scipy.stats import entropy
import numpy as np
import pyximport

from datetime import datetime
from semantic_analysis.subdue_mining.mining import prepare_subdue_graph_data, run_subdue_mining

pyximport.install(language_level=3)
from semantic_analysis.knowledge_base import get_sup_ent_lists, filter_kb_histograms, filter_graph_edges
pyximport.install(language_level=3)
from semantic_analysis.gspan_mining.mining import run_gspan_mining, prepare_gspan_graph_data

from config import freq_train_graphs_path, train_graphs_json_path, kb_pairwise_json_path, graph_mining_dir
import json
import os
import sys

### Choose an action ###
action = 'GRAPH_MINING'
#action = 'PRINT_GRAPHS'
########################


def graph_mining(experiment):
    kb_filter="_kbfilter" if experiment['filter_kb'] else ""

    sel_train_graphs_data_path = os.path.join(graph_mining_dir, f"train_graphs{kb_filter}_{experiment['alg']}.data")
    if experiment['alg']=='gspan':
        exp_name = f"train_freqGraph{kb_filter}_{experiment['alg']}_{str(experiment['minsup'])[2:]}"
    else:
        exp_name = f"train_freqGraph{kb_filter}_{experiment['alg']}_{experiment['nsubs']}"
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

        if experiment['alg']=='gspan':
            # Convert json graphs to the correct format for gspan mining.
            prepare_gspan_graph_data(sel_train_graphs_data_path, train_graphs_filtered)
        elif experiment['alg']=='subdue':
            prepare_subdue_graph_data(sel_train_graphs_data_path, train_graphs_filtered)

    # Mining of frequent graphs
    if experiment['alg'] == 'gspan':
        run_gspan_mining(sel_train_graphs_data_path, experiment['minsup'], sel_freq_graphs_path)
    elif experiment['alg'] == 'subdue':
        run_subdue_mining(sel_train_graphs_data_path, experiment['nsubs'], sel_freq_graphs_path)


def main():

    experiments = [{'alg':'gspan', 'filter_kb':True, 'minsup':0.1},
                   {'alg':'gspan', 'filter_kb':True, 'minsup':0.01},
                   {'alg': 'subdue', 'filter_kb': True, 'nsubs': 10},
                   {'alg': 'subdue', 'filter_kb': True, 'nsubs': 100}
                   ]

    if action=='GRAPH_MINING':
        #for i in [2,3]:
        if len(sys.argv)<2:
            exp = 2
        else:
            exp = int(sys.argv[1])
            print(f"Selected experiment: {experiments[exp]}")
        start_time = datetime.now()
        graph_mining(experiments[exp])
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))
    elif action=='PRINT_GRAPHS':
        with open(freq_train_graphs_path, 'r') as f:
            graphs = json.load(f)
            i = 0
            for g_dict in graphs:
                sup = g_dict['sup']
                g = json_to_nx(g_dict['g'])
                print_graph_picture(f"../COCO/kb/charts/g{i}.png", g)
                i+=1





    ################################## TODO: random walk multiple sullo stesso grafo, poi fare sequence mining #############################



if __name__ == '__main__':
    main()



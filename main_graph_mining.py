"""
Author: Andrea Pasini
This file provides the code for TODO
"""
import io
from datetime import datetime
import pyximport
pyximport.install(language_level=3)

from scipy.stats import entropy

from config import COCO_graph_mining_dir, train_graphs_json_path, train_graphs_subset_json_path, kb_dir, \
    kb_pairwise_json_path
from semantic_analysis.datasets import MiningConf
from semantic_analysis.knowledge_base import filter_kb_histograms

pyximport.install(language_level=3)
from semantic_analysis.conceptnet.places import Conceptnet
from semantic_analysis.graph_clustering import compute_image_freqgraph_count_mat, compute_freqgraph_place_count_mat, \
    compute_image_place_count_mat
from semantic_analysis.graph_mining import run_graph_mining, print_graphs, analyze_graphs
import pandas as pd
from semantic_analysis.graph_utils import print_graph_picture
from semantic_analysis.graph_utils import json_to_nx
import json

import os
import sys




def filter_COCO_paper_experiment():

    with open(kb_pairwise_json_path, 'r') as f:
        kb = json.load(f)

    kb = filter_kb_histograms(kb, 64, entropy([1/3,1/3,1/3]))
    kb2 = {}
    for k,v in kb.items():
        kb2[k]=v['sup']

    print("Selecting COCO subset for article...")
    # Read COCO Train graphs
    with open(train_graphs_json_path, 'r') as f:
        train_graphs = json.load(f)

    categories = [{'river'}, {'surfboard'}, {'clock'}, {'book'}]
    graphs = [[] for el in categories]
    for g in train_graphs:
        labels = {node['label'] for node in g['nodes']}
        for i, cat in enumerate(categories):
            if cat.issubset(labels):
                graphs[i].append(g)
                break

    all_graphs = []
    for el in graphs:
        all_graphs.extend(el)
    with open(train_graphs_subset_json_path, 'w') as f:
        json.dump(all_graphs, f)
    print("Done.")


def main():
    ### Choose methods to be run ###
    class RUN_CONFIG:
        graph_mining = True            # Mining of frequent graphs with Gspan or Subdue (may take minutes or hours)
        analyze_freq_graphs = False      # Plot table with statistics for the different methods
        compute_image_freqgraph_count_mat = False   # Associate training COCO images to frequent graphs (7 minutes)
        compute_freqgraph_place_count_mat = False   # Associate frequent graphs to places
        compute_image_place_count_mat = False    # Associate training COCO images to places
        print_graphs = True

        associate_to_freq_graphs = False

        # Experiment configuration
        experiment = 6 # Index of the experiment configuration to be run (if not specified as command-line argument)
        # Choose a dataset:
        dataset = 'COCO'
        # dataset = 'VG'
        # Choose options:
        dataset_info = 'COCO_subset' # Experiment with only 4 COCO scenes (for paper comparisons)
        #dataset_info = None # No other options, keep dataset as it is

    # Experiment configuration
    experiments = [{'alg':'gspan', 'edge_pruning':True, 'node_pruning':False, 'minsup':0.1},  #0) 5s
                   {'alg':'gspan', 'edge_pruning':True, 'node_pruning':False, 'minsup':0.01},  #1) 4h,30m
                   {'alg': 'subdue', 'edge_pruning': True, 'node_pruning':False, 'nsubs': 10},  #2) 12h
                   {'alg': 'subdue', 'edge_pruning': True, 'node_pruning':False, 'nsubs': 100},  #3) 12h
                   {'alg': 'subdue', 'edge_pruning': True, 'node_pruning':False, 'nsubs': 10000},  #4) 12h
                   {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.1},  #5) 1s
                   {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.01},  #6) 2s
                   {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.005},  #7) 3s
                   {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.001},  #8) 7s
                   {'alg': 'subdue', 'edge_pruning': True, 'node_pruning': True, 'nsubs': 10000},  #9) 17m

                   {'alg': 'gspan', 'edge_pruning': False, 'node_pruning': True, 'minsup': 0.01},  # 10) 12h 36m
                   {'alg': 'gspan', 'edge_pruning': False, 'node_pruning': False, 'minsup': 0.01},  # 11) 15h 55m
                   ]

    # Experiment selection
    if len(sys.argv) < 2:
        exp = RUN_CONFIG.experiment
    else:
        exp = int(sys.argv[1])

    # Setup configuration attributes
    experiment = experiments[exp]
    experiment['dataset'] = RUN_CONFIG.dataset
    experiment['dataset_info'] = RUN_CONFIG.dataset_info

    if RUN_CONFIG.graph_mining:
        if experiment['dataset_info'] == 'COCO_subset':
            filter_COCO_paper_experiment()

        print(f"Selected experiment: {experiments[exp]}")
        start_time = datetime.now()
        miningConf = MiningConf(experiment)
        run_graph_mining(miningConf)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))
    if RUN_CONFIG.analyze_freq_graphs:
        if RUN_CONFIG.dataset=='COCO':
            exp_list = [11, 1, 6, 8, 4, 9]    # Selected experiments for analyzing statistics
        else:
            exp_list = [11,6]

        results = []
        for exp in exp_list:
            experiment = experiments[exp]
            experiment['dataset']=RUN_CONFIG.dataset
            experiment['dataset_info'] = RUN_CONFIG.dataset_info
            miningConf = MiningConf(experiment)
            res = analyze_graphs(miningConf)
            results.append(res)
        print("Graph mining statistics.")
        res_df = pd.DataFrame(results, columns=["Minsup","Edge pruning","Node pruning","N. graphs",
                                                "N. distinct classes","Distinct set ratio","Avg. nodes","Std. nodes",
                                                "Distinct node ratio"])#,"Max. distinct classes","Avg. distinct classes"
        # Print latex table
        print(res_df.to_latex(index=False))


    if RUN_CONFIG.compute_image_freqgraph_count_mat:
        print(f"Selected experiment: {experiments[exp]}")
        start_time = datetime.now()
        compute_image_freqgraph_count_mat(experiment)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))
    if RUN_CONFIG.compute_freqgraph_place_count_mat:
        print(f"Selected experiment: {experiments[exp]}")
        start_time = datetime.now()
        compute_freqgraph_place_count_mat(experiment)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))

    if RUN_CONFIG.compute_image_place_count_mat:
        print(f"Selected experiment: {experiments[exp]}")
        start_time = datetime.now()
        compute_image_place_count_mat()
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))

    if RUN_CONFIG.print_graphs:
        miningConf = MiningConf(experiment)
        print(f"Selected experiment: {experiments[exp]}")
        # Print graphs to file
        # print_graphs(miningConf, subsample = [154, 155, 784, 786]) # for article images (issues of graph mining), exp=11
        # print_graphs(miningConf, subsample=list(range(1100, 1190)))
        print_graphs(miningConf)






    if RUN_CONFIG.associate_to_freq_graphs:
        with open(os.path.join(COCO_graph_mining_dir, 'train_freqGraph_kbfilter_prune_gspan_005.json')) as f:
            freq_graphs = json.load(f)
        conceptnet = Conceptnet()

        def associate_graph(g, i):
            rank = conceptnet.rank_related_places(g['g'])
            if len(rank) > 0:
                print_graph_picture(f"{COCO_graph_mining_dir}/clusters/{rank[0][0]}_{i}.png", json_to_nx(g['g']))

        for i, g in enumerate(freq_graphs):
            associate_graph(g, i)



    ################################## TODO: random walk multiple sullo stesso grafo, poi fare sequence mining #############################



if __name__ == '__main__':
    main()



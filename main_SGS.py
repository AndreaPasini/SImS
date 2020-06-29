"""
Author: Andrea Pasini
This file provides the code for TODO
"""
from datetime import datetime
import pyximport
pyximport.install(language_level=3)

from scipy.stats import entropy

from config import COCO_SGS_dir, COCO_train_graphs_json_path, COCO_train_graphs_subset_json_path, COCO_PRS_dir, \
    COCO_PRS_json_path
from sims.sims_config import SImS_config
from sims.prs import filter_PRS_histograms

pyximport.install(language_level=3)
from sims.conceptnet.places import Conceptnet
from sims.graph_clustering import compute_image_freqgraph_count_mat, compute_freqgraph_place_count_mat, \
    compute_image_place_count_mat
from sims.graph_mining import build_SGS, print_graphs, analyze_graphs
import pandas as pd
from sims.graph_utils import print_graph_picture
from sims.graph_utils import json_to_nx
import json

import os
import sys




def filter_COCO_paper_experiment():

    with open(COCO_PRS_json_path, 'r') as f:
        kb = json.load(f)

    kb = filter_PRS_histograms(kb, 64, entropy([1 / 3, 1 / 3, 1 / 3]))
    kb2 = {}
    for k,v in kb.items():
        kb2[k]=v['sup']

    print("Selecting COCO subset for article...")
    # Read COCO Train graphs
    with open(COCO_train_graphs_json_path, 'r') as f:
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
    with open(COCO_train_graphs_subset_json_path, 'w') as f:
        json.dump(all_graphs, f)
    print("Done.")


def main():
    ### Choose methods to be run ###
    class RUN_CONFIG:
        compute_SGS = False          # Compute the Scene Graph Summary
        analyze_SGS = False          # Plot table with statistics for the different mining methods
        print_SGS_graphs = False    # Plot SGS scene graphs
        compute_image_freqgraph_count_mat = False   # Associate training COCO images to frequent graphs (7 minutes)
        compute_freqgraph_place_count_mat = False   # Associate frequent graphs to places
        compute_image_place_count_mat = False       # Associate training COCO images to places
        associate_to_freq_graphs = False

        # Experiment configuration
        experiment = 6 # Index of the experiment configuration to be run (if not specified as command-line argument)
        # Choose a dataset:
        dataset = 'COCO'
        # dataset = 'COCO_subset' # Experiment with only 4 COCO scenes (for paper comparisons)
        # dataset = 'VG'

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
        experiment = experiments[RUN_CONFIG.experiment]
    else:
        experiment = experiments[int(sys.argv[1])]

    # SImS configuration
    config = SImS_config(RUN_CONFIG.dataset)
    config.setSGS_params(experiment)

    if RUN_CONFIG.compute_SGS:
        if RUN_CONFIG.dataset == 'COCO_subset':
            filter_COCO_paper_experiment()

        print(f"Selected experiment: {experiment}")
        start_time = datetime.now()
        build_SGS(config)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))







    if RUN_CONFIG.analyze_SGS:
        if RUN_CONFIG.dataset=='COCO':
            exp_list = [11, 1, 6, 8, 4, 9]    # Selected experiments for analyzing statistics
        else:
            exp_list = [11,6]

        results = []
        for selected_experiment in exp_list:
            config.setSGS_params(experiments[selected_experiment])
            res = analyze_graphs(config)
            results.append(res)
        print("Graph mining statistics.")
        res_df = pd.DataFrame(results, columns=["Minsup","Edge pruning","Node pruning","N. graphs",
                                                "Sub-topic Coverage","Distinct Set Ratio","Avg. nodes","Std. nodes",
                                                "Distinct Node Ratio"])#,"Max. distinct classes","Avg. distinct classes"
        # Print latex table
        print(res_df.to_latex(index=False))

    if RUN_CONFIG.print_SGS_graphs:
        print(f"Selected experiment: {experiment}")
        # Print graphs to file

        # For the 4 article images (issues of graph mining), exp=11
        #print_graphs(config, subsample = [154, 155, 784, 786])
        # For the 2 examples on node pruning
        #print_graphs(config, subsample=[1175])

        # print_graphs(config, subsample=list(range(1100, 1190)))
        print_graphs(config)


    if RUN_CONFIG.compute_image_freqgraph_count_mat:
        print(f"Selected experiment: {experiments[selected_experiment]}")
        start_time = datetime.now()
        compute_image_freqgraph_count_mat(experiment)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))
    if RUN_CONFIG.compute_freqgraph_place_count_mat:
        print(f"Selected experiment: {experiments[selected_experiment]}")
        start_time = datetime.now()
        compute_freqgraph_place_count_mat(experiment)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))

    if RUN_CONFIG.compute_image_place_count_mat:
        print(f"Selected experiment: {experiments[selected_experiment]}")
        start_time = datetime.now()
        compute_image_place_count_mat()
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))








    if RUN_CONFIG.associate_to_freq_graphs:
        with open(os.path.join(COCO_SGS_dir, 'train_freqGraph_kbfilter_prune_gspan_005.json')) as f:
            freq_graphs = json.load(f)
        conceptnet = Conceptnet()

        def associate_graph(g, i):
            rank = conceptnet.rank_related_places(g['g'])
            if len(rank) > 0:
                print_graph_picture(f"{COCO_SGS_dir}/clusters/{rank[0][0]}_{i}.png", json_to_nx(g['g']))

        for i, g in enumerate(freq_graphs):
            associate_graph(g, i)



    ################################## TODO: random walk multiple sullo stesso grafo, poi fare sequence mining #############################



if __name__ == '__main__':
    main()



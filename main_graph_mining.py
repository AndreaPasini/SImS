"""
Author: Andrea Pasini
This file provides the code for TODO
"""
import io
from datetime import datetime
import pyximport

from config import COCO_graph_mining_dir

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


def main():
    ### Choose methods to be run ###
    class RUN_CONFIG:
        graph_mining = False            # Mining of frequent graphs with Gspan or Subdue (may take minutes or hours)
        analyze_freq_graphs = True
        compute_image_freqgraph_count_mat = False   # Associate training COCO images to frequent graphs (7 minutes)
        compute_freqgraph_place_count_mat = False   # Associate frequent graphs to places
        compute_image_place_count_mat = False    # Associate training COCO images to places
        print_graphs = False

        associate_to_freq_graphs = False

        experiment = 6 # Index of the experiment configuration to be run (if not specified as command-line argument)
        #dataset = 'COCO'
        dataset = 'VG'

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
    experiment = experiments[exp]
    experiment['dataset'] = RUN_CONFIG.dataset

    if RUN_CONFIG.graph_mining:
        print(f"Selected experiment: {experiments[exp]}")
        start_time = datetime.now()
        run_graph_mining(experiment)
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
            res = analyze_graphs(experiment)
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
        print(f"Selected experiment: {experiments[exp]}")
        # Print graphs to file
        # print_graphs(experiment, subsample = [154, 155, 784, 786]) # for article images (issues of graph mining), exp=11
        # print_graphs(experiment, subsample=list(range(1100, 1190)))
        print_graphs(experiment)






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



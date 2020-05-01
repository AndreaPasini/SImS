"""
Author: Andrea Pasini
This file provides the code for TODO
"""
import io
from datetime import datetime
import pyximport
pyximport.install(language_level=3)
from semantic_analysis.conceptnet.places import Conceptnet
from semantic_analysis.graph_clustering import compute_image_freqgraph_count_mat, compute_freqgraph_place_count_mat, \
    compute_image_place_count_mat
from semantic_analysis.graph_mining import run_graph_mining



from semantic_analysis.graph_utils import print_graph_picture, json_to_graphviz
from semantic_analysis.graph_utils import json_to_nx
from config import graph_mining_dir
import json
import os
import sys


def main():
    ### Choose methods to be run ###
    class RUN_CONFIG:
        graph_mining = True            # Mining of frequent graphs with Gspan or Subdue (may take minutes or hours)
        compute_image_freqgraph_count_mat = False   # Associate training COCO images to frequent graphs (7 minutes)
        compute_freqgraph_place_count_mat = False   # Associate frequent graphs to places
        compute_image_place_count_mat = False    # Associate training COCO images to places

        associate_to_freq_graphs = False
        print_graphs = False
        experiment = 11 # Index of the experiment configuration to be run (if not specified as command-line argument)

    # Experiment configuration
    experiments = [{'alg':'gspan', 'filter_kb':True, 'prune_nodes':False, 'minsup':0.1},  #0) 5s
                   {'alg':'gspan', 'filter_kb':True, 'prune_nodes':False, 'minsup':0.01},  #1) 4h,30m
                   {'alg': 'subdue', 'filter_kb': True, 'prune_nodes':False, 'nsubs': 10},  #2) 12h
                   {'alg': 'subdue', 'filter_kb': True, 'prune_nodes':False, 'nsubs': 100},  #3) 12h
                   {'alg': 'subdue', 'filter_kb': True, 'prune_nodes':False, 'nsubs': 10000},  #4) 12h
                   {'alg': 'gspan', 'filter_kb': True, 'prune_nodes': True, 'minsup': 0.1},  #5) 1s
                   {'alg': 'gspan', 'filter_kb': True, 'prune_nodes': True, 'minsup': 0.01},  #6) 2s
                   {'alg': 'gspan', 'filter_kb': True, 'prune_nodes': True, 'minsup': 0.005},  #7) 3s
                   {'alg': 'gspan', 'filter_kb': True, 'prune_nodes': True, 'minsup': 0.001},  #8) 7s
                   {'alg': 'subdue', 'filter_kb': True, 'prune_nodes': True, 'nsubs': 10000},  #9) 17m

                   {'alg': 'gspan', 'filter_kb': False, 'prune_nodes': True, 'minsup': 0.01},  # 10) 12h 36m
                   {'alg': 'gspan', 'filter_kb': False, 'prune_nodes': False, 'minsup': 0.01},  # 11) 12h 36m
                   ]

    # Experiment selection
    if len(sys.argv) < 2:
        exp = RUN_CONFIG.experiment
    else:
        exp = int(sys.argv[1])
    experiment = experiments[exp]
    print(f"Selected experiment: {experiments[exp]}")

    if RUN_CONFIG.graph_mining:
        start_time = datetime.now()
        run_graph_mining(experiment)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))
    if RUN_CONFIG.compute_image_freqgraph_count_mat:
        start_time = datetime.now()
        compute_image_freqgraph_count_mat(experiment)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))
    if RUN_CONFIG.compute_freqgraph_place_count_mat:
        start_time = datetime.now()
        compute_freqgraph_place_count_mat(experiment)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))

    if RUN_CONFIG.compute_image_place_count_mat:
        start_time = datetime.now()
        compute_image_place_count_mat()
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))






    if RUN_CONFIG.associate_to_freq_graphs:
        with open(os.path.join(graph_mining_dir, 'train_freqGraph_kbfilter_prune_gspan_005.json')) as f:
            freq_graphs = json.load(f)
        conceptnet = Conceptnet()

        def associate_graph(g, i):
            rank = conceptnet.rank_related_places(g['g'])
            if len(rank) > 0:
                print_graph_picture(f"{graph_mining_dir}/clusters/{rank[0][0]}_{i}.png", json_to_nx(g['g']))

        for i, g in enumerate(freq_graphs):
            associate_graph(g, i)



    if RUN_CONFIG.print_graphs:

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
                #g = json_to_nx(g_dict['g'])
                g = json_to_graphviz(g_dict['g'])
                #g.render(f"{out_path}/g{i}_s_{sup}.png", format='png')
                with open(f"{out_path}/g{i}_s_{sup}.png", "wb") as f:
                    #f.write(bytesio_object.getbuffer())
                    f.write(g.pipe(format='png'))
                #print_graph_picture(f"{out_path}/g{i}_s_{sup}.png", g)
                i+=1

    ################################## TODO: random walk multiple sullo stesso grafo, poi fare sequence mining #############################



if __name__ == '__main__':
    main()



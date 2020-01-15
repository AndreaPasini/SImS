import pandas as pd
import pyximport
pyximport.install(language_level=3)
from semantic_analysis.gspan_mining.graph import json_to_nx, print_graph_picture
from semantic_analysis.gspan_mining.gspan import gSpan
from semantic_analysis.gspan_mining.mining import prepare_gspan_graph_data, gspan_to_final_graphs, _read_graphs

from config import position_dataset_res_dir, kb_freq_graphs_path, train_graphs_json_path, COCO_train_json_path
import json
import networkx as nx
import os

# Configuration
train_graphs_data_path = os.path.join(position_dataset_res_dir, 'train_graphs.data')

### Choose an action ###
#action = 'GRAPH_MINING'
action = 'PRINT_GRAPHS'
########################


def graph_mining():
    # Convert json graphs to the correct format for gspan mining.
    prepare_gspan_graph_data(train_graphs_data_path, train_graphs_json_path)
    print("Mining...")
    ############# Con supporto 0.01 ci mette 899 secondi e trova 11500 grafi frequenti
    os.system('./gSpan-64 -f ../COCO/positionDataset/results/train_graphs.data -s 0.02 -o')
    print("Done.")
    freq_graphs = _read_graphs('../COCO/positionDataset/results/train_graphs.data.fp')
    print("Saving frequent graphs...")
    with open(kb_freq_graphs_path, 'w') as f:
        f.write(json.dumps(freq_graphs))
    print("Done.")

def main():

    if action=='GRAPH_MINING':
        graph_mining()
    elif action=='PRINT_GRAPHS':
        with open(kb_freq_graphs_path, 'r') as f:
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
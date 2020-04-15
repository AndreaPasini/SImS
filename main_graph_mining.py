from scipy.stats import entropy
import numpy as np
import pyximport

pyximport.install(language_level=3)
from semantic_analysis.knowledge_base import get_sup_ent_lists, filter_kb_histograms, filter_graph_edges

pyximport.install(language_level=3)
from semantic_analysis.gspan_mining.graph import json_to_nx, print_graph_picture
from semantic_analysis.gspan_mining.mining import prepare_gspan_graph_data, _read_graphs

from config import freq_train_graphs_path, train_graphs_json_path, train_graphs_data_path, kb_pairwise_json_path, \
    train_graphs_data_kbfilter_path, freq_train_graphs_kbfilter_path, kb_dir, position_dataset_res_dir, graph_mining_dir
import json
import os

### Choose an action ###
action = 'GRAPH_MINING'
#action = 'PRINT_GRAPHS'
########################


def graph_mining(experiment):
    kb_filter="_kbfilter" if experiment['filter_kb'] else ""

    sel_train_graphs_data_path = os.path.join(graph_mining_dir, f"train_graphs{kb_filter}_{experiment['alg']}.data")
    exp_name = f"train_freqGraph{kb_filter}_{experiment['alg']}_{str(experiment['minsup'])[2:]}"
    sel_freq_graphs_path = os.path.join(graph_mining_dir, exp_name+'.json')

    if not os.path.exists(graph_mining_dir):
        os.makedirs(graph_mining_dir)

    # Check whether graph data has already been converted for
    if not os.path.exists(sel_train_graphs_data_path):
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
        train_graph_filtered = filter_graph_edges(kb_filtered, train_graphs)

        # Convert json graphs to the correct format for gspan mining.
        prepare_gspan_graph_data(sel_train_graphs_data_path, train_graph_filtered)



    print("Mining...")

    ############# Con supporto 0.01 ci mette 899 secondi e trova 11500 grafi frequenti
    minsup_graph = min_sup / len(train_graphs)  # 64/len()

    # GSpan, c implementation (https://www.researchgate.net/publication/296573357_gSpan_Implementation,
    # https://sites.cs.ucsb.edu/~xyan/software/gSpan.htm
    # gSpan: graph-based substructure pattern mining (ICDM 2003),)
    os.system(f'./gSpan-64 -f {sel_train_graphs_data_path} -s {minsup_graph} -o')
    print("Done.")
    freq_graphs = _read_graphs(f'{sel_train_graphs_data_path}.fp')
    print("Saving frequent graphs...")
    with open(sel_freq_graphs_path, 'w') as f:
        f.write(json.dumps(freq_graphs))
    print("Done.")

def main():

    experiments = [{'alg':'gspan', 'filter_kb':True, 'minsup':0.1},
                   {'alg':'gspan', 'filter_kb':True, 'minsup':0.01},
                   {'alg': 'subdue', 'filter_kb': True, 'minsup': 0.1},
                   {'alg': 'subdue', 'filter_kb': True, 'minsup': 0.01}
                   ]

    if action=='GRAPH_MINING':
        graph_mining(experiments[0])
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



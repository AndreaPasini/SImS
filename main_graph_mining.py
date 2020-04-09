from scipy.stats import entropy
import numpy as np
import pyximport

pyximport.install(language_level=3)
from semantic_analysis.knowledge_base import get_sup_ent_lists, filter_kb_histograms, filter_graph_edges

pyximport.install(language_level=3)
from semantic_analysis.gspan_mining.graph import json_to_nx, print_graph_picture
from semantic_analysis.gspan_mining.mining import prepare_gspan_graph_data, _read_graphs

from config import freq_train_graphs_path, train_graphs_json_path, train_graphs_data_path, kb_pairwise_json_path, \
    train_graphs_data_kbfilter_path, freq_train_graphs_kbfilter_path
import json
import os

### Choose an action ###
action = 'GRAPH_MINING'
#action = 'PRINT_GRAPHS'
########################


def graph_mining():
    filter_KB = True

    # Read KB
    with open(kb_pairwise_json_path, 'r') as f:
        kb = json.load(f)
    # Get support and entropy
    sup, entr = get_sup_ent_lists(kb)
    max_entropy = entropy([1 / 3, 1 / 3, 1 / 3])
    med = np.median(np.log10(sup))
    min_sup = int(round(10 ** med))

    if filter_KB:
        kb_filtered = filter_kb_histograms(kb, min_sup, max_entropy)
        sel_train_graphs_data_path = train_graphs_data_kbfilter_path
        sel_freq_graphs_path = freq_train_graphs_kbfilter_path
    else:
        kb_filtered = filter_kb_histograms(kb, min_sup, 100)  # No filter for entropy
        sel_train_graphs_data_path = train_graphs_data_path
        sel_freq_graphs_path = freq_train_graphs_path

    with open(train_graphs_json_path, 'r') as f:
        train_graphs = json.load(f)
    train_graph_filtered = filter_graph_edges(kb_filtered, train_graphs)

    ############# Con supporto 0.01 ci mette 899 secondi e trova 11500 grafi frequenti
    minsup_graph = min_sup/len(train_graphs) #64/len()

    # Convert json graphs to the correct format for gspan mining.
    prepare_gspan_graph_data(sel_train_graphs_data_path, train_graph_filtered)

    print("Mining...")

    # GSpan, c implementation (https://www.researchgate.net/publication/296573357_gSpan_Implementation,
    # gSpan: graph-based substructure pattern mining (ICDM 2003),)
    os.system(f'./gSpan-64 -f {sel_train_graphs_data_path} -s {minsup_graph} -o')
    print("Done.")
    freq_graphs = _read_graphs(f'{sel_train_graphs_data_path}.fp')
    print("Saving frequent graphs...")
    with open(sel_freq_graphs_path, 'w') as f:
        f.write(json.dumps(freq_graphs))
    print("Done.")

def main():

    if action=='GRAPH_MINING':
        graph_mining()
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



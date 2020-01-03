import pyximport
pyximport.install(language_level=3)
from semantic_analysis.gspan_mining.graph import json_to_nx, print_graph_picture
from semantic_analysis.gspan_mining.gspan import gSpan
from semantic_analysis.gspan_mining.mining import prepare_gspan_graph_data, gspan_to_final_graphs, _read_graphs

from config import position_dataset_res_dir, kb_freq_graphs_path, train_graphs_json_path
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

    # Apply mining algorithm
    # gs = gSpan(
    #     database_file_name=train_graphs_data_path,
    #     min_support=40,  # 50 troppo poco. 60 e' ok
    #     verbose=False,
    #     visualize=False,
    # )
    # gs.run()
    # gs.time_stats()
    # freq_graphs = gspan_to_final_graphs(gs.get_result())
    # print("Saving frequent graphs...")
    # with open(kb_freq_graphs_path, 'w') as f:
    #     f.write(json.dumps(freq_graphs))
    # print("Done.")

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

                #nx.draw_networkx_edges(G, pos=nx.spring_layout(G), edge_color='black', width=4, alpha=0.5)
                #nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), node_size=2, node_color='blue', alpha=0.5)
                #nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), node_size=5, node_color="k")

                #plt.figure()
                # pos = nx.spring_layout(g)
                # #nx.draw(g, pos=pos, node_color=(0.5,0.5,0.8), labels=g.nodes, with_labels = True)
                # nx.draw_networkx_edge_labels(g, pos=pos)
                #
                #
                # nx.draw(g, pos, node_color=(0.5,0.5,0.8), font_size=16, with_labels=False)
                # for p in pos:  # raise text positions
                #     pos[p][1] += 0.06
                # nx.draw_networkx_labels(g, pos, labels=g.nodes)
                #
                # #nx.draw_networkx_labels(g, pos=nx.spring_layout(g))

                print_graph_picture(f"../COCO/kb/charts/g{i}.png", g)
                i+=1
                #plt.tight_layout()

            print()

    ################################## TODO: random walk multiple sullo stesso grafo, poi fare sequence mining #############################






if __name__ == '__main__':
    main()
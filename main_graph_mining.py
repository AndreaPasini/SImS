import pyximport
pyximport.install(language_level=3)
from config import COCO_panoptic_cat_info_path, position_labels_csv_path, position_dataset_res_dir
from panopticapi.utils import load_panoptic_categ_list
from semantic_analysis.gspan_mining.gspan import gSpan
from semantic_analysis.position_classifier import train_graphs_json_path
import json
import networkx as nx
import os

# Configuration
train_graphs_data_path = os.path.join(position_dataset_res_dir, 'train_graphs.data')

def json_graph_to_gspan(graph, conv_coco_category, conv_pos_category):
    vmap = {}  # old -> new
    vindex = 0

    descr = ""
    # for all nodes
    for v in graph['nodes']:
        l = conv_coco_category[v['label']]
        vmap[v['id']] = vindex
        descr += f"v {vindex} {l}\n"
        vindex += 1
    # for all edges
    for e in graph['links']:
        l = conv_pos_category[e['pos']]  # label
        s = vmap[e['s']]  # subject
        r = vmap[e['r']]  # reference
        descr += f"e {s} {r} {l}\n"

    return descr

def create_graph_data_file():
    with open(train_graphs_json_path, 'r') as f:
        graphs = json.load(f)

    # Encode panoptic class ids
    # gspan requests as input label values that start from 2.
    coco_categories = load_panoptic_categ_list()
    conv_coco_category = {c:i+2 for i,c in enumerate(coco_categories.values())}

    # Encode position labels
    position_labels = tuple(s.strip() for s in open(position_labels_csv_path).readlines())
    conv_pos_category = {c: i + 2 for i, c in enumerate(position_labels)}

    # Prepare nodes with the correct format for graph gspan_mining
    with open(train_graphs_data_path, 'w') as f:
        for g_index, g in enumerate(graphs):
            f.write(f"t # {g_index}\n")
            f.write(json_graph_to_gspan(g, conv_coco_category, conv_pos_category))
        f.write("t # -1")


################################## TODO: random walk multiple sullo stesso grafo, poi fare sequence mining #############################

def main():
    # Convert json graphs to the correct format for gspan mining.
    #create_graph_data_file()

    gs = gSpan(
        database_file_name=train_graphs_data_path,
        min_support=5,  # 5000,
        verbose=False,
        visualize=False,
    )

    gs.run()
    gs.time_stats()

    graphs = gs._frequent_subgraphs
    for g, sup in graphs:
        # Convert graph
        for n in g.nodes:
            ...
        #res = nx.node_link_data(sn[0][0])

    #stringa = json.dumps(res)

    print("done")













    print("Done")


if __name__ == '__main__':
    main()
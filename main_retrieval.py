import json
from scipy.stats import entropy
import pyximport
pyximport.install(language_level=3)
from config import COCO_panoptic_val_graphs_json_path
from config import COCO_PRS_json_path
from sims.prs import filter_PRS_histograms, get_sup_ent_lists
from tqdm import tqdm
def get_positive_relationships(graph, kb):
    rel_list = []
    nodes = {}
    for node in graph['nodes']:
        nodes[node['id']] = node['label']
    # Print links
    for link in graph['links']:
        sub = nodes[link['s']]
        ref = nodes[link['r']]
        pos = link['pos']
        pair = f"{sub},{ref}"
        # if pair in kb:
        #     hist = kb[f"{sub},{ref}"]
        #     if pos in hist:
        #         if hist[pos] > 0.1:
        #             rel_list.append((sub, pos, ref))
        rel_list.append((sub, pos, ref))
    return rel_list

def get_graph(graphs, graph_id):
    for g in graphs:
        name = g['graph']['name']
        if name==selected:
            return g
    return None

def get_distance(g1, g2, kb):
    n1 = set([n['label'] for n in g1['nodes']])
    n2 = set([n['label'] for n in g2['nodes']])
    rel1 = set(get_positive_relationships(g1, kb))
    rel2 = set(get_positive_relationships(g2, kb))
    return - len(n1 & n2) -len(rel1 & rel2)


    # if len(rel1)>0 and len(rel2)>0:
    #     print()
    # return -len(n1 & n2) #+ 0.25*abs(len(g1['nodes'])-len(g2['nodes']))

def get_bison():
    with open('../COCO/annotations/bison_annotations.cocoval2014.json', 'r') as f:
        bison = json.load(f)
    with open('../COCO/annotations/panoptic_val2017.json', 'r') as f:
        panval17 = json.load(f)
    # with open('../COCO/annotations/panoptic_train2017.json', 'r') as f:
    #     pantrain17 = json.load(f)
    pan17val_ids = {img['id'] for img in panval17['images']}
    #pan17train_ids = {img['id'] for img in pantrain17['images']}

    bison_map = {}

    for sample in bison['data']:
        a, b = [int(img['image_filename'].split('_')[-1].split('.')[0]) for img in sample['image_candidates']]

        if a in pan17val_ids and b in pan17val_ids:
            if a in bison_map:
                bison_map[a].add(b)
            else:
                bison_map[a] = {b}
            if b in bison_map:
                bison_map[b].add(a)
            else:
                bison_map[b] = {a}
    return bison_map

def get_neighbors(val_graphs, filtered_kb, selected):
    g_sel = get_graph(val_graphs, selected)

    distances = []
    for g in val_graphs:
        name = g['graph']['name']
        if name != selected:
            dist = get_distance(g_sel, g, filtered_kb)
            distances.append((dist, name))

    sorted_dist = sorted(distances, key=lambda d: d[0])
    return sorted_dist

if __name__ == "__main__":

    with open(COCO_PRS_json_path, 'r') as f:
        json_data = json.load(f)
    with open(COCO_panoptic_val_graphs_json_path, "r") as f:
        val_graphs = json.load(f)


    bison = get_bison()
    val_graphs = [g for g in val_graphs if g['graph']['name'] in bison.keys()]

    entr3 = entropy([1 / 3, 1 / 3, 1 / 3])
    minsup = 64
    filtered_kb = filter_PRS_histograms(json_data, minsup, entr3)
    sup_filtered, ent_filtered = get_sup_ent_lists(filtered_kb)

    ranks=[]
    pbar = tqdm(total=len(bison.items()))
    for i, (k,vs) in enumerate(bison.items()):
        selected = k
        sorted_dist = get_neighbors(val_graphs, filtered_kb, selected)
        neighbors = [s[1] for s in sorted_dist]
        for v in vs:
            ranks.append(neighbors.index(v))
        pbar.update()

    pbar.close()
    with open('../COCO/anomalies/positions_PairEdges.json','w') as f:
        json.dump(ranks,f)

    # Positions:

    print("Done")






    # graphs = []
    # for g in val_graphs:
    #     img_file = getImageName(g['graph']['name'], COCO_img_val_dir, 'jpg')
    #     if os.path.exists(img_file):
    #         graphs.append(g)






    #get_distance(best_g, g_sel, filtered_kb)

    print("done")
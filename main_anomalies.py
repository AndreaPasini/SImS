import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from config import val_panoptic_graphs, likelihoods_json_path, out_panoptic_json_path, out_panoptic_dir, \
    position_classifier_path, COCO_val_json_path, \
    COCO_ann_val_dir, kb_pairwise_json_path, kb_clean_pairwise_json_path, likelihoods_path, \
    charts_anomalies_likelihoods_path, fp_chart, tp_chart, fp_tp_json_path
from main_inspection import pq_inspection
from semantic_analysis.position_classifier import create_kb_graphs


###  CHOOSE FLOW ###
generate_val_panoptic_graphs = True         # Apply position classifier to images (segmented by CNN) and compute graphs
compute_val_panoptic_likelihoods = False
analyze_likelihoods = True
####################


if __name__ == "__main__":

    if generate_val_panoptic_graphs:
        # Apply position classifier to images (segmented by CNN) and compute graphs
        create_kb_graphs(position_classifier_path, out_panoptic_json_path, out_panoptic_dir, val_panoptic_graphs)
        print("Process 'generate_val_panoptic_graphs' Completed")

    if compute_val_panoptic_likelihoods:

        with open(val_panoptic_graphs, 'r') as f:
            panoptic_graphs = json.load(f)

        with open(kb_clean_pairwise_json_path, 'r') as f:
            kb_pairwise = json.load(f)

        with open(COCO_val_json_path, 'r') as f:
            gt_json = json.load(f)
        categories = {el['id']: el for el in gt_json['categories']}

        with open(out_panoptic_json_path, 'r') as f:
            pred_json = json.load(f)

        pred_annotations = {el['image_id']: el for el in pred_json['annotations']}

        val_graph = {list(el['graph'].values())[0]: el for el in panoptic_graphs}
        likelihoods = {}
        for gt_ann in gt_json['annotations']:
            image_id = gt_ann['image_id']
            if image_id not in pred_annotations:
                continue
            fp_img = pq_inspection(gt_ann, pred_annotations[image_id], image_id, COCO_ann_val_dir, out_panoptic_dir,
                                   categories)
            graph = val_graph[image_id]
            nodes = {}
            for node in graph['nodes']:
                nodes[node['id']] = node['label']
            for link in graph['links']:
                sub = nodes[link['s']]
                ref = nodes[link['r']]
                pos = link['pos']
                item = {(sub, ref): value for key, value in kb_pairwise.items() if sub in key and ref in key}
                if len(item) != 1 or not item:
                    continue
                try:
                    likelihood = item[(sub, ref)][pos]
                except Exception:
                    likelihood = None
                support = item[(sub, ref)]['sup']
                entropy = item[(sub, ref)]['entropy']
                pair = str(link['s']) + "," + str(link['r'])
                if image_id not in likelihoods.keys():
                    likelihoods[image_id] = {'fp': fp_img["fp"],
                                             'pairs': {pair: {'l': likelihood, 's': support, 'e': entropy}}}
                else:
                    likelihoods[image_id]['pairs'].update({pair: {'l': likelihood, 's': support, 'e': entropy}})
        if not os.path.isdir(likelihoods_path):
            os.makedirs(likelihoods_path)
        with open(likelihoods_json_path, "w") as f:
            json.dump({k: v for k, v in likelihoods.items()}, f)
        print("Process 'compute_val_panoptic_likelihoods' Completed")

    if analyze_likelihoods:
        if not os.path.isdir(charts_anomalies_likelihoods_path):
            os.makedirs(charts_anomalies_likelihoods_path)

        with open(likelihoods_json_path, 'r') as f:
            val_panoptic_likelihoods = json.load(f)
        fp = []
        tp = []
        noLikelihoods = []
        for img in val_panoptic_likelihoods.values():
            for k, v in img['pairs'].items():
                if v['l'] is not None:
                    objs = k.split(",")
                    if int(objs[0]) in img['fp'] or int(objs[1]) in img['fp']:
                        fp.append(v['l'])
                    else:
                        tp.append(v['l'])
                else:
                    noLikelihoods.append(k)
        likelihoods = {'fp': {'likelihood': fp}, 'tp': {'likelihood': tp}, 'noLikelihoods': {'pairs': noLikelihoods}}
        with open(fp_tp_json_path, "w") as f:
            json.dump(likelihoods, f)
        plt.subplots(1, 1, figsize=(10, 6))
        sns.kdeplot(fp, shade=True, cut=0,  color="r", label='False Positive')
        sns.rugplot(fp, color="r")
        plt.savefig(fp_chart)
        plt.subplots(1, 1, figsize=(10, 6))
        sns.kdeplot(tp, shade=True, cut=0, color="b", label='False Positive')
        sns.rugplot(tp, color="b")
        plt.savefig(tp_chart)
        print("Process 'analyze_likelihoods' Completed")
    print("Anomalies elaboration Completed")

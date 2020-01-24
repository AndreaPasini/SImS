import os

# COCO dataset
COCO_dir = '../COCO/'
COCO_ann_val_dir = os.path.join(COCO_dir, 'annotations/panoptic_val2017/')
COCO_val_json_path = os.path.join(COCO_dir, 'annotations/panoptic_val2017.json')
COCO_ann_train_dir = os.path.join(COCO_dir, 'annotations/panoptic_train2017/')
COCO_train_json_path = os.path.join(COCO_dir, 'annotations/panoptic_train2017.json')
COCO_panoptic_cat_info_path = './classes/panoptic_coco_categories.json'
COCO_panoptic_cat_list_path = './classes/panoptic.csv'


# Position classifier
position_dataset_res_dir = os.path.join(COCO_dir, 'positionDataset/results')
position_classifier_path = os.path.join(position_dataset_res_dir, 'final_model.clf')
train_graphs_json_path = os.path.join(position_dataset_res_dir, 'train_graphs.json')
position_labels_csv_path = os.path.join(COCO_dir, 'positionDataset/training/LabelsList.csv')

# Knowledge base
kb_dir = os.path.join(COCO_dir, 'kb/')
kb_pairwise_json_path = os.path.join(kb_dir, 'pairwiseKB.json')
kb_clean_pairwise_json_path = os.path.join(kb_dir, 'pairwiseKBclean.json')
kb_freq_graphs_path = os.path.join(kb_dir, 'freqGraphs.json')
position_dataset_dir = os.path.join(COCO_dir, 'positionDataset')

# Panoptic
val_panoptic_graphs = os.path.join(position_dataset_res_dir, 'val_panoptic_graphs.json')
output_panoptic_path = os.path.join(COCO_dir, 'output/panoptic')
output_panoptic_json_path = os.path.join(COCO_dir, 'output/panoptic/panoptic.json')
cnn_nodes_links_json_path = os.path.join(position_dataset_res_dir, 'cnn_nodes_links.json')


# Anomalies
likelihoods_path = os.path.join(COCO_dir, 'anomalies/')
charts_anomalies_likelihoods_path = os.path.join(likelihoods_path, 'charts/')
fp_chart = os.path.join(charts_anomalies_likelihoods_path, 'fp_chart.png')
tp_chart = os.path.join(charts_anomalies_likelihoods_path, 'tp_chart.png')
likelihoods_json_path = os.path.join(likelihoods_path, 'val_panoptic_likelihoods.json')
fp_tp_json_path = os.path.join(likelihoods_path, 'fp_tp_likelihoods.json')
import os

# COCO dataset
COCO_dir = '../COCO/'
COCO_img_val_dir = os.path.join(COCO_dir, 'images/val2017/')
COCO_img_train_dir = os.path.join(COCO_dir, 'images/train2017/')
COCO_ann_val_dir = os.path.join(COCO_dir, 'annotations/panoptic_val2017/')
COCO_val_json_path = os.path.join(COCO_dir, 'annotations/panoptic_val2017.json')
COCO_ann_train_dir = os.path.join(COCO_dir, 'annotations/panoptic_train2017/')
COCO_train_json_path = os.path.join(COCO_dir, 'annotations/panoptic_train2017.json')
COCO_panoptic_cat_info_path = './classes/panoptic_coco_categories.json'
COCO_panoptic_cat_list_path = './classes/panoptic.csv'



# Position classifier
position_dataset_dir = os.path.join(COCO_dir, 'positionDataset/')
position_dataset_res_dir = os.path.join(position_dataset_dir, 'results/')
position_classifier_path = os.path.join(position_dataset_res_dir, 'final_model.clf')
train_graphs_json_path = os.path.join(position_dataset_res_dir, 'train_graphs.json')
position_labels_csv_path = os.path.join(position_dataset_dir, 'training/LabelsList.csv')

# Panoptic
out_panoptic_dir = os.path.join(COCO_dir, 'output/panoptic/')
out_panoptic_json_path = os.path.join(out_panoptic_dir, 'panoptic.json')
out_panoptic_val_graphs_json_path = os.path.join(position_dataset_res_dir, 'panopt_val_graphs.json')
##??
cnn_nodes_links_json_path = os.path.join(position_dataset_res_dir, 'cnn_nodes_links.json')

# Knowledge base
kb_dir = os.path.join(COCO_dir, 'kb/')
kb_pairwise_json_path = os.path.join(kb_dir, 'pairwiseKB.json')     # Contains the KB extracted from Validation Images (COCO)
kb_clean_pairwise_json_path = os.path.join(kb_dir, 'pairwiseKBclean.json')
kb_freq_graphs_path = os.path.join(kb_dir, 'freqGraphs.json')


# Anomalies
anomaly_detection_dir = os.path.join(COCO_dir, 'anomalies/')
anomaly_statistics_json_path = os.path.join(anomaly_detection_dir, 'pq_statistics.json')

charts_anomalies_likelihoods_path = os.path.join(anomaly_detection_dir, 'charts/')
fp_chart = os.path.join(charts_anomalies_likelihoods_path, 'fp_chart.png')
tp_chart = os.path.join(charts_anomalies_likelihoods_path, 'tp_chart.png')
likelihoods_json_path = os.path.join(anomaly_detection_dir, 'val_panoptic_likelihoods.json')
fp_tp_json_path = os.path.join(anomaly_detection_dir, 'fp_tp_likelihoods.json')
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
# Visual Genome (subset)
VG_dir = '../VGenome'
VG_train_json_path = os.path.join(VG_dir, 'annotations_train.json')
VG_predicates_json_path = os.path.join(VG_dir, "predicates.json")
VG_objects_json_path = os.path.join(VG_dir, "objects.json")

"""
Position classifier (COCO)
"""
position_dataset_dir = os.path.join(COCO_dir, 'positionDataset/')
position_dataset_res_dir = os.path.join(position_dataset_dir, 'results/')
position_classifier_path = os.path.join(position_dataset_res_dir, 'final_model.clf')    # Final position classifier
position_labels_csv_path = os.path.join(position_dataset_dir, 'training/LabelsList.csv')# List of position classes

"""
Panoptic segmentation results (Deeplab + Matterport rcnn)
"""
out_panoptic_dir = os.path.join(COCO_dir, 'output/panoptic/')
out_panoptic_json_path = os.path.join(out_panoptic_dir, 'panoptic.json') # Output predictions of panoptic on val set
pq_info_path = os.path.join(out_panoptic_dir, 'panoptic_quality.json') # Panoptic quality of all objects in predictions

"""
Scene graphs extracted from COCO and VG
"""
# Scene graphs of COCO train
COCO_train_graphs_json_path = os.path.join(position_dataset_res_dir, 'train_graphs.json')
# Subset of scene graphs for Paper experiments
COCO_train_graphs_subset_json_path = os.path.join(position_dataset_res_dir, 'train_graphs_subset.json')
# Scene graphs of predictions (Deeplab + Matterport rcnn) in COCO val
COCO_panoptic_val_graphs_json_path = os.path.join(position_dataset_res_dir, 'panopt_val_graphs.json')
# Scene graphs of VG train
VG_train_graphs_json_path = os.path.join(VG_dir, 'train_graphs.json')

"""
Pairwise Relationship Summary (PRS)
"""
COCO_PRS_dir = os.path.join(COCO_dir, 'PRS/')
VG_PRS_dir = os.path.join(VG_dir, 'PRS/')
COCO_PRS_json_path = os.path.join(COCO_PRS_dir, 'PRS.json') # PRS extracted from Validation Images (COCO)
VG_PRS_json_path = os.path.join(VG_PRS_dir, 'PRS.json') # PRS extracted from VG train

"""
SGS (Scene Graph Summary) generation, with graph mining
"""
COCO_SGS_dir = os.path.join(COCO_dir, 'SGS')
VG_SGS_dir = os.path.join(VG_dir, 'SGS')
"""
Conceptnet
"""
conceptnet_dir = '../ConceptNet'
conceptnet_full_csv_path = os.path.join(conceptnet_dir, 'conceptnet.csv')
conceptnet_coco_places_csv_path = os.path.join(conceptnet_dir, "conceptnet_coco_places.csv")
places_json_path = os.path.join(conceptnet_dir, 'places.json')
panoptic_concepts_csv_path = './classes/panoptic_conceptnet.csv'

"""
Graph clustering
"""
graph_clustering_dir = os.path.join(COCO_dir, 'gclustering/')
trainimage_freqgraph_csv_path = os.path.join(graph_clustering_dir,"trainimage_freqgraph.csv")   # Count matrix images->freq graphs
freqgraph_place_csv_path = os.path.join(graph_clustering_dir,"freqgraph_place.csv")  # Count matrix freq graphs -> conceptnet places
trainimage_place_csv_path = os.path.join(graph_clustering_dir,"trainimage_place.csv")  # Count matrix train image -> conceptnet places



"""
Anomalies
"""
anomaly_detection_dir = os.path.join(COCO_dir, 'anomalies/')
pairanomaly_json_path = os.path.join(anomaly_detection_dir, 'pairanomaly.json') # Pairwise anomaly detection
pairanomaly_kbfilter_json_path = os.path.join(anomaly_detection_dir, 'pairanomaly_kbfilter.json') # Pairwise anomaly detection
objectanomaly_json_path = os.path.join(anomaly_detection_dir, 'objanomaly.json') # Object anomaly detection
objectanomaly_kbfilter_json_path = os.path.join(anomaly_detection_dir, 'objanomaly_kbfilter.json') # Object anomaly detection




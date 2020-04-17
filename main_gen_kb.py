"""
 Author: Andrea Pasini
 This file provides the code for Semantic Anomaly Detection (SAD) on COCO.

"""
import pyximport
import os
from sklearn.ensemble import RandomForestClassifier

pyximport.install(language_level=3)

from datetime import datetime
from config import position_classifier_path, COCO_train_json_path, \
    COCO_ann_train_dir, train_graphs_json_path, kb_pairwise_json_path, position_dataset_res_dir, \
    out_panoptic_val_graphs_json_path, out_panoptic_dir, out_panoptic_json_path, kb_clean_pairwise_json_path
from semantic_analysis.position_classifier import validate_classifiers_grid_search, build_final_model, create_kb_graphs
from semantic_analysis.knowledge_base import create_kb_histograms

######################

### CHOOSE CLASSIFIER ###
Nearest_Neighbors = False
Linear_SVM = False
RBF_SVM = False
Decision_Tree = False
Random_Forest = True
Naive_Bayes = False
#########################

### Choose methods to be run ###
class RUN_CONFIG:
    validate_classifiers = False    # Run cross-validation for relative-position classifiers
    build_final_model = False       # Build relative-position classifier, final model
    generate_train_graphs = True    # Build graphs (object positions) for training images (118,287, may take some hours)
    generate_val_graphs = False     # Build graphs (object positions) for CNN predictions on validation set  (5,000)
    generate_kb = False             # Generate knowledge base from training graphs: save graphs and histograms

    # Choose a model for building the final classifier (used by build_final_model)
    final_classifier = RandomForestClassifier(max_depth = 10, n_estimators = 35) # Best model selected by grid-search

if __name__ == "__main__":
    start_time = datetime.now()

    if RUN_CONFIG.validate_classifiers:
        validate_classifiers_grid_search(os.path.join(position_dataset_res_dir, 'evaluation.txt'))
    elif RUN_CONFIG.build_final_model:
        build_final_model(position_classifier_path, RUN_CONFIG.final_classifier)
    elif RUN_CONFIG.generate_train_graphs:
        # Apply position classifier to images and compute graphs
        create_kb_graphs(position_classifier_path, COCO_train_json_path, COCO_ann_train_dir, train_graphs_json_path)
    elif RUN_CONFIG.generate_val_graphs:
        # Create graphs from CNN predictions (panoptic segmentation)
        create_kb_graphs(position_classifier_path, out_panoptic_json_path, out_panoptic_dir, out_panoptic_val_graphs_json_path)
    elif RUN_CONFIG.generate_kb:
        # Generate kb histograms from graphs
        create_kb_histograms(train_graphs_json_path, kb_pairwise_json_path)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

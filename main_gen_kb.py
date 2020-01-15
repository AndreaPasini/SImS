"""
 Author: Andrea Pasini
 This file provides the code for Semantic Anomaly Detection (SAD) on COCO.

"""
import pyximport
pyximport.install(language_level=3)

from datetime import datetime
from config import position_classifier_path, COCO_val_json_path, COCO_ann_val_dir, COCO_train_json_path, \
    COCO_ann_train_dir, output_panoptic_json_path, output_panoptic_path
from semantic_analysis.position_classifier import validate_classifiers, build_final_model, generate_kb, \
    validate_classifiers_grid_search
from semantic_analysis.knowledge_base import generate_kb

### CONFIGURATION ###
output_path = '../COCO/positionDataset/results/evaluation.txt'
######################

### CHOOSE CLASSIFIER ###
Nearest_Neighbors = False
Linear_SVM = False
RBF_SVM = False
Decision_Tree = False
Random_Forest = True
Naive_Bayes = False
#########################

###  CHOOSE FLOW ###
use_validate_classifiers = False    # Run cross-validation for relative-position classifiers
use_build_final_model = False       # Build relative-position classifier, final model
use_generate_kb = True              # Generate knowledge base: save graphs and histograms
####################

###  CHOOSE SET OF IMAGES ###
use_validation_image = False
use_train_image = False
use_panoptic_image = True
####################


if __name__ == "__main__":
    start_time = datetime.now()
    classifier = [Nearest_Neighbors,
                  Linear_SVM,
                  RBF_SVM,
                  Decision_Tree,
                  Random_Forest,
                  Naive_Bayes]
    if use_validate_classifiers:
        # validate_classifiers_grid_search()
        validate_classifiers(output_path)
    elif use_build_final_model:
        build_final_model(position_classifier_path, classifier)
    elif use_generate_kb:
        if use_validation_image:
            COCO_json_path, COCO_ann_dir, cnnFlag = COCO_val_json_path, COCO_ann_val_dir, False
        elif use_train_image:
            COCO_json_path, COCO_ann_dir, cnnFlag = COCO_train_json_path, COCO_ann_train_dir, False
        elif use_panoptic_image:
            COCO_json_path, COCO_ann_dir, cnnFlag = output_panoptic_json_path, output_panoptic_path, True
        generate_kb(position_classifier_path, COCO_json_path, COCO_ann_dir, cnnFlag)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

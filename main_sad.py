"""
 Author: Andrea Pasini
 This file provides the code for Semantic Anomaly Detection (SAD) on COCO.

"""
from datetime import datetime
import pyximport

from config import position_classifier_path
from semantic_analysis.position_classifier import validate_classifiers, build_final_model, analyze_statics


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
use_validate_classifiers = False
use_build_final_model = False
use_analyze_statics = True
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
    elif use_analyze_statics:
        analyze_statics(position_classifier_path)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

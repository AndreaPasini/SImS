"""
Author: Andrea Pasini
This file provides the code for
- Evaluating position classifier with grid-search
- Building relative-position classifier
"""
import pyximport
pyximport.install(language_level=3)
import os
from datetime import datetime
from sims.scene_graphs.position_classifier import validate_classifiers_grid_search, build_final_model
from sklearn.ensemble import RandomForestClassifier
from config import position_classifier_path, position_dataset_res_dir

### Choose methods to be run ###
class RUN_CONFIG:
    validate_classifiers = False    # Run cross-validation for relative-position classifiers
    build_final_model = False       # Build relative-position classifier, final model

    # Choose a model for building the final position classifier (used by build_final_model)
    final_classifier = RandomForestClassifier(max_depth=10, n_estimators=35)  # Best model selected by grid-search


if __name__ == "__main__":
    start_time = datetime.now()

    if RUN_CONFIG.validate_classifiers:
        validate_classifiers_grid_search(os.path.join(position_dataset_res_dir, 'evaluation.txt'))
    elif RUN_CONFIG.build_final_model:
        build_final_model(position_classifier_path, RUN_CONFIG.final_classifier)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))
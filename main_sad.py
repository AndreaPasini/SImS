"""
 Author: Andrea Pasini
 This file provides the code for Semantic Anomaly Detection (SAD) on COCO.

"""
import json
import operator
import os
import pickle
from collections import defaultdict
from datetime import datetime
from multiprocessing.pool import Pool
from os import listdir

import numpy as np
import pandas as pd
import pyximport
from tqdm import tqdm

from panopticapi.utils import load_png_annotation
from semantic_analysis.position_classifier import validate_classifiers, build_final_model, \
    validate_classifiers_grid_search
from semantic_analysis.algorithms import image2strings, get_features, compute_string_positions
pyximport.install(language_level=3)

### CONFIGURATION ###
output_path = '../COCO/positionDataset/results/evaluation.txt'
fileModel_path = '../COCO/positionDataset/results/final_model.clf'
path_json_file = '../COCO/annotations/panoptic_val2017.json'
path_annot_folder = '../COCO/annotations/panoptic_val2017'
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
####################


def analyze_image(image_name, segments_info, cat_info, annot_folder, model, hist):
    # Load png annotation
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    strings = image2strings(img_ann)
    catInf = pd.DataFrame(cat_info).T
    cat = pd.DataFrame(segments_info)
    merge = pd.concat([cat.set_index('category_id'), catInf.set_index('id')], axis=1, join='inner').reset_index()
    object_ordering = []
    for string_ids, count in strings:
        id = np.unique(np.array(string_ids))
        result = merge[['id', 'name']].loc[merge['id'].isin(id)]
        e = dict(zip(result['id'].values, result['name'].values))
        a = sorted(e.items(), key=operator.itemgetter(1))
        ids = [i[0] for i in a]
        object_ordering.append((ids, []))

    positions = compute_string_positions(None, object_ordering)

    X_test = positions.items()
    for pair in list(X_test):
        featuresRow = get_features(img_ann, "", pair[0][0], pair[0][1], positions)
        subject = merge[['name']].loc[merge['id'] == pair[0][0], 'name'].values[0]
        reference = merge[['name']].loc[merge['id'] == pair[0][1], 'name'].values[0]
        prediction = model.predict([np.asarray(featuresRow[3:])])
        hist[subject, reference, prediction[0]] += 1
        #hist[subject, reference][prediction[0]] += 1
    return hist



def run_tasks(json_file, annot_folder, model):
    """
    Run tasks: analyze training annotations
    :param json_file: annotation file with classes for each segment
    :param annot_folder: folder with png annotations
    """
    # Load annotations
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        annot_dict = {}
        cat_dict = {}
        for img_ann in json_data['annotations']:
            annot_dict[img_ann['file_name']] = img_ann['segments_info']
        for img_ann in json_data['categories']:
            cat_dict[img_ann['id']] = img_ann
    # Get files to be analyzed
    files = sorted(listdir(annot_folder))

    # Init progress bar
    pbar = tqdm(total=len(files))

    def update(x):
        pbar.update()

    print("Number of images: %d" % len(files))
    print("Scheduling tasks...")
    pool = Pool(10)
    result = []
    hist = defaultdict()
    for img in files:
        hist = pool.apply_async(analyze_image, args=(img, annot_dict[img], cat_dict, annot_folder, model, hist), callback=update).get()
    pool.close()
    pool.join()

    pbar.close()
    print(hist)
    print("Done")

if __name__ == "__main__":
    start_time = datetime.now()
    classifier = [Nearest_Neighbors,
                  Linear_SVM,
                  RBF_SVM,
                  Decision_Tree,
                  Random_Forest,
                  Naive_Bayes]
    if use_validate_classifiers:
        #validate_classifiers_grid_search()
        validate_classifiers(output_path)
    elif use_build_final_model:
        build_final_model(fileModel_path, classifier)
    input_images = '../COCO/images/val2017/'
    loaded_model = pickle.load(open(fileModel_path, 'rb'))
    run_tasks(path_json_file, path_annot_folder, loaded_model)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

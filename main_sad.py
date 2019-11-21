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
import networkx as nx

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


def analyze_image(image_name, segments_info, cat_info, annot_folder, model):
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    catInf = pd.DataFrame(cat_info).T
    segInfoDf = pd.DataFrame(segments_info)
    merge = pd.concat([segInfoDf.set_index('category_id'), catInf.set_index('id')], axis=1, join='inner').reset_index()
    object_ordering = []
    hist = {}

    result = merge[['id', 'name']].loc[merge['id'].isin(segInfoDf['id'].values)]
    pair = dict(zip(result['id'].values, result['name'].values))
    pairSorted = sorted(pair.items(), key=operator.itemgetter(1))
    object_ordering.append(([i[0] for i in pairSorted], []))
    positions = compute_string_positions(None, object_ordering)

    X_test = positions.items()
    for pairObj in list(X_test):
        featuresRow = get_features(img_ann, "", pairObj[0][0], pairObj[0][1], positions)
        subject = merge[['name', 'id']].loc[merge['id'] == pairObj[0][0], ('name', 'id')].values[0]
        reference = merge[['name', 'id']].loc[merge['id'] == pairObj[0][1], ('name', 'id')].values[0]
        prediction = model.predict([np.asarray(featuresRow[3:])])
        hist[tuple(subject), tuple(reference)] = prediction[0]

    g = nx.Graph()
    for p in pairSorted:
        g.add_node(p[0], object=p[1])

    for key, value in hist.items():
        g.add_edge(key[0][1], key[1][1], position=value)

    return g


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

    for img in files:
        result.append(pool.apply_async(analyze_image, args=(img, annot_dict[img], cat_dict, annot_folder, model),
                                       callback=update))
    pool.close()
    pool.join()

    dataset = []
    for img in result:
        if img.get() is not None:
            dataset.append(img.get())

    g1_json = nx.node_link_data(dataset[0])
    g2_json = nx.node_link_data(dataset[1])
    graph_list = [g1_json, g2_json]
    #graphs_string = json.dumps(graph_list)

    pbar.close()
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
        # validate_classifiers_grid_search()
        validate_classifiers(output_path)
    elif use_build_final_model:
        build_final_model(fileModel_path, classifier)
    input_images = '../COCO/images/val2017/'
    loaded_model = pickle.load(open(fileModel_path, 'rb'))
    run_tasks(path_json_file, path_annot_folder, loaded_model)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

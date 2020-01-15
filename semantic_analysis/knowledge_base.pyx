import pyximport
pyximport.install(language_level=3)

import json
import os
import pickle
import traceback
from multiprocessing.pool import Pool
from os import listdir

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

from config import train_graphs_json_path, kb_dir, kb_pairwise_json_path
from panopticapi.utils import load_png_annotation
from semantic_analysis.algorithms import image2strings, compute_string_positions, get_features
from semantic_analysis.gspan_mining.graph import nx_to_json


def generate_kb(fileModel_path, COCO_json_path, COCO_ann_dir):
    """
    Generate knowledge base: graphs and histograms
    - describe the image as a graph with object relationships (save graph in train_graphs_json_path)
    - collect frequent relationships between object classes (histograms of the KB in kb_pairwise_json_path)
    """
    loaded_model = pickle.load(open(fileModel_path, 'rb'))
    run_tasks(COCO_json_path, COCO_ann_dir, loaded_model)

def run_tasks(json_file, annot_folder, model):
    """
    Run tasks: analyze annotations
    Steps of this program (for each image):
    - analyze image annotations: run position classifier for each object pair
    - describe the image as a graph with object relationships (save graph in train_graphs_json_path)
    - collect frequent relationships between object classes (histograms of the KB in kb_pairwise_json_path)


    :param json_file: annotation file with classes for each segment
    :param annot_folder: folder with png annotations
    :param model: position classifier
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
    results = []

    # 1. Apply position classifier to all images, retrieve graphs
    for img in files:
        results.append(pool.apply_async(analyze_image, args=(img, annot_dict[img], cat_dict, annot_folder, model),
                                        callback=update))
    pool.close()
    pool.join()

    resultGraph = []
    resultHist = []
    # 2. Retrieve results from the tasks
    for img in results:
        if img.get() is not None:
            graph, hist = img.get()
            # Get graph description for this image
            resultGraph.append(nx_to_json(graph))
            # Get position histograms for this image
            resultHist.append(hist)

    # Save graphs
    with open(train_graphs_json_path, "w") as f:
        json.dump(resultGraph, f)

    histograms = {}

    # 3. Add up all histogram statistics extracted from the images
    for pair, hist in [(k, v) for img in resultHist for (k, v) in img.items()]:
        if pair not in histograms:
            # add histogram as it is if pair is not existing
            histograms[pair] = hist
        else:
            total_hist = histograms[pair]
            # update histograms if pair already existing
            for key in hist:
                if key in total_hist:
                    total_hist[key] += hist[key]
                else:
                    total_hist[key] = hist[key]
    # For each histogram compute support, entropy and relative frequencies
    for hist in histograms.values():
        sup = sum(hist.values())  # support: sum of all occurrences in the histogram
        ent = []
        for pos, count in hist.items():
            perc = count / sup
            hist[pos] = perc
            ent.append(perc)
        hist['sup'] = sup
        # Important: the dictionary inside hist may not contain all the different relative positions
        # E.g. {'side':0.5, 'side-up':0.5}
        # There are missing zeros like ... 'above':0, 'on':0,...
        # However these 0 terms does not influence entropy (because 0*log(0)=0)
        hist['entropy'] = entropy(ent, base=2)

    if not os.path.isdir(kb_dir):
        os.makedirs(kb_dir)
    # Save histograms
    with open(kb_pairwise_json_path, "w") as f:
        json.dump({str(k): v for k, v in histograms.items()}, f)

    pbar.close()
    print("Done")

def analyze_image(image_name, segments_info, cat_info, annot_folder, model):
    """
    Analyze image with relative-position classifier
    :param image_name: name of the image (no extension)
    :param segments_info: json with segment class information
    :param cat_info: COCO category information
    :param annot_folder: path to annotations
    :param model: relative-position classifier
    :return: (g, hist) where g is the graph representing this image (object relationships)
             and hist are the histogram statistics (frequent class relationships)
    """

    try:
        catInf = pd.DataFrame(cat_info).T
        segInfoDf = pd.DataFrame(segments_info)

        # This image has no objects
        if len(segInfoDf) == 0:
            return None

        merge = pd.concat([segInfoDf.set_index('category_id'), catInf.set_index('id')], axis=1,
                          join='inner').set_index('id')

        result = merge['name'].sort_values()
        img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
        strings = image2strings(img_ann)
        object_ordering = result.index.tolist()
        # Compute string description of the image
        positions = compute_string_positions(strings, object_ordering)
        g = nx.Graph()
        hist = {}
        for id, name in result.iteritems():
            g.add_node(id, label=name)
        for (s, r), pos in list(positions.items()):
            # Get features for this object pair (s, r)
            featuresRow = get_features(img_ann, "", s, r, positions)
            subject = result[s]
            reference = result.loc[r]
            # Apply classifier for this object pair (s, r)
            prediction = model.predict([np.asarray(featuresRow[3:])])[0]
            # Fill the graph describing this image
            g.add_edge(s, r, pos=prediction)
            if (subject, reference) not in hist.keys():
                hist[subject, reference] = {prediction: 1}
            else:
                hist[subject, reference].update({prediction: 0})
                hist[subject, reference][prediction] += 1
        return g, hist
    except Exception as e:
        print('Caught exception in analyze_image:')
        traceback.print_exc()
        return None

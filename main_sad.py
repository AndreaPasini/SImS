"""
 Author: Andrea Pasini
 This file provides the code for Semantic Anomaly Detection (SAD) on COCO.

"""

from datetime import datetime
import os
from shutil import rmtree
import random
import pandas as pd
from os import listdir
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool
from panopticapi.utils import load_png_annotation
from image_analysis.ImageProcessing import getImage
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn import tree

import pyximport
pyximport.install(language_level=3)
from semantic_analysis.algorithms import image2strings, compute_string_positions

def is_on(vector, first_i, first_j):
    return first_i + 1 == first_j

def analyze_image(image_name, segments_info, image_id, annot_folder):
    # Load png annotation
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    strings = image2strings(img_ann)
    positions = compute_string_positions(strings)
    rand = random.choice(list(positions.items()))
    getImage(image_name, img_ann, rand)
    featuresRow = [image_id, rand[0][0], rand[0][1]] + extractDict(rand[1])


    print("Done")

    return featuresRow

def extractDict(d):
    features = []
    for k, v in d.items():
        features.append(v)
    return features

def inizializePath():
    if not os.path.exists('../COCO/positionDataset/training'):
        os.mkdir('../COCO/positionDataset/training')
    else:
        rmtree('../COCO/positionDataset/training')
        os.mkdir('../COCO/positionDataset/training')

def run_tasks(json_file, annot_folder):
    """
    Run tasks: analyze training annotations
    :param json_file: annotation file with classes for each segment
    :param annot_folder: folder with png annotations
    """
    # Load annotations
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        annot_dict = {}
        id_dict = {}
        for img_ann in json_data['annotations']:
            annot_dict[img_ann['file_name']] = img_ann['segments_info']
        for img_ann in json_data['annotations']:
            id_dict[img_ann['file_name']] = img_ann['image_id']
    # Get files to be analyzed
    files = sorted(listdir(annot_folder))

    # Init progress bar
    pbar = tqdm(total=len(files))

    def update(x):
        pbar.update()

    print("Number of images: %d" % len(files))
    print("Scheduling tasks...")
    pool = Pool(num_processes)

    result = []
    for img in files:
        result.append(
            pool.apply_async(analyze_image, args=(img, annot_dict[img], id_dict[img], annot_folder), callback=update))
    pool.close()
    pool.join()

    createCSV(result)

    pbar.close()
    print("Done")

def createCSV(result):
    datasetFeatures = []
    for img in result:
        datasetFeatures.append(img.get())

    df = pd.DataFrame(datasetFeatures, columns=['image_id', 'Subject', 'Reference', 'i on j', 'j on i', 'i above j',
                                                'j above i', 'i around j', 'j around i', 'other'])
    df.to_csv('../COCO/positionDataset/training/Features.csv', sep=';', index=None, header=True)
    print("Create Features.csv")

    imageDetails = []
    for array in datasetFeatures:
        imageDetails.append(array[:3] + [""])

    df = pd.DataFrame(imageDetails, columns=['image_id', 'Subject', 'Reference', 'Position'])
    df.to_csv('../COCO/positionDataset/training/ImageDetails.csv', sep=';', index=None, header=True)
    print("Create ImageDetails.csv")

def example2():
    start_time = datetime.now()

    lista = []
    # TODO: use training images, instead of validation
    for i in range(0, 100000):
        a = 100 * 100
    lista.append(a)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

def classificator():
    data = pd.read_csv('../COCO/Features.csv', sep=';')
    data_img = pd.read_csv('../COCO/ImageDetails.csv', sep=';')
    svc = tree.DecisionTreeClassifier()

    X = np.array(data[["Subject", "Reference", "i on j",	"j on i",	"i above j", 	"j above i", 	"i around j", 	"j around i", 	"other"]])
    y = np.array(data_img["Position"])

    loo = LeaveOneOut()
    y_pred = cross_val_predict(svc, X, y, cv=loo)

    precision, recall, f1, s = precision_recall_fscore_support(y, y_pred)
    column_names = np.unique(y)

    matrix_precision = np.reshape(precision, (1, precision.size))
    df_precision = pd.DataFrame(matrix_precision, columns=column_names, index=['Precision'])
    matrix_recall = np.reshape(recall, (1, recall.size))
    df_recall = pd.DataFrame(matrix_recall, columns=column_names, index=['Recall   '])

    print(df_precision)
    print(df_recall)

    conf_mat = confusion_matrix(y, y_pred)
    conf_mat_df = pd.DataFrame(conf_mat)
    conf_mat_df.index.name = 'Actual'
    conf_mat_df.columns.name = 'Predicted'
    print("      ")
    print(conf_mat_df)

if __name__ == "__main__":
    start_time = datetime.now()
    chunck_size = 10  # number of images processed for each task
    num_processes = 10  # number of processes where scheduling tasks
    # TODO: use training images, instead of validation
    input_images = '../COCO/images/train2017/'
    inizializePath()
    #classificator()
    run_tasks('../COCO/annotations/panoptic_train2017.json', '../COCO/annotations/panoptic_train2017')
    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

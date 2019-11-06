"""
 Author: Andrea Pasini
 This file provides the code for Semantic Anomaly Detection (SAD) on COCO.

"""
from datetime import datetime
import os
import random
import pandas as pd
from os import listdir
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool
import seaborn as sns
from image_analysis.DatasetUtils import inizializePath
from panopticapi.utils import load_png_annotation
import pickle
import pyximport
from image_analysis.ImageProcessing import getImage
from semantic_analysis.algorithms import image2strings, compute_string_positions, getSideFeatures, getWidthSubject

from semantic_analysis.position_classifier import validate_classifiers, build_final_model

pyximport.install(language_level=3)


### CONFIGURATION ###
path = '../COCO/positionDataset/training'
pathImageDetail = path + '/ImageDetails.csv'
output_path = '../COCO/positionDataset/results/evaluation.txt'
result_path = '../COCO/positionDataset/results'
pathImageDetailBalanced = path + '/ImageDetailsBalance.csv'
pathFeaturesBalanced = path + '/FeaturesBalanced.csv'
pathFeaturesScatterplot = path + '/FeaturesScatterplot.csv'
pathFeatures = path + '/Features.csv'
groundPathImage = path + "/" + "groundTruth"
classifierPathImage = path + "/" + "classifier"
fileModel = '../COCO/output/finalized_model.clf'
path_json_file = '../COCO/annotations/panoptic_train2017.json'
path_annot_folder = '../COCO/annotations/panoptic_train2017'
input_images = '../COCO/images/train2017/'
n_features = 10  # number of images for each class
chunck_size = 10  # number of images processed for each task
num_processes = 10  # number of processes where scheduling tasks
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
use_create_dataset = False
create_balanced_dataset = False
use_validate_classifiers = False
use_build_final_model = True
####################


def is_on(vector, first_i, first_j):
    return first_i + 1 == first_j


def analyze_image(image_name, segments_info, image_id, annot_folder):
    # Load png annotation
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    strings = image2strings(img_ann)
    positions = compute_string_positions(strings)
    rand = random.choice(list(positions.items()))
    getImage(image_name, img_ann, rand)  # Save image with subject and reference
    subject = rand[0][0]
    reference = rand[0][1]
    widthSub = getWidthSubject(img_ann, subject)
    featuresRow = [image_id, subject, reference] + extractDict(rand[1], widthSub)
    featuresRow.extend(getSideFeatures(img_ann, subject, reference))
    return featuresRow


def extractDict(d, widthSub):
    features = []
    for k, v in d.items():
        features.append(v / widthSub)
    return features


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
    imageDf = pd.DataFrame()
    if os.path.isfile(pathFeatures):
        imageDf = pd.read_csv(pathFeatures, usecols=['image_id'], sep=';')
    pool = Pool(num_processes)
    result = []
    for img in files:
        if id_dict[img] not in imageDf.values:
            result.append(pool.apply_async(analyze_image, args=(img, annot_dict[img], id_dict[img], annot_folder),
                                           callback=update))
    pool.close()
    pool.join()

    createCSV(result)

    pbar.close()
    print("Done")


def createCSV(result):
    datasetFeatures = []
    for img in result:
        datasetFeatures.append(img.get())
    dfFeatures = setDfFeaturs(datasetFeatures)
    checkCSV(pathFeatures, dfFeatures)
    print("Create Features.csv")

    imageDetails = []
    for array in datasetFeatures:
        imageDetails.append(array[:3] + [""])
    dfImageDetails = setDfImageDetails(imageDetails)
    checkCSV(pathImageDetail, dfImageDetails)
    print("Create ImageDetails.csv")


def checkCSV(nameCSV, df):
    if not os.path.isfile(nameCSV):
        df.to_csv(nameCSV, sep=';', index=None, header=True)
    else:
        df.to_csv(nameCSV, sep=';', mode='a', index=None, header=False)


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


def getHistogram(data):
    hist = data
    hist.head()
    print(hist.shape)
    print(hist['Position'].unique())
    print(hist.groupby('Position').size())
    sns.countplot(hist['Position'], label="Count")

def useClassifier():
    loaded_model = pickle.load(open(fileModel, 'rb'))
    print(loaded_model)

def createBalancedDataset():
    if os.path.isfile(pathImageDetailBalanced):
        os.remove(pathImageDetailBalanced)
    if os.path.isfile(pathFeaturesBalanced):
        os.remove(pathFeaturesBalanced)

    dirList = [item for item in os.listdir(groundPathImage) if os.path.isdir(os.path.join(groundPathImage, item))]

    for elem in dirList:
        if elem == 'DOUBT':
            continue
        classPath = groundPathImage + "/" + elem
        dataImg = []
        dataFea = []
        imgId = []
        n_elem = len([name for name in os.listdir(classPath) if os.path.isfile(os.path.join(classPath, name))])
        dfImg = pd.read_csv(pathImageDetail, sep=';')
        for index, row in dfImg.iterrows():
            if row[3] == elem and len(dataImg) != n_features:
                dataImg, imgId = addRowBalancedDataset(dataImg, row, imgId)
        checkCSV(pathImageDetailBalanced, setDfImageDetails(dataImg))
        dfFea = pd.read_csv(pathFeatures, sep=';')
        for index, row in dfFea.iterrows():
            if int(row[0]) in imgId:
                dataFea.extend(np.array([row]))
        checkCSV(pathFeaturesBalanced, setDfFeaturs(dataFea))
    getHistogram(pd.read_csv(pathImageDetailBalanced, sep=';'))
    print("ok")


def addRowBalancedDataset(dataImg, row, imgId):
    dataImg.extend(np.array([row]))
    imgId.extend(np.array([row[0]]))
    return dataImg, imgId


def setDfImageDetails(data):
    return pd.DataFrame(data, columns=['image_id', 'Subject', 'Reference', 'Position'])


def setDfFeaturs(data):
    return pd.DataFrame(data, columns=['image_id', 'Subject', 'Reference', 'i on j', 'j on i', 'i above j',
                                       'j above i', 'i around j', 'j around i', 'other', 'deltaY1',
                                       'deltaY2', 'deltaX1', 'deltaX2'])


def update_labels_from_folder_division():
    print("Update Images")
    df = pd.read_csv(pathImageDetail, sep=';')
    dirList = [item for item in os.listdir(groundPathImage) if os.path.isdir(os.path.join(groundPathImage, item))]
    for elem in dirList:
        classPath = groundPathImage + "/" + elem
        for file in os.listdir(classPath):
            id = file.lstrip("0").rstrip(".png")
            row = df.query('image_id == '+id)
            originalFolder = row['Position'].values[0]
            if originalFolder == elem:
                continue
            else:
                index = row.index.values
                df.at[index[0], 'Position'] = elem
                df.to_csv(pathImageDetail, encoding='utf-8', index=False, sep=';')
                print("moved image "+file + " from "+ originalFolder + " to "+elem)
    print("Update Completed")


if __name__ == "__main__":
    start_time = datetime.now()
    classifier = [Nearest_Neighbors,
                  Linear_SVM,
                  RBF_SVM,
                  Decision_Tree,
                  Random_Forest,
                  Naive_Bayes]
    if use_create_dataset:
        inizializePath(path)
        run_tasks(path_json_file, path_annot_folder)
    elif create_balanced_dataset:
        createBalancedDataset()
    elif use_validate_classifiers:
        inizializePath(result_path)
        validate_classifiers(output_path)
    elif use_build_final_model:
        inizializePath(result_path)
        build_final_model(classifier)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

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
import easygui

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageTk
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from multiprocessing import Pool
import seaborn as sns
from panopticapi.utils import load_png_annotation
from image_analysis.ImageProcessing import getImage
from image_analysis.SetFeatures import setFeatures, getImageName
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
import pickle

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn import tree

import pyximport

pyximport.install(language_level=3)
from semantic_analysis.algorithms import image2strings, compute_string_positions, getSideFeatures, getWidthSubject

### CONFIGURATION ###
path = '../COCO/positionDataset/training'
pathImageDetail = path + '/ImageDetails.csv'
pathImageDetailBalanced = path + '/ImageDetailsBalance.csv'
pathFeaturesBalanced = path + '/FeaturesBalanced.csv'
pathFeatures = path + '/Features.csv'
fileModel = '../COCO/output/finalized_model.sav'
path_json_file = '../COCO/annotations/panoptic_train2017.json'
path_annot_folder = '../COCO/annotations/panoptic_train2017'
input_images = '../COCO/images/train2017/'
n_features = 8
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
set_ground_truth = False
set_balanced_dataset = False
select_best_classifier = False
build_classifier = False
use_classifier = False
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


def inizializePath():
    if not os.path.exists(path):
        os.mkdir(path)


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
        if (id_dict[img] not in imageDf.values):
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


def getConfusionMatrix(y, y_pred):
    conf_mat = confusion_matrix(y, y_pred)
    conf_mat_df = pd.DataFrame(conf_mat)
    conf_mat_df.index.name = 'Actual'
    conf_mat_df.columns.name = 'Predicted'
    print("      ")
    print(conf_mat_df)


def getAccuracy(y, y_pred, nameClf, dfAccuracy):
    precision, recall, f1, s = precision_recall_fscore_support(y, y_pred)
    column_names = np.unique(y)

    matrix_precision = np.reshape(precision, (1, precision.size))
    #df_precision = pd.DataFrame(matrix_precision, columns=column_names, index=['Precision'])

    matrix_recall = np.reshape(recall, (1, recall.size))
    #df_recall = pd.DataFrame(matrix_recall, columns=column_names, index=['Recall   '])

    matrix_f1 = np.reshape(f1, (1, f1.size))
    df_f1 = pd.DataFrame(matrix_f1, columns=column_names, index=[nameClf])

    dfAccuracy = dfAccuracy.append(df_f1)
    return dfAccuracy


def checkClassifier():
    classifier = [Nearest_Neighbors,
                  Linear_SVM,
                  RBF_SVM,
                  Decision_Tree,
                  Random_Forest,
                  Naive_Bayes]

    if not any(classifier):
        easygui.msgbox("You must choose a classifier!", title="Classifier")
    elif sum(classifier) == 1:
        index = int(" ".join(str(x) for x in np.argwhere(classifier)[0]))
        getClassifier(index)
    else:
        easygui.msgbox("You must choose only a classifier!", title="Classifier")


def getClassifier(index):
    data = pd.read_csv(pathFeaturesBalanced, sep=';')
    data_img = pd.read_csv(pathImageDetailBalanced, sep=';')

    X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
    y = np.array(data_img["Position"])
    dfAccuracy = pd.DataFrame()

    X = StandardScaler().fit_transform(X)
    classifier = LeaveOneOut()
    names = ["Nearest Neighbors",
             "Linear SVM",
             "RBF SVM",
             "Decision Tree",
             "Random Forest",
             "Naive Bayes"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0),
        GaussianNB()]
    try:

        if index is not None:
            y_pred = cross_val_predict(classifiers[index], X, y, cv=classifier)
            dfAccuracy = getAccuracy(y, y_pred, names[index], dfAccuracy)
            getConfusionMatrix(y, y_pred)
        else:
            for nameClf, clf in zip(names, classifiers):
                print(nameClf)
                y_pred = cross_val_predict(clf, X, y, cv=classifier)
                dfAccuracy = getAccuracy(y, y_pred, nameClf, dfAccuracy)
            dfAccuracy.plot.bar()
            dfAccuracy['mean'] = dfAccuracy.mean(axis=1)
        print(dfAccuracy.head().to_string())
        pickle.dump(y_pred, open(fileModel, 'wb'))
    except ValueError as e:
        print(e)

def useClassifier():
    loaded_model = pickle.load(open(fileModel, 'rb'))
    print(loaded_model)

def createFolderByClass():
    df = pd.read_csv(pathImageDetail, usecols=["image_id", "Position"], sep=';')
    for index, row in df.iterrows():
        imageSource = getImageName(row[0], path)
        if os.path.isfile(imageSource):
            classPath = path + "/" + row[1]
            imageDestination = getImageName(row[0], classPath)
            if not os.path.exists(classPath):
                os.mkdir(classPath)
            try:
                os.rename(imageSource, imageDestination)
            except FileNotFoundError as e:
                print(e)


def createBalancedDataset():
    os.remove(pathImageDetailBalanced)
    os.remove(pathFeaturesBalanced)
    dirlist = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]

    for elem in dirlist:
        classPath = path + "/" + elem
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


if __name__ == "__main__":
    start_time = datetime.now()
    if build_classifier:
        checkClassifier()
    elif use_create_dataset:
        inizializePath()
        run_tasks(path_json_file, path_annot_folder)
    elif set_ground_truth:
        setFeatures()
        #createFolderByClass()
    elif set_balanced_dataset:
        createBalancedDataset()
    elif select_best_classifier:
        getClassifier(None)
    elif use_classifier:
        useClassifier()

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

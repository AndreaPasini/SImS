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
from image_analysis.SetFeatures import setFeatures
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import  precision_recall_fscore_support
from sklearn.svm import SVC

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn import tree

import pyximport
pyximport.install(language_level=3)
from semantic_analysis.algorithms import image2strings, compute_string_positions, getSideFeatures, getWidthSubject

### CONFIGURATION ###
path = '../COCO/positionDataset/training'
use_classifier = True
use_create_folder = False
### CONFIGURATION ###




def is_on(vector, first_i, first_j):
    return first_i + 1 == first_j

def analyze_image(image_name, segments_info, image_id, annot_folder):
    # Load png annotation
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    strings = image2strings(img_ann)
    positions = compute_string_positions(strings)
    rand = random.choice(list(positions.items()))
    getImage(image_name, img_ann, rand)     # Save image with subject and reference
    subject = rand[0][0]
    reference = rand[0][1]
    widthSub = getWidthSubject(img_ann, subject)
    featuresRow = [image_id, subject, reference] + extractDict(rand[1], widthSub)
    featuresRow.extend(getSideFeatures(img_ann, subject, reference))
    return featuresRow

def extractDict(d, widthSub):
    features = []
    for k, v in d.items():
        features.append(v/widthSub)
    return features

def inizializePath():
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        rmtree(path)
        os.mkdir(path)

def setGroundTruth():
    setFeatures()


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
    if os.path.isfile(path + '/Features.csv'):
        imageDf = pd.read_csv(path+"/Features.csv", usecols=['image_id'], sep=';')
    pool = Pool(num_processes)
    result = []
    for img in files:
        if (id_dict[img] not in imageDf.values):
            result.append(pool.apply_async(analyze_image, args=(img, annot_dict[img], id_dict[img], annot_folder), callback=update))
    pool.close()
    pool.join()

    createCSV(result)

    pbar.close()
    print("Done")

def createCSV(result):
    datasetFeatures = []
    for img in result:
        datasetFeatures.append(img.get())
    dfFeatures = pd.DataFrame(datasetFeatures, columns=['image_id', 'Subject', 'Reference', 'i on j', 'j on i', 'i above j',
                                                'j above i', 'i around j', 'j around i', 'other', 'deltaY1',
                                                'deltaY2', 'deltaX1', 'deltaX2'])
    checkCSV(path + '/Features.csv', dfFeatures)
    print("Create Features.csv")

    imageDetails = []
    for array in datasetFeatures:
        imageDetails.append(array[:3] + [""])
    dfImageDetails = pd.DataFrame(imageDetails, columns=['image_id', 'Subject', 'Reference', 'Position'])
    checkCSV(path + '/ImageDetails.csv', dfImageDetails)
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

def getAccuracy(y, y_pred):
    precision, recall, f1, s = precision_recall_fscore_support(y, y_pred)
    column_names = np.unique(y)

    matrix_precision = np.reshape(precision, (1, precision.size))
    df_precision = pd.DataFrame(matrix_precision, columns=column_names, index=['Precision'])

    matrix_recall = np.reshape(recall, (1, recall.size))
    df_recall = pd.DataFrame(matrix_recall, columns=column_names, index=['Recall   '])

    matrix_f1 = np.reshape(f1, (1, f1.size))
    df_f1 = pd.DataFrame(matrix_f1, columns=column_names, index=['F1        '])

    #print(df_precision.head().to_string())
    #print(df_recall.head().to_string())
    print(df_f1.head().to_string())
    print("     ")

def getClassifier():
    data = pd.read_csv(path + '/Features.csv', sep=';')
    data_img = pd.read_csv(path + '/ImageDetails.csv', sep=';')

    getHistogram(data_img)

    X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
    y = np.array(data_img["Position"])

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)        #TODO: la cross-validation divide gia' X in training e test set

    classifier = LeaveOneOut()

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net",
             "Naive Bayes"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0),
        MLPClassifier(alpha=1, max_iter=1000),
        GaussianNB()]

    for nameClf, clf in zip(names, classifiers):
        print(nameClf)
        clf.fit(X_train, y_train)                                   #TODO: la fit non serve perche' cross_val_predict addestra gia' 1 modello per ogni partizione della cross-validation (vedi slides scikit-learn)
        y_pred = cross_val_predict(clf, X, y, cv=classifier)
        getAccuracy(y, y_pred)

if __name__ == "__main__":
    start_time = datetime.now()
    chunck_size = 10  # number of images processed for each task
    num_processes = 10  # number of processes where scheduling tasks
    # TODO: use training images, instead of validation
    input_images = '../COCO/images/train2017/'
    if use_classifier:
        getClassifier()
    elif use_create_folder:
        #inizializePath()
        #run_tasks('../COCO/annotations/panoptic_train2017.json', '../COCO/annotations/panoptic_train2017')
        setGroundTruth()
    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

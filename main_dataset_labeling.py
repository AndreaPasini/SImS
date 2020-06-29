import shutil
from datetime import datetime
import tkinter as tk
import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
from os import listdir
from multiprocessing import Pool
from tkinter.ttk import Combobox
from PIL import Image, ImageTk
from tqdm import tqdm
import random

from config import position_dataset_dir, COCO_train_json_path, COCO_ann_train_dir, position_labels_csv_path
from sims.scene_graphs.image_processing import getImageName, getImage
from panopticapi.utils import load_png_annotation

import pyximport
pyximport.install(language_level=3)
from sims.scene_graphs.feature_extraction import image2strings, compute_string_positions, get_features

from sklearn.utils import shuffle
from collections import defaultdict

### CONFIGURATION ###
position_train_dir = os.path.join(position_dataset_dir, 'training')
groundPathImage = os.path.join(position_train_dir, 'groundTruth')

# List of labels to be recognized
position_labels = tuple(s.strip() for s in open(position_labels_csv_path).readlines())

# Feature matrix for object position computation
pathFeatures = os.path.join(position_train_dir, 'Features.csv')
pathFeaturesBalanced = os.path.join(position_train_dir, 'FeaturesBalanced.csv')
# Ground truth labels for object position computation
pathGroundTruth = os.path.join(position_train_dir, 'GroundTruth.csv')
pathGroundTruthBalanced = os.path.join(position_train_dir, 'GroundTruthBalanced.csv')

num_processes = 10  # number of processes where scheduling tasks


#####################

############ ACTION ###########
# This file allows creating a dataset for object positions.
# Each sample of the dataset consists of an image for which an object pair is considered.
# The label associated to a sample corresponds to the relative position of the object pair.

# 1. Add new unlabeled samples to the dataset. These new samples must be labeled with 'LABELING_GUI' functionality.
num_new_images = 600    # number of new images to add to the dataset
filterSideImages = True # true if you want to obtain (more likely) side images to be labeled (useful because typically side images are rare in the dataset)
#action = 'ADD_NEW_IMAGES'
# 2. Use this Graphic Interface to manually set ground truth labels to unlabeled samples
#action = 'LABELING_GUI'
# 3. Use this method to update ground truth labels of dataset samples, according to their position into folders.
# Before using this method you have to move the sample images to the folder with the correct label (e.g. move samples
# with "on" label to the "on" folder)
#action = 'LABELS_FROM_FOLDERS'
# 4. Update features matrix for existing samples.
# Given samples and labels in the ground truth file, re-create the features matrix file by computing again features
#action = 'RECOMPUTE_FEATURES'
# 5. Use this method when the labeling process is complete. This will create a balanced dataset with the specified number of images for each class
num_img_by_class = 60  # number of images for each class
action = 'CREATE_BALANCED_DATASET'

###############################

def set_ground_truth_header(data):
    return pd.DataFrame(data, columns=['image_id', 'Subject', 'Reference', 'Position'])


def set_features_matrix_header(data):
    return pd.DataFrame(data, columns=['image_id', 'Subject', 'Reference', 'i on j', 'j on i', 'i above j',
                                       'j above i', 'i around j', 'j around i', 'other', 'deltaY1',
                                       'deltaY2', 'deltaX1', 'deltaX2'])

def update_csv(nameCSV, df):
    """ Write dataframe to file, append if already existing. """
    if not os.path.isfile(nameCSV):
        df.to_csv(nameCSV, sep=';', index=None, header=True)
    else:
        df.to_csv(nameCSV, sep=';', mode='a', index=None, header=False)

def createCSV(result, filterSideImages=False):
    """
    Given the list of results from Pool processes, extract results and save image features.
    dfFeatures contains the feature extraction results for computing relative object positions inside images.
    dfImageDetails contains the ground truth labels
    """
    datasetFeatures = []
    for img in result:
        if img.get() is not None:
            datasetFeatures.append(img.get())
    dfFeatures = set_features_matrix_header(datasetFeatures)

    groundTruth = []
    for array in datasetFeatures:
        groundTruth.append(array[:3] + [""])
    dfGroundTruth = set_ground_truth_header(groundTruth)

    if filterSideImages:
        overlapped_df = dfFeatures[['i on j','j on i', 'i above j', 'j above i', 'i around j', 'j around i']]
        overlapped_df = overlapped_df.sum(axis=1)
        mask_side = overlapped_df<=0.1
        dfFeatures = dfFeatures[mask_side]
        dfGroundTruth = dfGroundTruth[mask_side]

    update_csv(pathFeatures, dfFeatures)
    print("Create Features.csv")


    update_csv(pathGroundTruth, dfGroundTruth)
    print("Create ImageDetails.csv")


def analyze_image(image_name, image_id, annot_folder):
    # Load png annotation
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    strings = image2strings(img_ann)
    positions = compute_string_positions(strings)
    if not positions:
        return
    rand = random.choice(list(positions.items()))

    getImage(image_name, img_ann, rand)  # Save image with subject and reference
    subject = rand[0][0]
    reference = rand[0][1]
    featuresRow = get_features(img_ann, image_id, subject, reference, positions)
    return featuresRow

def recompute_features():
    """
    Update features matrix for existing samples.
    Given samples and labels in the ground truth file, re-create the features matrix file by computing again features
    """

    print("Recomputing features of labeled images...")
    df = pd.read_csv(pathGroundTruth, sep=';')
    #df_features = pd.read_csv(pathFeatures, sep=';')
    features_list = []
    for i,row in df.iterrows():
        image_id = row['image_id']
        # Load png annotation
        image_name = f"{image_id:012d}.png"
        img_ann = load_png_annotation(os.path.join(COCO_ann_train_dir, image_name))
        strings = image2strings(img_ann)
        positions = compute_string_positions(strings)
        subject = row['Subject']
        reference = row['Reference']
        features_list.append(get_features(img_ann, image_id, subject, reference, positions))

    features_df = set_features_matrix_header(features_list)
    features_df.to_csv(pathFeatures, encoding='utf-8', index=False, sep=';')


def add_new_images(json_file, annot_folder):
    """
    Analyze training annotations and add new images
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
    files = shuffle(sorted(listdir(annot_folder)))

    # Init progress bar
    pbar = tqdm(total=num_new_images)

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
            if len(result) == num_new_images:
                pool.close()
                pool.join()
                break
            else:
                result.append(pool.apply_async(analyze_image, args=(img, id_dict[img], annot_folder),
                                               callback=update))

    # Save features matrix and prepare csv for annotations
    createCSV(result, filterSideImages)

    pbar.close()
    print("Done")



def callback(dfGroundTruth, dfFeatures, index, img, window, cb):
    """ React to confirmation of class label from the user GUI """
    val = cb.get()
    if val != 'Discard':
        dfGroundTruth.at[index, 'Position'] = val
        dfGroundTruth.to_csv(pathGroundTruth, encoding='utf-8', index=False, sep=';')
    else:
        dfGroundTruth.drop(index, inplace=True)
        dfFeatures.drop(index, inplace=True)
        dfGroundTruth.to_csv(pathGroundTruth, encoding='utf-8', index=False, sep=';')
        dfFeatures.to_csv(pathFeatures, encoding='utf-8', index=False, sep=';')
    img.config(image='')
    cb.set('')
    window.quit()

def labeling_gui():
    window = tk.Tk()
    cb = Combobox(window, values=position_labels+('Discard',))
    try:
        if os.path.isfile(pathGroundTruth):
            df = pd.read_csv(pathGroundTruth, sep=';')
            df = df.replace(np.nan, '', regex=True)
            dfFeatures = pd.read_csv(pathFeatures, sep=';')
            dfSelected = df[df.iloc[:, 3] == ''].copy()
            for index, row in dfSelected.iterrows():
                image = getImageName(row[0], position_train_dir)
                img = Image.open(image)
                render = ImageTk.PhotoImage(img)
                img = tk.Label(image=render)
                img.image = render
                img.place(x=0, y=0)
                label = tk.Label(window, text='Choose the correct label: ')
                label.place(x=730, y=160)
                cb.place(x=730, y=180)
                cb.current(len(position_labels))
                button = tk.Button(window, text='Add', fg='blue',
                                   command=lambda: callback(df, dfFeatures, index, img, window, cb))
                button.config(width=20)
                button.config(width=20)
                button.place(x=730, y=230)
                window.title(image)
                window.geometry("1000x800+10+10")
                window.mainloop()
    except FileNotFoundError as e:
        print(image + ' Not found')
    except RuntimeError as e:
        print(e)
    except AttributeError as e:
        print(e)

def update_labels_from_folder_division():
    print("Update Images")
    df = pd.read_csv(pathGroundTruth, sep=';')
    df_features = pd.read_csv(pathFeatures, sep=';')
    dirList = [item for item in os.listdir(groundPathImage) if os.path.isdir(os.path.join(groundPathImage, item))]

    for elem in dirList:
        classPath = groundPathImage + "/" + elem
        for file in os.listdir(classPath):
            id = file.lstrip("0").rstrip(".png")
            row = df.query('image_id=='+id)
            if row.shape[0]==0:
                os.remove(classPath+"/"+file)
                continue
            originalFolder = row['Position'].values[0]
            if originalFolder == elem:
                continue
            elif elem != 'discard':
                index = row.index.values[0]
                df.at[index, 'Position'] = elem
                print("moved image " + file + " from " + originalFolder + " to " + elem)
            else:
                os.remove(classPath + "/" + file)
                index = row.index.values
                df.drop(index[0], inplace=True)
                df_features.drop(index[0], inplace=True)
                print("removed image " + file + " from " + str(originalFolder))
    df.to_csv(pathGroundTruth, encoding='utf-8', index=False, sep=';')
    df_features.to_csv(pathFeatures, encoding='utf-8', index=False, sep=';')
    print("Update Completed")


def moveImagesToLabelFolders(column):
    """
    Read generated ground truth annotations and divide images into folders according to their labels.
    This operation is useful for visualizing if labeling is correct
    """
    if column == 'Position':
        pathImage = groundPathImage
        file = pathGroundTruth
    df = pd.read_csv(file, usecols=["image_id", column], sep=';')
    for index, row in df.iterrows():
        imageSource = getImageName(row[0], position_train_dir)
        if os.path.isfile(imageSource):
            if str(row[1]) == 'nan':
                continue
            else:
                classPath = pathImage + "/" + str(row[1])
                imageDestination = getImageName(row[0], classPath)
                if not os.path.exists(pathImage):
                    os.mkdir(pathImage)
                if not os.path.exists(classPath):
                    os.mkdir(classPath)
                try:
                    if not os.path.isfile(imageDestination):
                        shutil.move(imageSource, imageDestination)
                except FileNotFoundError as e:
                    print(e)



def addRowBalancedDataset(dataImg, row, imgId):
    dataImg.extend(np.array([row]))
    imgId.extend(np.array([row[0]]))
    return dataImg, imgId

def getHistogram(data):
    hist = data
    hist.head()
    print(hist.shape)
    print(hist['Position'].unique())
    print(hist.groupby('Position').size())
    sns.countplot(hist['Position'], label="Count")



def createBalancedDataset(num_img_by_class):
    """ Generate balanced dataset from input labeling. """
    if os.path.isfile(pathGroundTruthBalanced):
        os.remove(pathGroundTruthBalanced)
    if os.path.isfile(pathFeaturesBalanced):
        os.remove(pathFeaturesBalanced)


    dfImg = pd.read_csv(pathGroundTruth, sep=';')
    dfFea = pd.read_csv(pathFeatures, sep=';')
    listImg = []
    listFea = []
    for elem in position_labels:
        dataImg = dfImg[dfImg['Position']==elem][:num_img_by_class]
        listImg.append(dataImg)
        listFea.append(dfFea.loc[dataImg.index])
    dfImg = pd.concat(listImg)
    dfFea = pd.concat(listFea)

    dfImg = shuffle(dfImg)
    dfFea = dfFea.loc[dfImg.index]
    update_csv(pathGroundTruthBalanced, set_ground_truth_header(dfImg))
    update_csv(pathFeaturesBalanced, set_features_matrix_header(dfFea))



def inizializePath(path):
    if not os.path.exists(path):
        os.mkdir(path)

def print_cardinality():
    dirList = [item for item in os.listdir(groundPathImage) if os.path.isdir(os.path.join(groundPathImage, item))]
    counts = defaultdict(int)
    for elem in dirList:
        classPath = groundPathImage + "/" + elem
        for file in os.listdir(classPath):
            counts[elem] += 1
    print("Dataset cardinality:")
    for k, v in counts.items():
        print(f"{k}:{v}")
    print("\n")

if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time.strftime("Start date: %Y-%m-%d %H:%M:%S"))


    if action == 'ADD_NEW_IMAGES':
        add_new_images(COCO_train_json_path, COCO_ann_train_dir)
    elif action == 'LABELS_FROM_FOLDERS':
        update_labels_from_folder_division()
    elif action == 'LABELING_GUI':
        print_cardinality()
        labeling_gui()
        moveImagesToLabelFolders("Position")
    elif action == 'RECOMPUTE_FEATURES':
        recompute_features()
    elif action == 'CREATE_BALANCED_DATASET':
        createBalancedDataset(num_img_by_class)


    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

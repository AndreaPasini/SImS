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
from image_analysis.ImageProcessing import getImageName, getImage
from panopticapi.utils import load_png_annotation
from semantic_analysis.algorithms import image2strings, compute_string_positions, getSideFeatures, getWidthSubject

### CONFIGURATION ###
path = '../COCO/positionDataset/training'
pathImageDetail = path + '/ImageDetails.csv'
groundPathImage = path + "/" + "groundTruth"
path = '../COCO/positionDataset/training'
path_json_file = '../COCO/annotations/panoptic_train2017.json'
path_annot_folder = '../COCO/annotations/panoptic_train2017'
pathImageDetailBalanced = path + '/ImageDetailsBalance.csv'
pathFeaturesBalanced = path + '/FeaturesBalanced.csv'
pathFeatures = path + '/Features.csv'
pathImageDetail = path + '/ImageDetails.csv'
result_path = '../COCO/positionDataset/results'
data = ("on", "hanging", "above", "below", "inside", "around", "side", "side-up", "side-down")
Positions = pd.Series([])
num_processes = 10  # number of processes where scheduling tasks
n_features = 10  # number of images for each class
chunck_size = 10  # number of images processed for each task
images_load = 10  # number of images to load
#####################

############ ACTION ###########
action = 'LABELS_FROM_FOLDERS'
#action = 'LABELING_GUI'
#action = 'ADD_NEW_IMAGES'
###############################


def update_labels_from_folder_division():
    print("Update Images")
    df = pd.read_csv(pathImageDetail, sep=';')
    dirList = [item for item in os.listdir(groundPathImage) if os.path.isdir(os.path.join(groundPathImage, item))]
    for elem in dirList:
        classPath = groundPathImage + "/" + elem
        for file in os.listdir(classPath):
            id = file.lstrip("0").rstrip(".png")
            row = df.query('image_id == ' + id)
            originalFolder = row['Position'].values[0]
            if originalFolder == elem:
                continue
            else:
                index = row.index.values
                df.at[index[0], 'Position'] = elem
                print("moved image " + file + " from " + str(originalFolder) + " to " + elem)
    df.to_csv(pathImageDetail, encoding='utf-8', index=False, sep=';')
    print("Update Completed")


def labeling_gui():
    window = tk.Tk()
    cb = Combobox(window, values=data)
    try:
        if os.path.isfile(pathImageDetail):
            df = pd.read_csv(pathImageDetail, sep=';')
            df = df.replace(np.nan, '', regex=True)
            for index, row in df.iterrows():
                if (row[3] == ''):
                    image = getImageName(row[0], path)
                    img = Image.open(image)
                    render = ImageTk.PhotoImage(img)
                    img = tk.Label(image=render)
                    img.image = render
                    img.place(x=0, y=0)
                    label = tk.Label(window, text='Choose the correct label: ')
                    label.place(x=730, y=160)
                    cb.place(x=730, y=180)
                    button = tk.Button(window, text='Add', fg='blue',
                                       command=lambda: callback(df, index, img, window, cb))
                    button.config(width=20)
                    button.config(width=20)
                    button.place(x=730, y=230)
                    window.title(image)
                    window.geometry("1000x500+10+10")
                    window.mainloop()
    except FileNotFoundError as e:
        print(image + ' Not found')
    except RuntimeError as e:
        print(e)
    except AttributeError as e:
        print(e)


def callback(df, index, img, window, cb):
    val = cb.get()
    Positions[index] = val
    df.at[index, 'Position'] = val
    df.to_csv(pathImageDetail, encoding='utf-8', index=False, sep=';')
    img.config(image='')
    cb.set('')
    window.quit()


def add_new_images():
    run_tasks(path_json_file, path_annot_folder)


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


def analyze_image(image_name, segments_info, image_id, annot_folder):
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
    widthSub = getWidthSubject(img_ann, subject)
    featuresRow = [image_id, subject, reference] + extractDict(rand[1], widthSub)
    featuresRow.extend(getSideFeatures(img_ann, subject, reference))
    return featuresRow


def createCSV(result):
    datasetFeatures = []
    for img in result:
        if img.get() is not None:
            datasetFeatures.append(img.get())
    dfFeatures = setDfFeatures(datasetFeatures)
    checkCSV(pathFeatures, dfFeatures)
    print("Create Features.csv")

    imageDetails = []
    for array in datasetFeatures:
        imageDetails.append(array[:3] + [""])
    dfImageDetails = setDfImageDetails(imageDetails)
    checkCSV(pathImageDetail, dfImageDetails)
    print("Create ImageDetails.csv")


def setDfImageDetails(data):
    return pd.DataFrame(data, columns=['image_id', 'Subject', 'Reference', 'Position'])


def setDfFeatures(data):
    return pd.DataFrame(data, columns=['image_id', 'Subject', 'Reference', 'i on j', 'j on i', 'i above j',
                                       'j above i', 'i around j', 'j around i', 'other', 'deltaY1',
                                       'deltaY2', 'deltaX1', 'deltaX2'])


def extractDict(d, widthSub):
    features = []
    for k, v in d.items():
        features.append(v / widthSub)
    return features


def checkCSV(nameCSV, df):
    if not os.path.isfile(nameCSV):
        df.to_csv(nameCSV, sep=';', index=None, header=True)
    else:
        df.to_csv(nameCSV, sep=';', mode='a', index=None, header=False)


def addRowBalancedDataset(dataImg, row, imgId):
    dataImg.extend(np.array([row]))
    imgId.extend(np.array([row[0]]))
    return dataImg, imgId


def update_labels_from_folder_division():
    print("Update Images")
    df = pd.read_csv(pathImageDetail, sep=';')
    dirList = [item for item in os.listdir(groundPathImage) if os.path.isdir(os.path.join(groundPathImage, item))]
    for elem in dirList:
        classPath = groundPathImage + "/" + elem
        for file in os.listdir(classPath):
            id = file.lstrip("0").rstrip(".png")
            row = df.query('image_id == ' + id)
            originalFolder = row['Position'].values[0]
            if originalFolder == elem:
                continue
            else:
                index = row.index.values
                df.at[index[0], 'Position'] = elem
                df.to_csv(pathImageDetail, encoding='utf-8', index=False, sep=';')
                print("moved image " + file + " from " + originalFolder + " to " + elem)
    print("Update Completed")


def getHistogram(data):
    hist = data
    hist.head()
    print(hist.shape)
    print(hist['Position'].unique())
    print(hist.groupby('Position').size())
    sns.countplot(hist['Position'], label="Count")


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
        checkCSV(pathFeaturesBalanced, setDfFeatures(dataFea))
    getHistogram(pd.read_csv(pathImageDetailBalanced, sep=';'))
    print("ok")


def createFolderByClass(column):
    if column == 'Position':
        pathImage = groundPathImage
        file = pathImageDetail
    df = pd.read_csv(file, usecols=["image_id", column], sep=';')
    for index, row in df.iterrows():
        imageSource = getImageName(row[0], path)
        if os.path.isfile(imageSource):
            classPath = pathImage + "/" + str(row[1])
            imageDestination = getImageName(row[0], classPath)
            if not os.path.exists(pathImage):
                os.mkdir(pathImage)
            if not os.path.exists(classPath):
                os.mkdir(classPath)
            try:
                shutil.copy(imageSource, imageDestination)
            except FileNotFoundError as e:
                print(e)


def inizializePath(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time.strftime("Start date: %Y-%m-%d %H:%M:%S"))

    if action == 'ADD_NEW_IMAGES':
        inizializePath(path)
        add_new_images()
    elif action == 'LABELS_FROM_FOLDERS':
        update_labels_from_folder_division()
    elif action == 'LABELING_GUI':
        labeling_gui()
        createFolderByClass("Position")

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

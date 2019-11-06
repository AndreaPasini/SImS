import os
import pandas as pd
from image_analysis.ImageProcessing import getImageName
import shutil

### CONFIGURATION ###
path = '../COCO/positionDataset/training'
pathImageDetail = path + '/ImageDetails.csv'
pathImageDetailBalanced = path + '/ImageDetailsBalance.csv'
groundPathImage = path + "/" + "groundTruth"
classifierPathImage = path + "/" + "classifier"


def createFolderByClass(column):
    if column == 'Position':
        pathImage = groundPathImage
        file = pathImageDetail
    else:
        pathImage = classifierPathImage
        file = pathImageDetailBalanced
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

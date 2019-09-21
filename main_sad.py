"""
 Author: Andrea Pasini
 This file provides the code for Semantic Anomaly Detection (SAD) on COCO.

"""

from datetime import datetime
import os
from shutil import rmtree
import random
import pandas as pd
from itertools import groupby
from os import listdir
import json
from tqdm import tqdm
from multiprocessing import Pool
import PIL.Image as Image
from panopticapi.utils import load_png_annotation
import numpy as np
import pyximport;

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

    img = Image.open('../COCO/annotations/panoptic_val2017' + '/' + image_name)
    subject = rand[0][0] #yellow
    reference = rand[0][1] #blue

    featuresRow = [image_id, subject, reference] + extractDict(rand[1])

    imgSub = changeColor(img, convertToRGB(subject), True)
    imgRef = changeColor(imgSub, convertToRGB(reference), False)
    imgRef.save('../COCO/positionDataset/training/' + image_name, 'PNG')

    print("Done")
    return featuresRow

def changeColor(img, rgbSub, firstImage):
    img = img.convert('RGBA')
    data = np.array(img)
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability
    subject_areas = (red == rgbSub[0]) & (green == rgbSub[1]) & (blue == rgbSub[2])
    data[..., :-1][subject_areas.T] = (128, 128, 0) if firstImage else (0, 0, 255)  # Transpose back needed
    img = Image.fromarray(data)
    return img

def extractDict(d):
    features = []
    for k, v in d.items():
        features.append(v)
    return features

def convertToRGB(decimalColor):
    b = decimalColor & 255
    g = (decimalColor >> 8) & 255
    r = (decimalColor >> 16) & 255
    return (b, g, r)

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
    datasetFeatures = []
    result = []
    for img in files:
        result.append(pool.apply_async(analyze_image, args=(img, annot_dict[img], id_dict[img], annot_folder), callback=update))
    pool.close()
    pool.join()

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

    pbar.close()

    print("Done")


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

if __name__ == "__main__":
    start_time = datetime.now()
    chunck_size = 10  # number of images processed for each task
    num_processes = 10  # number of processes where scheduling tasks
    # TODO: use training images, instead of validation
    input_images = '../COCO/images/val2017/'
    inizializePath()
    run_tasks('../COCO/annotations/panoptic_val2017.json', '../COCO/annotations/panoptic_val2017')
    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

"""
 Author: Andrea Pasini
 This file provides the code for Semantic Anomaly Detection (SAD) on COCO.

"""

from datetime import datetime
import os
from shutil import rmtree
import sys
import csv
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
    return first_i+1==first_j

def analyze_image(image_name, segments_info, image_id, annot_folder, df):
    #Load png annotation
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    strings = image2strings(img_ann)
    positions = compute_string_positions(strings)
    rand = random.choice(list(positions.keys()))

    img = Image.open('../COCO/annotations/panoptic_val2017'+'/'+image_name)

    subject = rand[0]
    reference = rand[1]

    rS, gS, bS  = convertToRGB(subject)
    rR, gR, bR  = convertToRGB(reference)

    img = img.convert('RGBA')
    dataS = np.array(img)
    red, green, blue, alpha = dataS.T  # Temporarily unpack the bands for readability
    # Replace white with yellow... (leaves alpha values alone...)
    subject_areas = (red == rS) & (green == gS) & (blue == bS)
    dataS[..., :-1][subject_areas.T] = (128, 128, 0)  # Transpose back needed
    imgS = Image.fromarray(dataS)

    img2 = imgS.convert('RGBA')
    dataR = np.array(img2)
    red, green, blue, alpha = dataR.T  # Temporarily unpack the bands for readability
    # Replace white with blue... (leaves alpha values alone...)
    reference_areas = (red == rR) & (green == gR) & (blue == bR)
    dataR[..., :-1][reference_areas.T] = (0, 0, 255)  # Transpose back needed
    imgR = Image.fromarray(dataR)

    imgR.save('Features/'+image_name, 'PNG')

    df = pd.read_csv("Features/ImageDetails.csv", sep=',', encoding="utf-8")
    df.loc[df.index.max() + 1] = [image_id, subject, reference, '']
    df.to_csv("Features/ImageDetails.csv", index=False)


def convertToRGB(decimalColor):
    b = decimalColor & 255
    g = (decimalColor >> 8) & 255
    r = (decimalColor >> 16) & 255
    return b, g, r

def inizializeCSV():
    if not os.path.exists('Features/'):
        os.mkdir('Features')
    else:
        rmtree('Features')
        os.mkdir('Features')
    data = {'image_id': [], 'Subject': [], 'Reference': [], 'Position': []}
    df = pd.DataFrame(data, columns=['image_id', 'Subject', 'Reference', 'Position'])
    df.to_csv('Features/ImageDetails.csv', index=None, header=True)
    return df

def run_tasks(json_file, annot_folder, df):
    """
    Run tasks: analyze training annotations
    :param json_file: annotation file with classes for each segment
    :param annot_folder: folder with png annotations
    """
    #Load annotations
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        annot_dict = {}
        id_dict = {}
        for img_ann in json_data['annotations']:
            annot_dict[img_ann['file_name']]=img_ann['segments_info']
        for img_ann in json_data['annotations']:
            id_dict[img_ann['file_name']]=img_ann['image_id']
    #Get files to be analyzed
    files = sorted(listdir(annot_folder))

    #Init progress bar
    pbar = tqdm(total=len(files))
    def update(x):
        pbar.update()

    print("Number of images: %d" % len(files))
    print("Scheduling tasks...")

    pool = Pool(num_processes)
    for img in files:
        pool.apply_async(analyze_image, args=(img, annot_dict[img], id_dict[img], annot_folder, df), callback=update)
    pool.close()
    pool.join()
    pbar.close()

    print("Done")


def example2():
    start_time = datetime.now()

    lista=[]
    #TODO: use training images, instead of validation
    for i in range(0,100000):
        a=100*100
    lista.append(a)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))

if __name__ == "__main__":

    start_time = datetime.now()
    chunck_size = 10    # number of images processed for each task
    num_processes = 10  # number of processes where scheduling tasks

    #TODO: use training images, instead of validation
    input_images = '../COCO/images/val2017/'
    df = inizializeCSV()
    run_tasks('../COCO/annotations/panoptic_val2017.json', '../COCO/annotations/panoptic_val2017', df)
    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))
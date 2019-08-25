"""
 Author: Andrea Pasini
 This file provides the code for Semantic Anomaly Detection (SAD) on COCO.

"""

from datetime import datetime
import os
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

def analyze_image(image_name, segments_info, annot_folder):
    #Load png annotation
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    strings = image2strings(img_ann)
    positions = compute_string_positions(strings)
    print("done")

def run_tasks(json_file, annot_folder):
    """
    Run tasks: analyze training annotations
    :param json_file: annotation file with classes for each segment
    :param annot_folder: folder with png annotations
    """
    #Load annotations
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        annot_dict = {}
        for img_ann in json_data['annotations']:
            annot_dict[img_ann['file_name']]=img_ann['segments_info']
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
        pool.apply_async(analyze_image, args=(img, annot_dict[img], annot_folder), callback=update)
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
    run_tasks('../COCO/annotations/panoptic_val2017.json', '../COCO/annotations/panoptic_val2017')
    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))
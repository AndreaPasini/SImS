"""
Author: Andrea Pasini
This file provides the code for extracting relevant subsets from ConceptNet dataset.
"""
import os
import json
from config import conceptnet_full_csv_path, conceptnet_dir, places_json_path, \
    conceptnet_coco_places_csv_path
from panopticapi.utils import load_panoptic_category_info
from pyspark import SparkContext, SQLContext
from semantic_analysis.conceptnet.utils import create_places_list, coco_to_concept, get_coco_concepts_map

def filter_coco(fields, categories, conceptnet_map):
    """
    Given a row of conceptnet database, filters it out if antecedent or consequent are not COCO categories
    :param fields: fields in a row of conceptnet database
    :param categories: COCO categories
    :param conceptnet_map: map to convert categories
    :return: True if the row must be included in the result
    """
    start, end = False, False
    for cat in categories:
        concept = coco_to_concept(cat, conceptnet_map)
        if fields[2] == f"/c/en/{concept}":
            start = True
        if fields[3] == f"/c/en/{concept}":
            end = True
    return start & end

def filter_set(fields, concepts_set):
    """
    Given a row of conceptnet database, filters it out if antecedent or consequent are not in concepts_set
    :param fields: fields in a row of conceptnet database
    :param concepts_set: set of allowed concepts
    :return: True if the row must be included in the result
    """
    c1 = fields[2].split('/')[-1]
    c2 = fields[3].split('/')[-1]
    return (c1 in concepts_set) & (c2 in concepts_set)

### Choose methods to be run ###
class RUN_CONFIG:
    filter_COCO = False         # Retrieve from conceptnet only triplets that contain COCO categories
    filter_place = False        # Retrieve from conceptnet only triplets that contain the substring "place"
    create_places_list = False  # Create "places" list, by analyzing "conceptnet_places.csv"
    filter_COCO_and_places = False   # Retrieve from conceptnet only triplets that contain places or COCO concepts
    associate_to_freq_graphs = True # Associate frequent graphs to conceptnet

if __name__ == '__main__':

    # Get COCO categories and super-categories
    categories_info = load_panoptic_category_info()
    super_categories = set([cat['supercategory'] for cat in categories_info.values()])
    categories = set([cat['name'] for cat in categories_info.values() if cat!='other'])
    # Mapping of "strange" COCO categories to concept available in ConceptNet (e.g. window-merged -> window)
    conceptnet_map = get_coco_concepts_map()
    # Obtain final concepts from COCO categories and super-categories
    concepts = {coco_to_concept(cat, conceptnet_map) for cat in categories | super_categories}

    if RUN_CONFIG.filter_COCO:
        sc = SparkContext("local", "Read ContextNet")
        sqlContext = SQLContext(sc)
        # Retrieve from conceptnet only triplets that contain COCO categories
        filtered_rdd = sc.textFile(conceptnet_full_csv_path) \
            .map(lambda line: line.split("\t")) \
            .filter(lambda fields: fields[2].startswith('/c/en/') & fields[3].startswith('/c/en/')) \
            .filter(lambda fields: filter_set(fields, concepts))
        df = sqlContext.createDataFrame(filtered_rdd) \
            .toPandas() \
            .to_csv(os.path.join(conceptnet_dir, "conceptnet_coco.csv"), sep='\t', header=False, index=False)
    if RUN_CONFIG.filter_place:
        sc = SparkContext("local", "Find places in ContextNet")
        sqlContext = SQLContext(sc)
        # Retrieve from conceptnet only triplets that contain the substring "place"
        filtered_rdd = sc.textFile(conceptnet_full_csv_path) \
            .map(lambda line: line.split("\t")) \
            .filter(lambda fields: fields[2].startswith('/c/en/') & fields[3].startswith('/c/en/')) \
            .filter(lambda fields: ("place" in fields[2]) or ("place" in fields[3]))
        df = sqlContext.createDataFrame(filtered_rdd) \
            .toPandas() \
            .to_csv(os.path.join(conceptnet_dir, "conceptnet_places.csv"), sep='\t', header=False, index=False)
    if RUN_CONFIG.create_places_list:
        create_places_list(places_json_path)
    if RUN_CONFIG.filter_COCO_and_places:
        with open(places_json_path, 'r') as f:
            places_dict = json.load(f)
        places = set(places_dict['places'] + places_dict['sub-places'])
        sc = SparkContext("local", "Extract places+coco edges from ContextNet")
        sqlContext = SQLContext(sc)
        # Retrieve from conceptnet only triplets that contain COCO categories
        filtered_rdd = sc.textFile(conceptnet_full_csv_path) \
            .map(lambda line: line.split("\t")) \
            .filter(lambda fields: fields[2].startswith('/c/en/') & fields[3].startswith('/c/en/')) \
            .filter(lambda fields: filter_set(fields, concepts | places))
        df = sqlContext.createDataFrame(filtered_rdd) \
            .toPandas() \
            .to_csv(conceptnet_coco_places_csv_path, sep='\t', header=False, index=False)

    print("End")


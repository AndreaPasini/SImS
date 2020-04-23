import requests
import os
import json

from config import conceptnet_full_csv_path, conceptnet_dir
from panopticapi.utils import load_panoptic_category_info
#'http://api.conceptnet.io/a/%5B/r/UsedFor/,/c/en/example/,/c/en/explain/%5D'
from pyspark import SparkContext, SQLContext
import pandas as pd


def coco_to_concept(coco_class, conceptnet_map):
    """
    Convert a string with the COCO class to a name suitable for conceptnet
    Usees conceptnet_map when a conversion is available, otherwise leaves the class as it is
    Classes with names separated by space are joined with "_" as in Conceptnet
    :param coco_class: COCO class (string)
    :param conceptnet_map: mapping COCO class -> concepnet name, for categories like {"window-merged" : "window"}
    :return: converted class name
    """
    if coco_class in conceptnet_map.keys():
        coco_class = conceptnet_map[coco_class]
    return coco_class.replace(" ", "_")

def filter_data(fields, categories, conceptnet_map):
    """
    Given a row of conceptnet database, filters it out if antecedent and consequent are not COCO categories
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





def get_edges(start, end):
    return requests.get('http://api.conceptnet.io/query?node=/c/en/%s&other=/c/en/%s' % (start, end)).json()
def get_node(node):

    'http://api.conceptnet.io/query?node=/c/en/%s&other=/c/en'

    res = requests.get('http://api.conceptnet.io/c/en/%s' % node).json()
    if 'error' in res.keys():
        return None
    else:
        return res





if __name__ == '__main__':
    ### Choose methods to be run ###
    class RUN_CONFIG:
        create_filtered_Conceptnet = False
        get_places = False

    # Get COCO categories and super-categories
    categories = load_panoptic_category_info()
    super_categories = set([cat['supercategory'] for cat in categories.values()])
    categories = set([cat['name'] for cat in categories.values() if cat!='other'])
    # Mapping of "strange" COCO categories to concept available in ConceptNet (e.g. window-merged -> window)
    with open('./classes/panoptic_conceptnet.csv', 'r') as f:
        conceptnet_map = {line.split(":")[1]:line.strip().split(":")[2] for line in f.readlines()}

    if RUN_CONFIG.create_filtered_Conceptnet:
        sc = SparkContext("local", "Read ContextNet")
        sqlContext = SQLContext(sc)
        # Retrieve from conceptnet only triplets that contain COCO categories
        filtered_rdd = sc.textFile(conceptnet_full_csv_path) \
            .map(lambda line: line.split("\t")) \
            .filter(lambda fields: fields[2].startswith('/c/en/') & fields[3].startswith('/c/en/')) \
            .filter(lambda fields: filter_data(fields, categories | super_categories, conceptnet_map))
        df = sqlContext.createDataFrame(filtered_rdd) \
            .toPandas() \
            .to_csv(os.path.join(conceptnet_dir, "conceptnet_coco.csv"), sep='\t', header=False, index=False)
    if RUN_CONFIG.get_places:
        sc = SparkContext("local", "Find places in ContextNet")
        sqlContext = SQLContext(sc)
        # Retrieve from conceptnet only triplets that contain COCO categories
        filtered_rdd = sc.textFile(conceptnet_full_csv_path) \
            .map(lambda line: line.split("\t")) \
            .filter(lambda fields: fields[2].startswith('/c/en/') & fields[3].startswith('/c/en/')) \
            .filter(lambda fields: ("place" in fields[2]) or ("place" in fields[3]))
        df = sqlContext.createDataFrame(filtered_rdd) \
            .toPandas() \
            .to_csv(os.path.join(conceptnet_dir, "conceptnet_places.csv"), sep='\t', header=False, index=False)

    #df = pd.read_csv("../ConceptNet/conceptnet_coco.csv", sep='\t', header=None)
    df = pd.read_csv("../ConceptNet/conceptnet_places.csv", sep='\t', header=None)
    # for category in categories:
    #     cat_json = get_node('house') #category['name'])
    #     if cat_json is None:
    #         print("no")
    #
    # #sky = get_node('sky')
    # edges = get_edges('chair', 'floor')
    print("End")


'AtLocation'
'PartOf'
'RelatedTo'
'IsA'
'HasProperty'
'UsedFor'
'CapableOf'
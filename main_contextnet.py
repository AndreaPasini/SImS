import requests
import os
import json
from panopticapi.utils import load_panoptic_category_info
#'http://api.conceptnet.io/a/%5B/r/UsedFor/,/c/en/example/,/c/en/explain/%5D'
from pyspark import SparkContext, SQLContext
import pandas as pd

def get_edges(start, end):
    return requests.get('http://api.conceptnet.io/query?node=/c/en/%s&other=/c/en/%s' % (start, end)).json()
def get_node(node):

    'http://api.conceptnet.io/query?node=/c/en/%s&other=/c/en'

    res = requests.get('http://api.conceptnet.io/c/en/%s' % node).json()
    if 'error' in res.keys():
        return None
    else:
        return res

# def match_word(target, reference):

def coco_to_concept(coco_class, conceptnet_map):
    if coco_class in conceptnet_map.keys():
        coco_class = conceptnet_map[coco_class]
    return coco_class.replace(" ", "_")

def filter_data(fields, categories, conceptnet_map):
    #.filter(lambda fields: (fields[2].split('/')[-1] in categories) & (fields[3].split('/')[-1] in categories) ) \
    start, end = False, False
    for cat in categories:
        concept = coco_to_concept(cat, conceptnet_map)
        if fields[2] == '/c/en/' + concept:
            start = True
        if fields[3] == '/c/en/' + concept:
            end = True
    return start & end

if __name__ == '__main__':
    categories = load_panoptic_category_info()
    super_categories = set([cat['supercategory'] for cat in categories.values()])
    categories = set([cat['name'] for cat in categories.values() if cat!='other'])
    with open('./classes/panoptic_conceptnet.csv', 'r') as f:
        conceptnet_map = {line.split(":")[1]:line.strip().split(":")[2] for line in f.readlines()}


    # sc = SparkContext("local", "Read ContextNet")
    # sqlContext = SQLContext(sc)
    #
    # filtered_rdd = sc.textFile("../ConceptNet/conceptnet.csv") \
    #     .map(lambda line: line.split("\t")) \
    #     .filter(lambda fields: fields[2].startswith('/c/en/') & fields[3].startswith('/c/en/')) \
    #     .filter(lambda fields: filter_data(fields, categories | super_categories, conceptnet_map))
    # df = sqlContext.createDataFrame(filtered_rdd) \
    #     .toPandas() \
    #     .to_csv("../ConceptNet/conceptnet_subset.csv", sep='\t', header=False, index=False)

    # 591 con solo categorie
    # 769 con super-categories

    df = pd.read_csv("../ConceptNet/conceptnet_subset.csv", sep='\t', header=None)

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
import requests
import os
import json

from config import conceptnet_full_csv_path, conceptnet_dir, places_json_path
from panopticapi.utils import load_panoptic_category_info
#'http://api.conceptnet.io/a/%5B/r/UsedFor/,/c/en/example/,/c/en/explain/%5D'
from pyspark import SparkContext, SQLContext
import pandas as pd

from semantic_analysis.conceptnet import create_places_list


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




def get_edges(start, end):
    return requests.get('http://api.conceptnet.io/query?node=/c/en/%s&other=/c/en/%s' % (start, end)).json()
def get_node(node):

    'http://api.conceptnet.io/query?node=/c/en/%s&other=/c/en'

    res = requests.get('http://api.conceptnet.io/c/en/%s' % node).json()
    if 'error' in res.keys():
        return None
    else:
        return res



def readable_conceptnet(df_conceptnet):
    res = df_conceptnet.copy()
    res[2] = res[2].map(lambda el: el.split("/")[-1])
    res[3] = res[3].map(lambda el: el.split("/")[-1])
    res[1] = res[1].map(lambda el: el.split("/")[-1])
    return res

df_place = readable_conceptnet(pd.read_csv("../ConceptNet/conceptnet_places.csv", sep='\t', header=None))
# Select only meaningful relationships
meaningful_rel = {'RelatedTo', 'AtLocation','IsA','HasA','PartOf','UsedFor','MadeOf','LocatedNear'}
df_place = df_place.loc[lambda x: x[1].isin(meaningful_rel)]

def find_good_places():
    p1 = df_place[2].loc[lambda x: (x.str.contains("_place")) | (x=="place") | x.str.startswith("place_")]
    p2 = df_place[3].loc[lambda x: (x.str.contains("_place")) | (x=="place") | x.str.startswith("place_")]
    places = p1.append(p2)
    places_count = places.value_counts()
    print(places_count)

# Candidate places
sel_places = ['place','eating_place','public_place','meeting_place','hiding_place',\
              'living_place','dwelling_place', 'resting_place','burial_place','work_place',\
              'gathering_place','worship_place','big_place','higher_place','food_place','sleeping_place',\
              'bar_place', 'baiting_place','storage_place','learning_place','writing_place','play_place',\
             'working_place','public_places','drinking_place','religious_place','animal_place','study_place',\
             'house_place', 'planting_place', 'ski_place', 'pizza_place', 'cafe_place','meeting_places',\
             'vacation_place','flower_place','parking_place','driving_place','vegetable_place','swimming_place',\
             'high_place','rural_place','skiing_place','agricultural_place','tree_place','computer_place',\
             'natural_places','eating_places','city_place','fishing_place','job_place','yard_place','good_place_to_relax',\
             'comfortable_place_to_sleep','good_place_for_vacation','doctors_place','selling_place','church_place',\
             'water_place','tennis_place','pleasant_outdoor_place','cooking_place','great_place_to_go_jogging',\
             'sand_place','urban_place','dining_place','cocktail_place','boat_place','nature_place','workout_place',\
             'theatre_place','dinner_place','sleep_place','harbour_place','book_place', 'place_basketball','place_baseball',\
              'place_to_eat','place_of_worship','place_to_sleep','place_to_learn',\
              'place_house','place_with_plants','place_to_clean_up_yourself','place_in_house',\
              'place_where_people_may_swim']

# Pick from candidate places only those that: have > 1 link or the single link is a "IsA"
p1 = df_place[2].loc[lambda x: x.isin(sel_places)]
p2 = df_place[3].loc[lambda x: x.isin(sel_places)]
places = p1.append(p2)
places_count = places.value_counts()
single_link_places = list(places_count[places_count==1].index)
meaningful_single = df_place.loc[(df_place[3].isin(single_link_places))&(df_place[1]=="IsA")][3]
meaningful_single.values.tolist()
final_places = set(sel_places) - set(single_link_places) | set(meaningful_single)

# Are there isA relationships between final places?
df_place.loc[lambda x: x[2].isin(final_places) & x[3].isin(final_places)] # No, there aren't
# Sub-classes of final_places?
sub_places = df_place.loc[lambda x: (x[3].isin(final_places)) & (x[1]=="IsA") & (x[2]!="n")]
sub_counts = sub_places[2].value_counts()
sub_places = list(sub_counts.index)

def is_num(s):
    return any(i.isdigit() for i in s)
sub_places = filter(lambda x: (not is_num(x)) and ("'" not in x), sub_places)

# Save to json
places_dict = {'places' : list(final_places), 'sub-places' : list(sub_places)}
with open('./places.json', 'w') as f:
    json.dump(places_dict, f)









if __name__ == '__main__':
    ### Choose methods to be run ###
    class RUN_CONFIG:
        filter_COCO = False         # Retrieve from conceptnet only triplets that contain COCO categories
        filter_place = False        # Retrieve from conceptnet only triplets that contain the substring "place"
        create_places_list = False  # Create "places" list, by analyzing "conceptnet_places.csv"
        filter_COCO_and_places = True   # Retrieve from conceptnet only triplets that contain places or COCO concepts

    # Get COCO categories and super-categories
    categories_info = load_panoptic_category_info()
    super_categories = set([cat['supercategory'] for cat in categories_info.values()])
    categories = set([cat['name'] for cat in categories_info.values() if cat!='other'])
    # Mapping of "strange" COCO categories to concept available in ConceptNet (e.g. window-merged -> window)
    with open('./classes/panoptic_conceptnet.csv', 'r') as f:
        conceptnet_map = {line.split(":")[1]:line.strip().split(":")[2] for line in f.readlines()}
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
        with open(os.path.join(conceptnet_dir, 'places.json'), 'r') as f:
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
            .to_csv(os.path.join(conceptnet_dir, "conceptnet_coco_places.csv"), sep='\t', header=False, index=False)


    print("End")


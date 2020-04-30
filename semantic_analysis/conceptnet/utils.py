import pandas as pd
from config import conceptnet_dir, panoptic_concepts_csv_path
import os
import json

def get_coco_concepts_map():
    """
    Return mapping COCO class -> concepnet name, for categories like {"window-merged" : "window"}
    Useful for coco_to_concept() function
    """
    with open(panoptic_concepts_csv_path, 'r') as f:
        concepts_map = {line.split(":")[1]:line.strip().split(":")[2] for line in f.readlines()}
    return concepts_map

def coco_to_concept(coco_class, coco_concepts_map):
    """
    Convert a string with the COCO class to a name suitable for conceptnet
    Usees conceptnet_map when a conversion is available, otherwise leaves the class as it is
    Classes with names separated by space are joined with "_" as in Conceptnet
    :param coco_class: COCO class (string)
    :param coco_concepts_map: mapping COCO class -> concepnet name, for categories like {"window-merged" : "window"}
    :return: converted class name
    """
    if coco_class in coco_concepts_map.keys():
        coco_class = coco_concepts_map[coco_class]
    return coco_class.replace(" ", "_")

def get_readable_concepts(df_conceptnet):
    """
    Return new dataframe with renamed Concepts (remove conceptnet header from concept names)
    i.e. '/c/en/cat' -> 'cat'
    :param df_conceptnet: dataframe with conceptnet edges
    :return: dataframe with renamed concepts
    """
    res = df_conceptnet.copy()
    res[2] = res[2].map(lambda el: el.split("/")[-1])
    res[3] = res[3].map(lambda el: el.split("/")[-1])
    res[1] = res[1].map(lambda el: el.split("/")[-1])
    return res

def __find_good_places(df_place):
    """
    Return Series of places-count. Place = contains "_place" or "place" or "place_"
    """
    p1 = df_place[2].loc[lambda x: (x.str.contains("_place")) | (x=="place") | x.str.startswith("place_")]
    p2 = df_place[3].loc[lambda x: (x.str.contains("_place")) | (x=="place") | x.str.startswith("place_")]
    places = p1.append(p2)
    places_count = places.value_counts()
    return places_count

def create_places_list(output_path):
    """ Analyze conceptnet_places to find place names in conceptnet. Save result to json """

    # Read Conceptnet subset with all edges that contain the substring "place"
    df_place = get_readable_concepts(pd.read_csv(os.path.join(conceptnet_dir, "conceptnet_places.csv"), sep='\t', header=None))
    # Select only meaningful relationships
    meaningful_rel = {'RelatedTo', 'AtLocation', 'IsA', 'HasA', 'PartOf', 'UsedFor', 'MadeOf', 'LocatedNear'}
    df_place = df_place.loc[lambda x: x[1].isin(meaningful_rel)]
    # List of unique places
    unique_places = __find_good_places(df_place)
    print(f"Concepts that contain 'place': {len(unique_places)}")
    # Pick from unique places only those that: have > 1 link or if they have 1 link, it is a "IsA"
    # >1 link because otherwise not interesting
    # If they have 1 link: IsA can be useful to find sub-places
    single_link_places = list(unique_places[unique_places == 1].index)
    meaningful_single = df_place.loc[(df_place[3].isin(single_link_places)) & (df_place[1] == "IsA")][3].value_counts()
    meaningful_places = set(unique_places.index) - set(single_link_places) | set(meaningful_single.index)
    print(f"Places with >1 link or at least 1 'IsA' link: {len(meaningful_places)}")
    # Manually selected places among the 375 meaningful_places (We exclude "place" that is too generic)
    sel_places = ['place', 'public_places', 'eating_places', 'burial_place', 'great_place_to_go_jogging', 'resting_place', 'place_to_learn',\
     'learning_place', 'ski_place', 'food_place', 'place_with_plants', 'eating_place', 'hiding_place', 'play_place',\
     'gathering_place', 'study_place', 'vacation_place', 'living_place', 'big_place', 'flower_place',\
     'dwelling_place', 'good_place_to_relax', 'sleeping_place', 'house_place', 'bar_place', 'meeting_places',\
     'meeting_place', 'work_place', 'planting_place', 'cafe_place', 'animal_place', 'natural_places', 'place_in_house',\
     'public_place', 'writing_place', 'place_of_worship', 'worship_place', 'vegetable_place', 'driving_place',\
     'pleasant_outdoor_place', 'working_place', 'religious_place', 'place_where_people_may_swim',\
     'place_to_clean_up_yourself', 'storage_place', 'drinking_place', 'place_to_eat']
    print(f"Manually selected places: {len(sel_places)}")
    # Are there isA relationships between selected places? (should note be)
    relationships = df_place.loc[lambda x: x[2].isin(sel_places) & x[3].isin(sel_places)]
    assert(len(relationships)==0)
    # Research for sub-places (IsA "place")
    sub_places = df_place.loc[lambda x: (x[3].isin(sel_places)) & (x[1] == "IsA") & (x[2] != "n")]
    sub_counts = sub_places[2].value_counts()
    sub_places = list(sub_counts.index)
    # Filter sub-places that contain numbers
    def is_num(s):
        return any(i.isdigit() for i in s)
    sub_places = list(filter(lambda x: (not is_num(x)) and ("'" not in x), sub_places))
    print(f"Obtained sub-places, from IsA relationship: {len(sub_places)}")
    # Manually select sub-places
    sub_places = ["living_room", "college", "temple", "school", "restaurant", "kitchen", "library", "desert",\
                  "sea", "airport", "gym", "pool", "countryside", "deserted_island", "outer_space", "pub", "interior_area",\
                  "war_zone", "camp", "town", "parking_zone", "mountain", "campsite", "repair_shop", "house", "parks",\
                  "valley", "stadium", "porch", "department_store", "church", "garden", "city", "university", "beach",\
                  "chapel", "bar", "basement", "vineyard", "bathroom", "room", "motel", "meadows"]
    print(f"Manually selected sub-places: {len(sub_places)}")
    # Manually added places (that do not have relationships with "place")
    sel_places.extend(["railroad"])
    # Remove "Place"
    sel_places.remove("place")
    # Save to json
    places_dict = {'places' : list(sel_places), 'sub-places' : sub_places}
    with open(output_path, 'w') as f:
        json.dump(places_dict, f)

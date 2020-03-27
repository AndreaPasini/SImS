"""
This file contains functions related to image processing for relative position computation
"""

from itertools import groupby
import numpy as np

def image2strings(img_ann):
    """
    Transform image annotation to vertical string representation
    example: aaaaabbbcc -> [(a,5),(b,3),(c,2)]

    ----
    :param img_ann: image png annotation (numpy matrix)
    :return: list of tuples. Each tuple contains the list of ids and counts
    """

    # Trannspose matrix:
    img_ann = np.transpose(img_ann)

    strings = []
    # For each column of the image (=rows in the transposed image)
    for col in img_ann:
        ids = []  # Object ids
        counts = []  # sub-string lengths
        for k, g in groupby(col):
            ids.append(k)
            counts.append(sum(1 for _ in g))
        strings.append((ids, counts))
    return strings

def compute_string_positions(strings, object_ordering=None):
    """
    Compute object positions given string representation.

    ----
    :param strings: string representation
    :param object_ordering: object ids ordered if you want a specific object order the returned pairs
    :return: dictionary with object positions for each object pair

    Pseudocode: position of two objects i, j given a string
    last_object = None
    last_occurrence = -1
    for pos in [0,len(string)]
      current=string[n]
      if current==i or current==j
          if last_object != current
              if pos > last_occurrence + 1
                  obj_list = obj_list + '+' + current
              else
                  obj_list = obj_list + current
          last_object = current
          last_occurrence = pos
    look for obj_list: ij, ji, i+j, j+i, iji, jij
    """
    positions = {}

    # For each string (image column)
    for string_ids, counts in strings:
        # Get unique object ids in the string
        if object_ordering:
            objects = [obj for obj in object_ordering if obj in string_ids]
        else:
            objects = sorted(list(set(string_ids)-{0}))
        # For all object pairs
        for i in range(0, len(objects)-1):
            for j in range(i+1, len(objects)):
                obj_i = objects[i]
                obj_j = objects[j]

                pair = (obj_i, obj_j)
                if not (pair in positions.keys()):
                    ij_positions = {"i on j": 0, "j on i": 0, "i above j": 0, "j above i": 0, "i around j": 0,
                                    "j around i": 0, "other": 0}
                    positions[pair] = ij_positions
                else:
                    ij_positions = positions[pair]

                obj_list = ''
                last_obj = None
                last_occur = -1
                pos = 0
                for current in string_ids:
                    if current == obj_i or current==obj_j:
                        if last_obj != current:
                            if pos > last_occur + 1 and last_occur!=-1:
                                obj_list = obj_list + ('+i' if current == obj_i else '+j')
                            else:
                                obj_list = obj_list + ('i' if current == obj_i else 'j')
                        last_obj = current
                        last_occur = pos
                    pos += 1

                if obj_list == 'ij':
                    ij_positions["i on j"] += 1
                elif obj_list == 'ji':
                    ij_positions["j on i"] += 1
                elif obj_list == 'i+j':
                    ij_positions["i above j"] += 1
                elif obj_list == 'j+i':
                    ij_positions["j above i"] += 1
                elif obj_list == 'iji' or obj_list == 'i+ji' or obj_list == 'ij+i' or obj_list == 'i+j+i':
                    ij_positions["i around j"] += 1
                elif obj_list == 'jij' or obj_list == 'j+ij' or obj_list == 'ji+j' or obj_list == 'j+i+j':
                    ij_positions["j around i"] += 1
                else:
                    ij_positions["other"] += 1
    return positions

def extract_bbox_from_mask(mask):
    """
    Extract bounding box, given a boolean mask with object pixels

    ----
    :return: [y1, x1, y2, x2]
    """
    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    vertical_indices = np.where(np.any(mask, axis=1))[0]
    if horizontal_indices.shape[0]:
        x1, x2 = horizontal_indices[[0, -1]]
        y1, y2 = vertical_indices[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    return np.array([y1, x1, y2, x2])

def extract_bbox(img_ann, object):
    """
    Extract bounding box, given image annotation and object id

    ----
    :return: [y1, x1, y2, x2]
    """
    mask = np.ma.mask_rowcols(img_ann == object, img_ann)
    mask = mask.astype(np.int)
    return extract_bbox_from_mask(mask)

def getWidth(bbox):
    """ Return the width of a bounding box """
    box_left = bbox[1]
    box_right = bbox[3]
    return box_right - box_left

def getSideFeatures(bboxSubject, bboxReference):
    """ Given bounding boxes of subject and reference, extract features for side relative position """
    boxSub_top, boxSub_left, boxSub_bottom, boxSub_right = bboxSubject
    boxRef_top, boxRef_left, boxRef_bottom, boxRef_right = bboxReference

    deltaY1 = boxRef_bottom - boxSub_top
    deltaY2 = boxRef_top - boxSub_bottom
    deltaX1 = boxRef_right - boxSub_left
    deltaX2 = boxRef_left - boxSub_right

    normY = max(abs(deltaY1), abs(deltaY2))
    normX = max(abs(deltaX1), abs(deltaX2))

    return [deltaY1/normY, deltaY2/normY, deltaX1/normX, deltaX2/normX]

def get_features(img_ann, image_id, subject, reference, positions):
    """
    Get features vector from a pair of object

    ----
    :param img_ann: numpy array with png annotation (from load_png_annotation() )
    :param image_id: identifier of the image (filename without .png)
    :param subject: subject id
    :param reference: reference id
    :param positions: positions extracted with compute_string_positions(strings)
    :return: the feature vector
    """
    pos = positions[(subject, reference)]
    subject_bbox = extract_bbox(img_ann, subject)
    reference_bbox = extract_bbox(img_ann, reference)
    subjectWidth = getWidth(subject_bbox)
    referenceWidth = getWidth(reference_bbox)
    featuresRow = [image_id, subject, reference] + [v / min(subjectWidth, referenceWidth) for k,v in pos.items()] + getSideFeatures(subject_bbox, reference_bbox)
    return featuresRow
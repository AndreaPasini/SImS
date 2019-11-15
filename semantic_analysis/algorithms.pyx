from itertools import groupby
import numpy as np

def image2strings(img_ann):
    """
    Transform image annotation to vertical string representation
    example: aaaaabbbcc -> [(a,5),(b,3),(c,2)]
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

def compute_string_positions(strings):
    """
    Compute object positions given string representation.
    :param strings: string representation
    :return: dictionary with object positions

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
        objects  = np.unique(np.array(string_ids))
        # For all object pairs
        for i in range(0, len(objects)-1):

            if objects[i]==0:
                continue
            for j in range(i+1, len(objects)):
                if objects[i] == 0:
                    continue

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

def extract_bboxes(mask):
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indices = np.where(np.any(m, axis=0))[0]
        vertical_indices = np.where(np.any(m, axis=1))[0]
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
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def getSideFeatures(img_ann, subject, reference):
    bboxSubject = getBbox(img_ann, subject)
    bboxReference = getBbox(img_ann, reference)

    img_height = img_ann.shape[0]
    img_width = img_ann.shape[1]

    boxSub_top = bboxSubject[0, 0] / img_height
    boxSub_bottom = bboxSubject[0, 2] / img_height
    boxRef_top = bboxReference[0, 0] / img_height
    boxRef_bottom = bboxReference[0, 2] / img_height

    boxSub_left = bboxSubject[0][1] / img_width
    boxSub_right = bboxSubject[0][3] / img_width
    boxRef_left = bboxReference[0][1] / img_width
    boxRef_right = bboxReference[0][3] / img_width

    return [boxRef_bottom - boxSub_top, boxRef_top - boxSub_bottom, boxRef_right - boxSub_left, boxRef_left - boxSub_right]

def getBbox(img_ann, object):
    mask = np.ma.mask_rowcols(img_ann == object, img_ann)
    mask = mask.astype(np.int)
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    return extract_bboxes(mask)

def getWidthSubject(img_ann, subject):
    bboxSubject = getBbox(img_ann, subject)
    boxSub_left = bboxSubject[0][1]
    boxSub_right = bboxSubject[0][3]
    return boxSub_right - boxSub_left


def get_features(img_ann, image_id, subject, reference, positions):
    """
    Get features vector from a pair of object
    :param img_ann: numpy array with png annotation (from load_png_annotation() )
    :param image_id: identifier of the image (filename without .png)
    :param subject: subject id
    :param reference: reference id
    :param positions: positions extracted with compute_string_positions(strings)
    :return: the feature vector
    """
    pos = positions[(subject, reference)]
    subjectWidth = getWidthSubject(img_ann, subject)
    featuresRow = [image_id, subject, reference] + [v / subjectWidth for k,v in pos.items()] + getSideFeatures(img_ann, subject, reference)
    return featuresRow
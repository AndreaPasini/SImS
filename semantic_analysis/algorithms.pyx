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
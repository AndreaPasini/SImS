from itertools import groupby

import numpy as np


def compress_mask(mask):
    """ Compress mask with RLE """
    cmask = []
    for row in mask:
        compressed_row = [(sum(1 for _ in g)) for k, g in groupby(row)]
        if row[0] == 1:
            compressed_row = [0] + compressed_row
        cmask.append(compressed_row)
    return cmask


def extract_mask(cmask):
    """ Extract compressed mask with RLE, 1/0 values """
    mask = []

    for crow in cmask:
        row = []
        val = 0
        for count in crow:
            if count > 0:
                row+=([val for i in range(count)])
            if val == 0:
                val = 1
            else:
                val = 0
        mask.append(row)
    return np.array(mask)


def extract_mask_bool(cmask):
    """ Extract compressed mask with RLE, True/False values """
    mask = []

    for crow in cmask:
        row = []
        val = False
        for count in crow:
            if count > 0:
                row+=([val for i in range(count)])
            if val == False:
                val = True
            else:
                val = False
        mask.append(row)
    return np.array(mask)
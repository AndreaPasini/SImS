import numpy as np
from PIL import Image
import os
import cv2
from config import COCO_img_train_dir, position_dataset_dir

def getImage(image_name, img_ann, rand):
    """
    Save image with colored subject and reference. Used to
    :param image_name: file name of the
    :param img_ann: numpy pixel annotation of the image (object ids).
    :param rand: contains ids of subject and reference
    """
    subject = rand[0][0]  # blue
    reference = rand[0][1]  # yellow

    rgbReference = [254, 204, 92]
    rgbSubject = [8, 81, 156]

    img = Image.open(os.path.join(COCO_img_train_dir, image_name[:-3] + 'jpg')).convert('L')

    stacked_img = np.stack((np.array(img),) * 3, axis=-1)
    stacked_img[img_ann==subject] = rgbSubject
    stacked_img[img_ann==reference] = rgbReference

    image = Image.fromarray(stacked_img, 'RGB')
    image.save(os.path.join(position_dataset_dir, 'training', image_name), 'PNG')

def mask_baricenter(mask):
    """
    Given a 2D boolean mask, return x,y coordinates of its baricenter
    """
    binary = np.zeros(mask.shape, dtype=np.uint8)
    binary[mask]=255
    m = cv2.moments(binary)
    if m["m00"]>0:
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        return x, y
    else: return None, None

def getImageName(imageId, pathFile, extension="png"):
    """ Return image file name, given COCO Id and path."""
    imageid = int(imageId)
    imageName = f"{imageid:012d}.{extension}"
    image = os.path.join(pathFile, imageName)
    return image
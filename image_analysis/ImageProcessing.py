import numpy as np
from PIL import Image
import pyximport
pyximport.install(language_level=3)


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

    img = Image.open('../COCO/images/train2017' + '/' + image_name[:-3] + 'jpg').convert('L')   #TODO: '../COCO/images/train2017' lo passerei come parametro di getImage

    stacked_img = np.stack((np.array(img),) * 3, axis=-1)
    mask = getMask(img_ann, subject)
    stacked_img[mask == True] = rgbSubject
    mask = getMask(img_ann, reference)
    stacked_img[mask == True] = rgbReference

    image = Image.fromarray(stacked_img, 'RGB')
    image.save('../COCO/positionDataset/training/' + image_name, 'PNG')         #TODO: '../COCO/positionDataset/training/' lo passerei come parametro di getImage

    print("Done")


def getMask(img_ann, object):
    mask = np.ma.mask_rowcols(img_ann == object, img_ann)
    return mask


def getImageName(imageId, pathFile):
    imageid = int(imageId)
    imageName = f"{imageid:012d}.png"
    image = pathFile + "/" + imageName
    return image
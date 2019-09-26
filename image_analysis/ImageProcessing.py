import numpy as np
from PIL import Image


def getImage(image_name, img_ann, rand):

    subject = rand[0][0]  # yellow
    reference = rand[0][1]  # blue

    rgbSubject = [254, 204, 92]
    rgbReference = [8, 81, 156]

    img = Image.open('../COCO/images/train2017' + '/' + image_name[:-3] + 'jpg').convert('L')

    stacked_img = np.stack((np.array(img),) * 3, axis=-1)
    getMask(stacked_img, img_ann, subject, rgbSubject)
    getMask(stacked_img, img_ann, reference, rgbReference)

    image = Image.fromarray(stacked_img, 'RGB')
    image.save('../COCO/positionDataset/training/' + image_name, 'PNG')

def getMask(stacked_img, img_ann, object, rgb):
    mask = np.ma.mask_rowcols(img_ann == object, img_ann)
    stacked_img[mask == True] = rgb
    return stacked_img

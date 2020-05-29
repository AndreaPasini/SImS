import cv2
import os
from config import COCO_img_train_dir
import numpy as np
from sklearn.cluster import KMeans
from joblib import dump, load
import pandas as pd
from datetime import datetime
competitors_dir = '../COCO/competitors/'

def get_SIFT(img, sift):
    """
    Return SIFT descriptors for this image (resized 200x200)
    """
    img = cv2.resize(img, (200,200))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (kps, descs) = sift.detectAndCompute(img_gray, None)
    return kps, descs

def get_descriptors(img_names):
    """
    Return SIFT descriptors matrix (N. images x 100 x 128)
    Max 100 descriptors for each image
    """
    X = None
    sift = cv2.xfeatures2d.SIFT_create()
    for i, img_name in enumerate(img_names):
        img = cv2.imread(os.path.join(COCO_img_train_dir, img_name))
        kps, descs = get_SIFT(img, sift)
        if descs is not None and len(descs)>0:
            if X is None:
                X = descs[:100]
            else:
                X = np.vstack([X, descs[:100]])
    return X

def get_BOW(img_names, codebook):
    X = None
    sift = cv2.xfeatures2d.SIFT_create()
    for i, img_name in enumerate(img_names):    # 34 secondi con 1000 immagini
        img = cv2.imread(os.path.join(COCO_img_train_dir, img_name))
        kps, descs = get_SIFT(img, sift)
        fvect = np.zeros(codebook.n_clusters)
        if descs is not None and len(descs)>0:
            y = codebook.predict(descs)
            unique, counts = np.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                fvect[u]=c/len(descs)

        if X is None:
            X = fvect
        else:
            X = np.vstack([X, fvect])
        if i%1000 == 0:
            print(f"Done: {i}")
    return X

images = os.listdir(COCO_img_train_dir)
# selected = np.random.choice(images, 10000, replace=False)
# # Computing descriptors
# print("Computing descriptors...")
# start_time = datetime.now()
# X = get_descriptors(selected)
# X.dump(os.path.join(competitors_dir, "sift_descr_collection.np"))
# print("Saved.")
# end_time = datetime.now()
# print('Duration: ' + str(end_time - start_time))
#
# print("Computing codebook with KMeans...")
# start_time = datetime.now()
# X = np.load(os.path.join(competitors_dir, "sift_descr_collection.np"),allow_pickle=True)
# print(f"Initial data: {X.shape[0]}")
# X = X[np.random.choice(X.shape[0], 100000, replace=False), :] # 100K samples
# print(f"Sampled data: {X.shape[0]}")
# codebook = KMeans(500) # Number of codes
# y = codebook.fit_transform(X)
# dump(codebook, os.path.join(competitors_dir, "sift_codebook.pkl"))
# print("Saved.")
# end_time = datetime.now()
# print('Duration: ' + str(end_time - start_time))

print("Computing feature vectors for all images...")
start_time = datetime.now()
codebook = load(os.path.join(competitors_dir, "sift_codebook.pkl"))
X = get_BOW(images, codebook)
df = pd.DataFrame(X)
df.to_csv(os.path.join(competitors_dir, "bow_images.np"), index=False)
X.dump(os.path.join(competitors_dir, "bow_images.np"))
print("Saved.")
end_time = datetime.now()
print('Duration: ' + str(end_time - start_time))
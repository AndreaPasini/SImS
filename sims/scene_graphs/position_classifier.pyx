"""
This file contains functions related to the relative position classifier:
- validate_classifiers_grid_search() -> evaluate position classifiers
- build_final_model() -> create final position classifier
- create_kb_graphs() -> apply position classifier to obtain image graphs
"""

import matplotlib
import numpy as np
import pandas as pd
import json
import os
import pickle
from multiprocessing.pool import Pool
from os import listdir
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from config import position_dataset_res_dir, COCO_panoptic_cat_info_path
from main_dataset_labeling import pathGroundTruthBalanced, pathFeaturesBalanced
from panopticapi.utils import load_png_annotation

import pyximport
pyximport.install(language_level=3)
from sims.graph_utils import nx_to_json
from sims.scene_graphs.feature_extraction import image2strings, compute_string_positions, get_features


def __getClassifiersGridSearch():
    """ Return a list of classifiers and parameters for grid-search (validate_classifiers_grid_search) """
    param_knn = {'n_neighbors': [5, 10, 15]}
    param_svc = {'gamma': ['auto']}
    param_dtree = {'max_depth': [5, 10, 15, 20, 25, 30, 35]}
    param_rforest = {'max_depth': [5, 10, 15, 20, 25, 30, 35],
                     'n_estimators': [10, 15, 20, 25, 30, 35, 40, 45, 50]}  # ,60,80,100] }

    classifiers = [("KNN", KNeighborsClassifier(), param_knn),
                   ("RBF-SVC", SVC(), param_svc),
                   ("Decision tree", DecisionTreeClassifier(), param_dtree),
                   ("Random forest", RandomForestClassifier(), param_rforest)]

    return classifiers

# Utilities for file management
def __removeFile(filePath):
    if os.path.isfile(filePath):
        os.remove(filePath)
def __inizializePath(path):
    if not os.path.exists(path):
        os.mkdir(path)

def __getClassF1_df(y, y_pred, nameClf, F1_df):
    """
    For the given results, return F1 scores, separately for each class.
    Append the results on to given table (dfAccuracy)
    :param y, y_pred: ground-truth and predictions
    :param nameClf: name of the classifier used for predictions (for pretty-printing)
    :param F1_df: table to which append the results
    """
    precision, recall, f1, s = precision_recall_fscore_support(y, y_pred)
    column_names = np.unique(y)
    matrix_f1 = np.reshape(f1, (1, f1.size))
    df_f1 = pd.DataFrame(matrix_f1, columns=column_names, index=[nameClf])
    F1_df = F1_df.append(df_f1)
    return F1_df

def __save_confusion_matrix(y, y_pred, clf_name, f1_macro, output_file):
    """
    Print comfusion matrix for the given predictions.
    Save results to file.
    :param y, y_pred: ground-truth and predictions
    :param clf_name: name of the classifier
    :param f1_macro: macro average f1 of these results
    :param output_file: output file (.eps)
    """
    column_names = np.unique(y)
    conf_mat = confusion_matrix(y, y_pred, labels=column_names)
    conf_mat_df = pd.DataFrame(conf_mat, index=column_names, columns=column_names)
    fig, ax = plt.subplots(figsize=[12,10])

    im, cbar = __heatmap(conf_mat_df, column_names, column_names, ax=ax,
                         cmap="YlGn", cbarlabel=".")
    __annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.title("Confusion Matrix, " + clf_name + '\nmacro-average f1: {0:.3f}'.format(f1_macro))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print(conf_mat_df)
    __removeFile(output_file)
    fig.set_size_inches(13, 10, forward=True)
    plt.savefig(output_file)


def validate_classifiers_grid_search(output_path):
    """
    Perform a grid-search to validate position classifiers.
    For each classifier validates the best parameter configuration with Kfold 10.
    Write confusion matrices and F1-scores for the best configuration, using leave-one-out
    """

    __inizializePath(position_dataset_res_dir)
    data = pd.read_csv(pathFeaturesBalanced, sep=';')
    data_img = pd.read_csv(pathGroundTruthBalanced, sep=';')

    X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
    y = np.array(data_img["Position"])

    F1_df = pd.DataFrame()  # F1 scores, separately for each class and classifier
    cv = LeaveOneOut()
    classifiers = __getClassifiersGridSearch()
    try:
        for clf_name, clf, params in classifiers:
            print(clf_name)
            gridSearch = GridSearchCV(cv=10, scoring='f1_macro', estimator=clf, param_grid=params)
            gridSearch.fit(X, y)
            print(f"- {clf_name}, F1: {gridSearch.best_score_:.3f}")
            print(gridSearch.best_params_)

            # Get best classifier from grid search
            best_clf = gridSearch.best_estimator_
            # Use leave-one out for printing its final evaluation
            y_pred = cross_val_predict(best_clf, X, y, cv=cv)
            F1_df = __getClassF1_df(y, y_pred, clf_name, F1_df)

            f1_macro = F1_df.tail().mean(axis=1).get(key=clf_name)
            __save_confusion_matrix(y, y_pred, clf_name, f1_macro, os.path.join(position_dataset_res_dir, clf_name + ".eps"))

        # Save results to file:
        F1_df['macro-average'] = F1_df.mean(axis=1)
        print(F1_df.head().to_string())
        resultFile = open(output_path, "w+")
        resultFile.writelines('F1 Score\n\n')
        resultFile.writelines(F1_df.head().to_string())
        resultFile.close()

    except ValueError as e:
        print(e)

def build_final_model(output_model_file, final_classifier):
    """
    Build the selected model (position classifier) on training data.
    :param output_model_file: path to output model file (sklearn, pickle)
    :param final_classifier: configured sklearn model to be trained
    """
    __inizializePath(position_dataset_res_dir)
    data = pd.read_csv(pathFeaturesBalanced, sep=';')
    data_img = pd.read_csv(pathGroundTruthBalanced, sep=';')

    X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
    y = np.array(data_img["Position"])

    # Fit model
    final_classifier.fit(X, y)
    # Save model to disk
    __removeFile(output_model_file)
    pickle.dump(final_classifier, open(output_model_file, 'wb'))

def image2scene_graph(image_name, image_id, segments_info, cat_info, annot_folder, model):
    """
    ** Applicable to data with COCO dataset format. **
    Apply position classifier to this image, compute scene graph
    In each relationships, the order of the pair subject-reference is chosen based on alphabetical order.
    E.g. (ceiling, floor) instead of (floor, ceiling)
    :param image_name: file name of the image
    :param image_id: identifier of the image (number extracted from image name, without leading zeros)
    :param segments_info: json with segment class information
    :param cat_info: COCO category information
    :param annot_folder: path to annotations
    :param model: relative-position classifier
    :return the scene graph
    """
    if len(segments_info)==0:
        print('Image has no segments.')
        return None

    catInf = pd.DataFrame(cat_info).T
    segInfoDf = pd.DataFrame(segments_info)

    merge = pd.concat([segInfoDf.set_index('category_id'), catInf.set_index('id')], axis=1,
                      join='inner').set_index('id')

    result = merge['name'].sort_values()
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    strings = image2strings(img_ann)
    object_ordering = result.index.tolist()
    positions = compute_string_positions(strings, object_ordering)
    g = nx.DiGraph()
    g.name = image_id
    for id, name in result.iteritems():
        g.add_node(id, label=name)
    for (s, r), pos in list(positions.items()):
        featuresRow = get_features(img_ann, "", s, r, positions)
        prediction = model.predict([np.asarray(featuresRow[3:])])[0]
        g.add_edge(s, r, pos=prediction)
    return g

def create_scene_graphs(fileModel_path, COCO_json_path, COCO_ann_dir, out_graphs_json_path):
    """
    ** Applicable to data with COCO dataset format. **
    Generate scene graphs from images, applying the relative position classifier
    :param fileModel_path: path to relative position model
    :param COCO_json_path: annotation file with classes for each segment (either CNN annotations or ground-truth)
    :param COCO_ann_dir: folder with png annotations (either CNN annotations or ground-truth)
    :param out_graphs_json_path: output json file with scene graphs
    """

    loaded_model = pickle.load(open(fileModel_path, 'rb'))

    # Load annotations
    with open(COCO_json_path, 'r') as f:
        json_data = json.load(f)
        annot_dict = {}
        id_dict = {}
        for img_ann in json_data['annotations']:
            annot_dict[img_ann['file_name']] = img_ann['segments_info']
        for img_ann in json_data['annotations']:
            id_dict[img_ann['file_name']] = img_ann['image_id']

        if 'categories' in json_data:
            cat_dict = {cat_info['id']: cat_info for cat_info in json_data['categories']}
        else:
            with open(COCO_panoptic_cat_info_path, 'r') as f:
                categories_list = json.load(f)
            cat_dict = {el['id']: el for el in categories_list}


    # Get files to be analyzed
    files = sorted(listdir(COCO_ann_dir))

    # Init progress bar
    pbar = tqdm(total=len(files))

    def update(x):
        pbar.update()

    print("Number of images: %d" % len(files))
    print("Scheduling tasks...")
    pool = Pool(10)
    results = []

    # Analyze all images
    for img in files:
        if img.endswith('.png'):
            results.append(pool.apply_async(image2scene_graph, args=(img, id_dict[img], annot_dict[img],
                                                                     cat_dict, COCO_ann_dir, loaded_model), callback=update))
    pool.close()
    pool.join()
    pbar.close()

    # Collect Graph results
    resultGraph = []
    for graph_getter in results:
        graph = graph_getter.get()
        if graph is not None:
            # Get graph description for this image
            resultGraph.append(nx_to_json(graph))

    # Write graphs to file
    with open(out_graphs_json_path, "w") as f:
        json.dump(resultGraph, f)
    print("Done")


def __heatmap(data, row_labels, col_labels, ax=None,
              cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def __annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                       textcolors=["black", "white"],
                       threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

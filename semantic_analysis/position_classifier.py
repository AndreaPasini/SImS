import traceback

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
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from config import kb_dir, COCO_ann_val_dir, COCO_val_json_path, kb_pairwise_json_path, position_dataset_res_dir, \
    train_graphs_json_path, COCO_panoptic_cat_info_path, val_panoptic_graphs
from main_dataset_labeling import pathGroundTruthBalanced, pathFeaturesBalanced
from panopticapi.utils import load_png_annotation

import pyximport
pyximport.install(language_level=3)
from semantic_analysis.gspan_mining.mining import nx_to_json
from scipy.stats import entropy
from semantic_analysis.algorithms import image2strings, compute_string_positions, get_features



def validate_classifiers_grid_search(output_path):
    inizializePath(position_dataset_res_dir)
    data = pd.read_csv(pathFeaturesBalanced, sep=';')
    data_img = pd.read_csv(pathGroundTruthBalanced, sep=';')

    X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
    y = np.array(data_img["Position"])

    dfAccuracy = pd.DataFrame()
    cv = LeaveOneOut()
    classifiers = getClassifiersGridSearch()
    try:
        for name, clf, params in classifiers:
            print(name)
            gridSearch = GridSearchCV(cv=10, scoring='f1_macro', estimator=clf, param_grid=params)
            gridSearch.fit(X, y)
            print(f"- {name}, F1: {gridSearch.best_score_:.3f}")
            print(gridSearch.best_params_)

            # Get best classifier from grid search
            best_clf = gridSearch.best_estimator_
            # Use leave-one out for printing its final evaluation
            y_pred = cross_val_predict(best_clf, X, y, cv=cv)
            dfAccuracy = getClassF1_df(y, y_pred, name, dfAccuracy)
            getConfusionMatrix(y, y_pred, name, dfAccuracy.tail())

        dfAccuracy.plot.bar()
        dfAccuracy['macro-average'] = dfAccuracy.mean(axis=1)
        print(dfAccuracy.head().to_string())
        resultFile = open(output_path, "w+")
        resultFile.writelines('F1 Score\n\n')
        resultFile.writelines(dfAccuracy.head().to_string())
        resultFile.close()

    except ValueError as e:
        print(e)


# def validate_classifiers(output_path):
#     inizializePath(position_dataset_res_dir)
#     data = pd.read_csv(pathFeaturesBalanced, sep=';')
#     data_img = pd.read_csv(pathGroundTruthBalanced, sep=';')
#
#     X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
#     y = np.array(data_img["Position"])
#
#     dfAccuracy = pd.DataFrame()
#     cv = LeaveOneOut()
#     names, classifiers = getClassifiers()
#     try:
#         for nameClf, clf in zip(names, classifiers):
#             print(nameClf)
#             y_pred = cross_val_predict(clf, X, y, cv=cv)
#             dfAccuracy = getClassF1_df(y, y_pred, nameClf, dfAccuracy)
#             getConfusionMatrix(y, y_pred, nameClf, dfAccuracy.tail())
#         dfAccuracy.plot.bar()
#         dfAccuracy['macro-average'] = dfAccuracy.mean(axis=1)
#         print(dfAccuracy.head().to_string())
#         resultFile = open(output_path, "w+")
#         resultFile.writelines('F1 Score\n\n')
#         resultFile.writelines(dfAccuracy.head().to_string())
#         resultFile.close()
#
#     except ValueError as e:
#         print(e)


def getClassifiersGridSearch():
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


# def getClassifiers():
#     names = ["Nearest Neighbors",
#              "Linear SVM",
#              "RBF SVM",
#              "Decision Tree",
#              "Random Forest",
#              "Naive Bayes"]
#     classifiers = [
#         KNeighborsClassifier(5),
#         SVC(kernel="linear", C=0.025),
#         SVC(gamma='auto'),
#         DecisionTreeClassifier(max_depth=10),
#         RandomForestClassifier(max_depth=20, n_estimators=35, random_state=0),
#         GaussianNB()]
#     return names, classifiers


def build_final_model(fileModel, final_classifier):
    inizializePath(position_dataset_res_dir)
    data = pd.read_csv(pathFeaturesBalanced, sep=';')
    data_img = pd.read_csv(pathGroundTruthBalanced, sep=';')

    X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
    y = np.array(data_img["Position"])

    # Fit model
    final_classifier.fit(X, y)
    # Save model to disk
    removeFile(fileModel)
    pickle.dump(final_classifier, open(fileModel, 'wb'))


def getClassF1_df(y, y_pred, nameClf, dfAccuracy):
    precision, recall, f1, s = precision_recall_fscore_support(y, y_pred)
    column_names = np.unique(y)
    matrix_f1 = np.reshape(f1, (1, f1.size))
    df_f1 = pd.DataFrame(matrix_f1, columns=column_names, index=[nameClf])
    dfAccuracy = dfAccuracy.append(df_f1)
    return dfAccuracy


def getConfusionMatrix(y, y_pred, nameClf, row):
    column_names = np.unique(y)
    conf_mat = confusion_matrix(y, y_pred, labels=column_names)
    conf_mat_df = pd.DataFrame(conf_mat, index=column_names, columns=column_names)
    fig, ax = plt.subplots()

    im, cbar = heatmap(conf_mat_df, column_names, column_names, ax=ax,
                       cmap="YlGn", cbarlabel=".")
    texts = annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    accuracy = row.mean(axis=1).get(key=nameClf)
    plt.title("Confusion Matrix " + nameClf + '\nmacro-average f1: {0:.3f}'.format(accuracy))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print(conf_mat_df)
    file = position_dataset_res_dir + "/" + nameClf + ".jpeg"
    removeFile(file)
    fig.set_size_inches(13, 10, forward=True)
    plt.savefig(file)
    # if os.path.isfile(pathImageDetailBalanced):
    #    os.remove(pathImageDetailBalanced)


def heatmap(data, row_labels, col_labels, ax=None,
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


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
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


def create_kb_graphs(fileModel_path, COCO_json_path, COCO_ann_dir, out_graphs_json_path):
    """
    Analyze annotations, by applying classifier
    Generate image graphs, with descriptions
    :param fileModel_path: path to relative position model
    :param COCO_json_path: annotation file with classes for each segment (either CNN annotations or ground-truth)
    :param COCO_ann_dir: folder with png annotations (either CNN annotations or ground-truth)
    :param out_graphs_json_path: output json file with graphs
    """

    loaded_model = pickle.load(open(fileModel_path, 'rb'))

    # Load annotations
    with open(COCO_json_path, 'r') as f:
        json_data = json.load(f)
        annot_dict = {}
        cat_dict = {}
        id_dict = {}
        for img_ann in json_data['annotations']:
            annot_dict[img_ann['file_name']] = img_ann['segments_info']
        for img_ann in json_data['annotations']:
            id_dict[img_ann['file_name']] = img_ann['image_id']
        # if cnnFlag:
        #     with open(COCO_panoptic_cat_info_path, 'r') as f:
        #         categories_list = json.load(f)
        #     cat_dict = {el['id']: el for el in categories_list}
        # else:
        cat_dict = {img_ann['id'] : img_ann for img_ann in json_data['categories']}
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
            results.append(pool.apply_async(compute_graph_from_image, args=(img, annot_dict[img], id_dict[img], cat_dict, COCO_ann_dir, loaded_model),
                                            callback=update))
    pool.close()
    pool.join()
    pbar.close()

    # Collect Graph results
    resultGraph = []
    for img in results:
        if img.get() is not None:
            graph = img.get()
            # Get graph description for this image
            resultGraph.append(nx_to_json(graph))

    # Write graphs to file
    with open(out_graphs_json_path, "w") as f:
        json.dump(resultGraph, f)

    print("Done")


























def compute_graph_from_image(image_name, segments_info, image_id, cat_info, annot_folder, model):
    # Apply position classifier to this image
    # @return the image converted to graph
    #try:

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
    g = nx.Graph()
    g.name = image_id
    for id, name in result.iteritems():
        g.add_node(id, label=name)
    for (s, r), pos in list(positions.items()):
        featuresRow = get_features(img_ann, "", s, r, positions)
        prediction = model.predict([np.asarray(featuresRow[3:])])[0]
        g.add_edge(s, r, pos=prediction)
    return g
    # except Exception as e:
    #     print('Image has no segments.')# Except: segInfoDf has no 'category_id'
    #     #traceback.print_exc()
    #     return None



# def analyze_image(image_name, segments_info, image_id, cat_info, annot_folder, model, cnnFlag):
#     try:
#         catInf = pd.DataFrame(cat_info).T
#         segInfoDf = pd.DataFrame(segments_info)
#         merge = pd.concat([segInfoDf.set_index('category_id'), catInf.set_index('id')], axis=1,
#                           join='inner').set_index('id')
#
#         result = merge['name'].sort_values()
#         img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
#         strings = image2strings(img_ann)
#         object_ordering = result.index.tolist()
#         positions = compute_string_positions(strings, object_ordering)
#         g = nx.Graph()
#         hist = {}
#         g.name = image_id
#         for id, name in result.iteritems():
#             g.add_node(id, label=name)
#         for (s, r), pos in list(positions.items()):
#             featuresRow = get_features(img_ann, "", s, r, positions)
#             subject = result[s]
#             reference = result.loc[r]
#             prediction = model.predict([np.asarray(featuresRow[3:])])[0]
#             g.add_edge(s, r, pos=prediction)
#             if not cnnFlag:
#                 if (subject, reference) not in hist.keys():
#                     hist[subject, reference] = {prediction: 1}
#                 else:
#                     hist[subject, reference].update({prediction: 0})
#                     hist[subject, reference][prediction] += 1
#         return g, hist
#     except Exception as e:
#         print('Caught exception in analyze_image:')
#         traceback.print_exc()
#         return None






















def create_histograms(json_file, annot_folder, model):

        histograms = {}

        # Add up all histogram statistics
        for pair, hist in [(k, v) for img in resultHist for (k, v) in img.items()]:
            if pair not in histograms:
                # add histogram as it is if pair is not existing
                histograms[pair] = hist
            else:
                total_hist = histograms[pair]
                # update histograms if pair already existing
                for key in hist:
                    if key in total_hist:
                        total_hist[key] += hist[key]
                    else:
                        total_hist[key] = hist[key]

        for hist in histograms.values():
            sup = sum(hist.values())  # support: sum of all occurrences in the histogram
            ent = []
            for pos, count in hist.items():
                perc = count / sup
                hist[pos] = perc
                ent.append(perc)
            hist['sup'] = sup
            hist['entropy'] = entropy(ent, base=2)



        if not os.path.isdir(kb_dir):
            os.makedirs(kb_dir)
        with open(kb_pairwise_json_path, "w") as f:
            json.dump({str(k): v for k, v in histograms.items()}, f)




def removeFile(filePath):
    if os.path.isfile(filePath):
        os.remove(filePath)


def inizializePath(path):
    if not os.path.exists(path):
        os.mkdir(path)

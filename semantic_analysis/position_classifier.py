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
import easygui
import pyximport
from networkx.readwrite import json_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from main_dataset_labeling import pathGroundTruthBalanced, pathFeaturesBalanced
from panopticapi.utils import load_png_annotation

pyximport.install(language_level=3)
from semantic_analysis.algorithms import image2strings, compute_string_positions, get_features

result_path = '../COCO/positionDataset/results'
path_json_file = '../COCO/annotations/panoptic_val2017.json'
path_annot_folder = '../COCO/annotations/panoptic_val2017'
json_path_links = '../COCO/positionDataset/results/links.json'
json_path_sup = '../COCO/kb/pairwiseKB.json'


def checkClassifier(classifier):
    if not any(classifier):
        easygui.msgbox("You must choose a classifier!", title="Classifier")
    elif sum(classifier) == 1:
        index = int(" ".join(str(x) for x in np.argwhere(classifier)[0]))
        return index
    else:
        easygui.msgbox("You must choose only a classifier!", title="Classifier")


def validate_classifiers_grid_search():
    inizializePath(result_path)
    data = pd.read_csv(pathFeaturesBalanced, sep=';')
    data_img = pd.read_csv(pathGroundTruthBalanced, sep=';')

    X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
    y = np.array(data_img["Position"])

    classifiers = getClassifiersGridSearch()
    print("Macro F1 scores for different classifiers:")
    for name, clf, params in classifiers:
        gridSearch = GridSearchCV(cv=10, scoring='f1_macro', estimator=clf, param_grid=params)
        gridSearch.fit(X, y)
        print(f"- {name}: {gridSearch.best_score_:.3f}")
        print(gridSearch.best_params_)


def validate_classifiers(output_path):
    inizializePath(result_path)
    data = pd.read_csv(pathFeaturesBalanced, sep=';')
    data_img = pd.read_csv(pathGroundTruthBalanced, sep=';')

    X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
    y = np.array(data_img["Position"])

    dfAccuracy = pd.DataFrame()
    cv = LeaveOneOut()
    names, classifiers = getClassifiers()
    try:
        for nameClf, clf in zip(names, classifiers):
            print(nameClf)
            y_pred = cross_val_predict(clf, X, y, cv=cv)
            dfAccuracy = getAccuracy(y, y_pred, nameClf, dfAccuracy)
            getConfusionMatrix(y, y_pred, nameClf, dfAccuracy.tail())
        dfAccuracy.plot.bar()
        dfAccuracy['macro-average'] = dfAccuracy.mean(axis=1)
        print(dfAccuracy.head().to_string())
        resultFile = open(output_path, "w+")
        resultFile.writelines('F1 Score\n\n')
        resultFile.writelines(dfAccuracy.head().to_string())
        resultFile.close()

    except ValueError as e:
        print(e)


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


def getClassifiers():
    names = ["Nearest Neighbors",
             "Linear SVM",
             "RBF SVM",
             "Decision Tree",
             "Random Forest",
             "Naive Bayes"]
    classifiers = [
        KNeighborsClassifier(5),
        SVC(kernel="linear", C=0.025),
        SVC(gamma='auto'),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=20, n_estimators=35, random_state=0),
        GaussianNB()]
    return names, classifiers


def build_final_model(fileModel, classifier):
    inizializePath(result_path)
    data = pd.read_csv(pathFeaturesBalanced, sep=';')
    data_img = pd.read_csv(pathGroundTruthBalanced, sep=';')

    X = np.array(data.drop(['image_id', 'Subject', 'Reference'], axis=1))
    y = np.array(data_img["Position"])
    index = checkClassifier(classifier)
    # Fit the model on training set
    names, classifiers = getClassifiers()
    model = classifiers[index]
    model.fit(X, y)
    # save the model to disk
    removeFile(fileModel)
    pickle.dump(model, open(fileModel, 'wb'))


def getAccuracy(y, y_pred, nameClf, dfAccuracy):
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
    file = result_path + "/" + nameClf + ".jpeg"
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


def analyze_statics(fileModel_path):
    loaded_model = pickle.load(open(fileModel_path, 'rb'))
    run_tasks(path_json_file, path_annot_folder, loaded_model)


def analyze_image(image_name, segments_info, cat_info, annot_folder, model):
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    catInf = pd.DataFrame(cat_info).T
    segInfoDf = pd.DataFrame(segments_info)
    merge = pd.concat([segInfoDf.set_index('category_id'), catInf.set_index('id')], axis=1,
                      join='inner').reset_index()

    result = merge[['name', 'id']].loc[merge['id'].isin(segInfoDf['id'].values)]
    result.sort_values(by=['name'], inplace=True)
    object_ordering = [(result['id'].tolist(), [])]
    positions = compute_string_positions(None, object_ordering)
    g = nx.Graph()
    hist = {}
    for index, val in enumerate(result.itertuples(), 1):
        g.add_node(val.id, label=val.name)
    for pairObj in list(positions.items()):
        featuresRow = get_features(img_ann, "", pairObj[0][0], pairObj[0][1], positions)
        subject = result.loc[result['id'] == pairObj[0][0], ('name', 'id')].values[0]
        reference = result.loc[result['id'] == pairObj[0][1], ('name', 'id')].values[0]
        prediction = model.predict([np.asarray(featuresRow[3:])])
        g.add_edge(subject[1], reference[1], position=prediction[0])
        hist[subject[0], reference[0]] = prediction[0]

    return g, hist


def run_tasks(json_file, annot_folder, model):
    """
    Run tasks: analyze training annotations
    :param json_file: annotation file with classes for each segment
    :param annot_folder: folder with png annotations
    """
    # Load annotations
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        annot_dict = {}
        cat_dict = {}
        for img_ann in json_data['annotations']:
            annot_dict[img_ann['file_name']] = img_ann['segments_info']
        for img_ann in json_data['categories']:
            cat_dict[img_ann['id']] = img_ann
    # Get files to be analyzed
    files = sorted(listdir(annot_folder))

    # Init progress bar
    pbar = tqdm(total=len(files))

    def update(x):
        pbar.update()

    print("Number of images: %d" % len(files))
    print("Scheduling tasks...")
    pool = Pool(10)
    results = []

    for img in files:
        results.append(pool.apply_async(analyze_image, args=(img, annot_dict[img], cat_dict, annot_folder, model),
                                        callback=update))
    pool.close()
    pool.join()

    resultGraph = []
    resultDict = []
    sup = {}
    positionDict = {}
    for img in results:
        if img.get() is not None:
            result = img.get()
            resultGraph.append(
                json_graph.node_link_data(result[0],
                                          dict(source='s', target='t', name='id', key='key', link='links')))
            resultDict.append(result[1])

    saveToJson(json_path_links, resultGraph)

    for objects, position in [(k, v) for x in resultDict for (k, v) in x.items()]:
        sub, ref = objects
        if objects not in sup.keys():
            sup[sub, ref] = {position: 0}
            sup[sub, ref][position] = 1
        else:
            sup[sub, ref].update({position: 0})
            sup[sub, ref][position] += 1

    for keys, values in sup.items():
        for innerKey, innerValues in values.items():
            values.update({innerKey: innerValues / len(values)})
        sup[keys[0], keys[1]].update({'sup': len(values)})

    for k in list(sup):
        positionDict[str(k)] = sup.pop(k)

    saveToJson(json_path_sup, positionDict)

    pbar.close()
    print("Done")


def saveToJson(path, dict):
    with open(path, "w") as f:
        json.dump(dict, f)


def removeFile(filePath):
    if os.path.isfile(filePath):
        os.remove(filePath)


def inizializePath(path):
    if not os.path.exists(path):
        os.mkdir(path)

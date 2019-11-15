import matplotlib
import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import easygui
from sklearn.preprocessing import StandardScaler
import pyximport
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from main_dataset_labeling import pathGroundTruthBalanced, pathFeaturesBalanced

pyximport.install(language_level=3)

result_path = '../COCO/positionDataset/results'

def checkClassifier(classifier):
    if not any(classifier):
        easygui.msgbox("You must choose a classifier!", title="Classifier")
    elif sum(classifier) == 1:
        index = int(" ".join(str(x) for x in np.argwhere(classifier)[0]))
        return index
    else:
        easygui.msgbox("You must choose only a classifier!", title="Classifier")


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


def getClassifiers():
    names = ["Nearest Neighbors",
             "Linear SVM",
             "RBF SVM",
             "Decision Tree",
             "Random Forest",
             "Naive Bayes"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0),
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
    plt.title("Confusion Matrix "+nameClf + '\nmacro-average f1: {0:.3f}'.format(accuracy))
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

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
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
        threshold = im.norm(data.max())/2.

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

def removeFile(filePath):
    if os.path.isfile(filePath):
        os.remove(filePath)


def inizializePath(path):
    if not os.path.exists(path):
        os.mkdir(path)

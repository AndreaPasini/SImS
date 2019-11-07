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

from main_dataset_labeling import pathImageDetailBalanced, pathFeaturesBalanced

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
    data_img = pd.read_csv(pathImageDetailBalanced, sep=';')

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
        dfAccuracy['mean'] = dfAccuracy.mean(axis=1)
        print(dfAccuracy.head().to_string())
        resultFile = open(output_path, "w+")
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
    data_img = pd.read_csv(pathImageDetailBalanced, sep=';')

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
    conf_mat_df.index.name = 'Actual'
    conf_mat_df.columns.name = 'Predicted'
    print(conf_mat_df)
    accuracy = row.mean(axis=1).get(key=nameClf)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_mat_df, annot=True)
    plt.title(nameClf+'\nAccuracy:{0:.3f}'.format(accuracy))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    file = result_path+"/"+nameClf+".jpeg"
    removeFile(file)
    plt.savefig(file)
    if os.path.isfile(pathImageDetailBalanced):
        os.remove(pathImageDetailBalanced)


def removeFile(filePath):
    if os.path.isfile(filePath):
        os.remove(filePath)


def inizializePath(path):
    if not os.path.exists(path):
        os.mkdir(path)

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

json_path_kb_histograms = '../COCO/kb/pairwiseKB.json'
charts_path = '../COCO/kb/charts'
support_distplot_path = '../COCO/kb/charts/support_distplot.png'
entropy_distplot_path = '../COCO/kb/charts/entropy_distplot.png'
hist_path = '../COCO/kb/charts/hist.png'
boxplot_path = '../COCO/kb/charts/boxplot.png'


def calcBins(data):
    return np.arange(min(data), max(data), 5)


def createBoxPlot(sup):
    fig, ax2 = plt.subplots(1, 1, figsize=(5, 4))
    ax2.set_title('Support Distribution')
    ax2.boxplot(sup, vert=False, whis=0.50)
    plt.savefig(boxplot_path)


def createHistogram(sup):
    bins = calcBins(sup)
    plt.figure(figsize=(5, 4))
    plt.hist(sup, bins=bins)
    plt.ylabel('#Histograms')
    plt.xlabel('Support')
    plt.savefig(hist_path)


def createHistPlot(data, nameLabel, autoBins, distplot_path):
    bins = 'auto' if autoBins else calcBins(data)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.distplot(data, bins=bins, axlabel=nameLabel)
    plt.savefig(distplot_path)


if __name__ == "__main__":

    if not os.path.isdir(charts_path):
        os.makedirs(charts_path)

    with open(json_path_kb_histograms, 'r') as f:
        json_data = json.load(f)

    sup = [pro['sup'] for pro in json_data.values()]
    ent = [pro['entropy'] for pro in json_data.values()]

    # support histogram plot
    createHistPlot(sup, 'Support', False, support_distplot_path)

    # entropy histogram plot
    createHistPlot(ent, 'Entropy', False, entropy_distplot_path)

    # support histogram
    createHistogram(sup)

    # support boxPlot
    createBoxPlot(sup)

    print("Graphing Completed")

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

json_path_kb_histograms = '../COCO/kb/pairwiseKB.json'
charts_path = '../COCO/kb/charts'
distplot_path = '../COCO/kb/charts/distplot.png'
hist_path = '../COCO/kb/charts/hist.png'
boxplot_path = '../COCO/kb/charts/boxplot.png'

if __name__ == "__main__":

    if not os.path.isdir(charts_path):
        os.makedirs(charts_path)

    with open(json_path_kb_histograms, 'r') as f:
        json_data = json.load(f)

    sup = [pro['sup'] for pro in json_data.values()]
    bins = np.arange(min(sup), max(sup), 5)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.distplot(sup, bins=bins, axlabel='Support')
    plt.savefig(distplot_path)

    plt.figure(figsize=(5, 4))
    plt.hist(sup, bins=bins)
    plt.ylabel('#Histograms')
    plt.xlabel('Support')
    plt.savefig(hist_path)

    fig, ax2 = plt.subplots(1, 1, figsize=(5, 4))
    ax2.set_title('Support Distribution')
    ax2.boxplot(sup, vert=False, whis=0.50)
    plt.savefig(boxplot_path)

    print("Done")

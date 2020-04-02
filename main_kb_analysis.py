import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import kb_clean_pairwise_json_path

json_path_kb_histograms = '../COCO/kb/pairwiseKB.json'
charts_path = '../COCO/kb/charts'
support_distplot_path = '../COCO/kb/charts/support_distplot.png'
entropy_distplot_path = '../COCO/kb/charts/entropy_distplot.png'
hist_path = '../COCO/kb/charts/hist.png'
boxplot_path = '../COCO/kb/charts/boxplot.png'


def calcBins(data):
    # return np.arange(min(data), max(data), 5)
    return np.round(np.linspace(min(data), 500, 100))  # max(data),500))


def createBoxPlot(sup):
    fig, ax2 = plt.subplots(1, 1, figsize=(5, 4))
    ax2.set_title('Support Distribution')
    ax2.boxplot(sup, vert=False, whis=0.50)
    plt.savefig(boxplot_path)


def createHistogram(sup):
    bins = np.round(np.linspace(min(sup), 500, 50))  # calcBins(sup)
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

    #cleanKB = {k: v for k, v in json_data.items() if v['entropy'] <= 1 and v['sup'] > 50}

    #with open(kb_clean_pairwise_json_path, "w") as f:
    #    json.dump({k: v for k, v in cleanKB.items()}, f)

    # cleanKB= [k[v] for k,v in json_data.items() if v['entropy']<=1.5 and v['entropy']>1.3 and v['sup']>50]

    # support histogram plot
    # createHistPlot(sup, 'Support', False, support_distplot_path)

    # entropy histogram plot
    # createHistPlot(ent, 'Entropy', False, entropy_distplot_path)

    # support histogram
    # createHistogram(sup)

    # plt.figure(figsize=(10, 6))
    # values, base = np.histogram(sup, bins=200)
    # plt.plot(values, base[:-1], c='red', linestyle='', marker='o', markersize=2, markerfacecoloralt='tab:red')
    # plt.savefig(hist_path)

    # plt.figure(figsize=(10, 6))
    # values, base = np.histogram(sup, bins=200)
    # values = values[::-1]
    # base = base[::-1]
    # evaluate the cumulative
    # cumulative = np.cumsum(np.sort(sup))
    # plot the cumulative function

    sup = np.array(sup)
    ent = np.array(ent)
    #mask = (ent <= 1) & (sup >= 50)
    #sup = sup[mask]
    #ent = ent[mask]

    plt.figure(figsize=(10, 6))
    y = np.sort(sup)
    plt.plot(np.arange(len(y)), np.log10(y), c='red', linestyle='', marker='o', markersize=2,
             markerfacecoloralt='tab:red')
    plt.xlabel("data")
    plt.ylabel("Log(sup)")
    plt.grid()
    plt.savefig('../COCO/kb/charts/points_logsup.png')

    plt.figure(figsize=(10, 6))
    y = np.sort(sup)
    plt.plot(np.arange(len(y[:])), y[:], c='red', linestyle='', marker='o', markersize=2, markerfacecoloralt='tab:red')
    plt.xlabel("data")
    plt.ylabel("sup")
    plt.grid()
    plt.savefig('../COCO/kb/charts/points_sup.png')

    plt.figure(figsize=(10, 6))
    y = np.sort(ent)
    plt.plot(np.arange(len(y)), y, c='red', linestyle='', marker='o', markersize=2,
             markerfacecoloralt='tab:red')
    plt.xlabel("data")
    plt.ylabel("entropy")
    plt.grid()
    plt.savefig('../COCO/kb/charts/points_entropy.png')

    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(sup), ent, c='red', linestyle='', marker='o', markersize=2,
             markerfacecoloralt='tab:red')
    plt.xlabel("Log(sup)")
    plt.ylabel("entropy")
    plt.grid()
    plt.savefig('../COCO/kb/charts/logsup_entropy.png')

    # plt.figure(figsize=(10, 6))
    # y = np.sort(ent)
    # # y = y[y>100]
    # plt.scatter(sup, ent, c='red')
    # plt.grid()
    # plt.savefig(entropy_distplot_path)

    # support boxPlot
    # createBoxPlot(sup)

    print("Graphing Completed")

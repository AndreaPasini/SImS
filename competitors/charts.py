import os

import numpy as np
from matplotlib import pyplot as plt, ticker
from matplotlib.ticker import MaxNLocator


# sims_df, sims_df_agg, kmed_df
def plot_comparison_both(mink, maxk, dfs, suffix2, output_path):
    fig, ax = plt.subplots(1, 2, figsize=[9, 3], sharey=True)
    # Coverage of the two methods
    colors = ['#33a02c', '#1f78b4', '#ff7f00']
    markerfacecolors = ['#b2df8a', '#a6cee3', '#fc8d62']
    
    for i, (name, df) in enumerate(dfs.items()):
        ax[0].plot(np.arange(mink, maxk + 1),
                   df.loc[df['N. graphs'] >= mink]['Coverage'], label=name,
                   marker='o', markersize='4', color=colors[i], markerfacecolor=markerfacecolors[i])

    ax[0].set_xlabel('# graphs (k)')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylabel('coverage')
    ax[0].grid(axis='y', which="major", linewidth=1)
    ax[0].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')

    for i, (name, df) in enumerate(dfs.items()):
        ax[1].plot(np.arange(mink, maxk + 1),
                   df.loc[df['N. graphs'] >= mink]['Diversity'], label=name,
                   marker='o', markersize='4', color=colors[i], markerfacecolor=markerfacecolors[i])

    ax[1].set_xlabel('# graphs (k)')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel('diversity')
    ax[1].grid(axis='y', which="major", linewidth=1)
    ax[1].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')
    ax[1].legend(bbox_to_anchor=(1.6, 0.4), loc="lower right")
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_tick_params(labelbottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'evaluationBoth{suffix2}.eps'), bbox_inches='tight')

# sims_df, sims_df_agg, kmed_df
def plot_comparison_coverage(mink, maxk, dfs, suffix2, output_path):
    fig, ax = plt.subplots(1, 2, figsize=[9, 3], sharey=True)
    
    colors = ['#33a02c', '#1f78b4', '#ff7f00']
    markerfacecolors = ['#b2df8a', '#a6cee3', '#fc8d62']

    # Coverage of the two methods
    
    for i, (name, df) in enumerate(dfs.items()):
        ax[0].plot(np.arange(mink, maxk + 1),
                   df.loc[df['N. graphs'] >= mink]['Coverage'], label=name,
                   marker='o', markersize='4', color=colors[i], markerfacecolor=markerfacecolors[i])

    for i, (name, df) in enumerate(dfs.items()):
        ax[1].plot(np.arange(mink, maxk + 1),
                   df.loc[df['N. graphs'] >= mink]['Coverage-overlap'], label=name,
                   marker='o', markersize='4', color=colors[i], markerfacecolor=markerfacecolors[i])

    ax[0].set_xlabel('# graphs (k)')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylabel('coverage')
    ax[0].grid(axis='y', which="major", linewidth=1)
    ax[0].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')
    ax[1].set_xlabel('# graphs (k)')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel('coverage degree')
    ax[1].grid(axis='y', which="major", linewidth=1)
    ax[1].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')
    ax[1].legend(bbox_to_anchor=(1.6, 0.4), loc="lower right")
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    ax[1].yaxis.set_tick_params(labelbottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'evaluationCoverage{suffix2}.eps'), bbox_inches='tight')


def plot_comparison_diversity(mink, maxk, dfs, suffix2, output_path):
    fig, ax = plt.subplots(1, 2, figsize=[9, 3], sharey=True)
    
    colors = ['#33a02c', '#1f78b4', '#ff7f00']
    markerfacecolors = ['#b2df8a', '#a6cee3', '#fc8d62']


    ax[0].set_xlabel('# graphs (k)')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylabel('node diversity')
    ax[0].grid(axis='y', which="major", linewidth=1)
    ax[0].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')

    for i, (name, df) in enumerate(dfs.items()):
        ax[0].plot(np.arange(mink, maxk + 1),
                   df.loc[df['N. graphs'] >= mink]['Diversity'], label=name,
                   marker='o', markersize='4', color=colors[i], markerfacecolor=markerfacecolors[i])

    for i, (name, df) in enumerate(dfs.items()):
        ax[1].plot(np.arange(mink, maxk + 1),
                   df.loc[df['N. graphs'] >= mink]['Diversity-ne'], label=name,
                   marker='o', markersize='4', color=colors[i], markerfacecolor=markerfacecolors[i])

    ax[1].set_xlabel('# graphs (k)')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel('edge diversity')
    ax[1].grid(axis='y', which="major", linewidth=1)
    ax[1].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')
    ax[1].legend(bbox_to_anchor=(1.6, 0.4), loc="lower right")
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_tick_params(labelbottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'evaluationDiversity{suffix2}.eps'), bbox_inches='tight')
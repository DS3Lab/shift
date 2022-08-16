import os
import pandas as pd
import click
import yaml
from yaml import Loader
from shift.layers.search.searcher import search

import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm

orders = {
        "Enumeration": 0,
        "MetaLearned (k_m=1)": 1,
        "Random (k=1)": 2,
        "Random (k=2)": 3,
        "task2vec": 4,
        "shift (SHiFT)": 5,
        "leep": 6,
    }

def create_line_plot(df, filename, title):
    sns.set(
        font="DejaVu Sans",
        context="paper",
        style="whitegrid",
        font_scale=3,
        rc={'figure.figsize':(8,8)},
    )
    sns.color_palette("colorblind")
    # keep min max
    df = df.groupby('strategy').agg({'best_ft_acc':['mean','min','max'], 'sum_ft_time':['mean','min','max']})
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    df.sort_values(by=['strategy'],key=lambda x: x.map(orders), inplace=True)
    ax = sns.scatterplot(x="sum_ft_time_mean", y="best_ft_acc_mean", hue='strategy', data=df, s=300)
    random_subdf = df[df['strategy'].str.contains('Random')]
    for idx, row in enumerate(random_subdf.iterrows()):
        plt.plot( (row[1]['sum_ft_time_min'],row[1]['sum_ft_time_mean']), (row[1]['best_ft_acc_mean'], row[1]['best_ft_acc_mean']), color=f'C{row[0]}', alpha=0.5, lw=3)
        plt.plot( (row[1]['sum_ft_time_mean'],row[1]['sum_ft_time_max']), (row[1]['best_ft_acc_mean'], row[1]['best_ft_acc_mean']), color=f'C{row[0]}', alpha=0.5, lw=3)
        plt.plot( (row[1]['sum_ft_time_mean'],row[1]['sum_ft_time_mean']), (row[1]['best_ft_acc_min'], row[1]['best_ft_acc_mean']), color=f'C{row[0]}', alpha=0.5, lw=3)
        plt.plot( (row[1]['sum_ft_time_mean'],row[1]['sum_ft_time_mean']), (row[1]['best_ft_acc_mean'], row[1]['best_ft_acc_max']), color=f'C{row[0]}', alpha=0.5, lw=3)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Total FT Time (h)")
    ax.set_ylabel("Best FT Time Accuracy")

    handles, labels = ax.get_legend_handles_labels()
    for idx, label in enumerate(labels):
        if 'MetaLearned' in label:
            labels[idx] = 'Q7'
        elif label == 'task2vec':
            labels[idx] = 'Q5'
        elif 'shift' in label:
            labels[idx] = 'Q2'
    plt.legend(handles=handles, labels=labels, markerscale=3)
    plt.savefig(f"figures/shift/search_{filename}_scatter.png", bbox_inches='tight', pad_inches=0)
    plt.close()

@click.command()
@click.option('--tasks', default='tasks', help='Tasks Folder')
@click.option('--repeat', default=30, help='The number of times the search be repeated')
def run_plot(tasks, repeat):
    task_path = tasks
    tasks = [x for x in os.listdir(tasks) if x.endswith('.yml')]
    for task in tasks:
        task = os.path.join(task_path, task)
        command = "python plotter/vldb_revision/compare_strategies.py --task {} --repeat {}".format(task, repeat)

        gpu_request_command = " -R \"rusage[mem=22000,ngpus_excl_p=1]\" "
        euler_command = "bsub -n 4 -W 24:00 "+gpu_request_command + command
        os.system(euler_command)

if __name__=="__main__":
    run_plot()
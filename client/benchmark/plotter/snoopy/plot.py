import pandas as pd
import click
import yaml
from yaml import Loader
from shift.layers.search.searcher import search
from shift.layers.plotter.bar_plot import create_bar_plot

@click.command()
@click.option('--task', default='tasks/cifar_vtab.yml', help='Name of the Dataset')
@click.option('--repeat', default=30, help='The number of times the search be repeated')
def run_plot(task, repeat):
    with open(task, "r") as f:
        config = yaml.load(f, Loader=Loader)
    results = []
    for i in range(repeat):
        results.extend(search(config))
    df = pd.DataFrame(results)
    create_bar_plot(df, config['target']['name'])

if __name__=="__main__":
    run_plot()
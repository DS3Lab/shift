import ast
import math
import seaborn as sns
import click
import pandas as pd
from libparser.parser import Parser
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shift.tasks.mapping import shift_to_aftp_map

parser = Parser()


def get_lowerbound(value, classes):
    return ((classes - 1.0)/float(classes)) * (1.0 - math.sqrt(max(0.0, 1 - ((float(classes) / (classes - 1.0)) * value))))


@click.command()
@click.option('--raw')
@click.option('--finetuned')
def run_snoopy_plot(raw, finetuned):
    # Preparing Drawing Data
    raw_df = pd.read_csv(raw)
    ft_df = pd.read_csv(finetuned)

    # Collecting all datasets
    datasets = set()
    for row_id, row in raw_df.iterrows():
        query = parser.parse(row['stmt'])
        train_ds_name = query['select']['trained_on']['datasets'][0]['name']
        datasets.add(train_ds_name)

    for ds in datasets:
        logger.info(f"Drawing {ds}")
        sub_raw_df = raw_df[raw_df['stmt'].str.contains(ds)]
        ft_ds_name = shift_to_aftp_map[ds]['name']
        ft_train_slice = shift_to_aftp_map[ds]['train_slice']
        num_classes = shift_to_aftp_map[ds]['num_classes']
        # we will need to validate if this ft_label_name is standard or not
        config = shift_to_aftp_map[ds]['config']
        ft_subset = ft_df[
            (ft_df['train_dataset_name'] == ft_ds_name) & (
                ft_df['train_slice'] == ft_train_slice) & (ft_df['configs'] == str(config))]

        full_results = ast.literal_eval(sub_raw_df['models'].values[0])[
            'Euclidean NN']
        models = [x['json_model']['hf_name'] for x in full_results]
        # now find finetune accuracy
        ft_acc = ft_subset[ft_subset['model_identifier'].isin(models)]
        ft_acc = ft_acc.sort_values(by='test_accuracy', ascending=False)
        minimal_proxy_error = full_results[0]['err']

        # now start to draw
        sns.set(font="DejaVu Sans", style="white", context="paper",
                font_scale=3, rc={'figure.figsize': (9, 6)})
        
        # draw a scatter plot, x is indirect proxy accuracy, y is finetune accuracy
        ax = sns.scatterplot(x='model_identifier',
                             y='test_accuracy',
                             color='b',
                             data=ft_acc)
        sns.despine()
        ax.set(xticklabels=[])
        ax.axhline(1 - get_lowerbound(minimal_proxy_error, num_classes),
                   label='Estimated Best Accuracy', color='r')
        ax.set_xlabel('Model')
        ax.set_ylabel('Finetune Accuracy')
        """
        if ft_ds_name=='smallnorb':
            label_name = config['label_name']
            ax.set_title(f'Finetune Accuracy and Estimated Best Accuracy (smallnorb/{label_name})')
        else:
            if 'preprocess' not in config or config['preprocess'] == 'empty':
                ax.set_title(f'Finetune Accuracy and Estimated Best Accuracy ({ft_ds_name})')
            else:
                preprocessing = config['preprocess']
                ax.set_title(f'Finetune Accuracy and Estimated Best Accuracy ({preprocessing})')
        """
        handles, labels = ax.get_legend_handles_labels()
        scatter_lgd = Line2D([0], [0], marker='o', color='b',
                             label='Finetune Accuracy', markerfacecolor='b', markersize=15)
        plt.legend(handles=handles, labels=labels)
        # save to file
        ax.get_figure().savefig(
            f'figures/snoopy/finetune_acc-{ds}.png', bbox_inches='tight',
            pad_inches=0,
        )
        logger.info(
            f"Done, file saved to figures/snoopy/finetune_acc-{ds}.png")
        plt.clf()


if __name__ == "__main__":
    run_snoopy_plot()

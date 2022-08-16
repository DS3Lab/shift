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
    drawing_data = []
    for row_id, row in raw_df.iterrows():
        query = parser.parse(row['stmt'])
        train_ds_name = query['select']['trained_on']['datasets'][0]['name']
        datasets.add(train_ds_name)

    for ds in datasets:
        
        sub_raw_df = raw_df[raw_df['stmt'].str.contains(ds)]
        ft_ds_name = shift_to_aftp_map[ds]['name']
        ft_train_slice = shift_to_aftp_map[ds]['train_slice']
        num_classes = shift_to_aftp_map[ds]['num_classes']
        # we will need to validate if this ft_label_name is standard or not
        config = shift_to_aftp_map[ds]['config']
        ft_subset = ft_df[
            (ft_df['train_dataset_name'] == ft_ds_name) & (
                ft_df['train_slice'] == ft_train_slice) & (ft_df['configs'] == str(config))]

        full_results = ast.literal_eval(sub_raw_df['models'].values[0])['Euclidean NN']
        models = [x['json_model']['hf_name'] for x in full_results]
        # now find finetune accuracy
        ft_acc = ft_subset[ft_subset['model_identifier'].isin(models)]
        ft_acc = ft_acc.sort_values(by='test_accuracy', ascending=False)
        minimal_proxy_error = full_results[0]['err']
        best_estimated_acc = 1 - get_lowerbound(minimal_proxy_error, num_classes)
        best_finetune_acc = ft_acc['test_accuracy'].max()
        drawing_data.append({
            'snoopy': best_estimated_acc - best_finetune_acc,
        })
    drawing_data = pd.DataFrame(drawing_data)
    drawing_data.to_csv("snoopy_histogram.csv", index=False)
    sns.set(font="DejaVu Sans", context="paper", style="whitegrid", font_scale=2)
    plt.hist(drawing_data['snoopy'].tolist(), 10, facecolor='C0', alpha=0.6, label='Snoopy')
    plt.xlabel('Delta to Max Fine-Tuned Accuracy')
    plt.ylabel('# of Datasets')
    plt.legend()
    plt.savefig("figures/snoopy/histogram.png", bbox_inches = 'tight', pad_inches = 0.02)

if __name__ == "__main__":
    run_snoopy_plot()

import ast
import seaborn as sns
import click
import pandas as pd
from libparser.parser import Parser
from loguru import logger
import matplotlib.pyplot as plt

from shift.tasks.mapping import shift_to_aftp_map

parser = Parser()


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
        drawing_data = []
        sub_raw_df = raw_df[raw_df['stmt'].str.contains(ds)]
        ft_ds_name = shift_to_aftp_map[ds]['name']
        ft_train_slice = shift_to_aftp_map[ds]['train_slice']
        # we will need to validate if this ft_label_name is standard or not
        config = shift_to_aftp_map[ds]['config']
        ft_subset = ft_df[
            (ft_df['train_dataset_name'] == ft_ds_name) & (
                ft_df['train_slice'] == ft_train_slice) & (ft_df['configs'] == str(config))]
        # someday we may want average instead of the first one
        models = ast.literal_eval(sub_raw_df['models'].values[0])[
            'Euclidean NN']
        for model_id, model in enumerate(models):
            model_name = model['json_model']['hf_name']
            indirect_proxy_acc = 1 - model['err']
            # now find finetune accuracy
            ft_acc = ft_subset[ft_subset['model_identifier']
                               == model_name]['val_accuracy'].mean()
            drawing_data.append({
                'model_identifier': model_name,
                '1nn_accuracy': indirect_proxy_acc,
                'ft_accuracy': ft_acc,
            })

        drawing_data = pd.DataFrame(drawing_data)
        drawing_data.to_csv(f"figures/draw_data/{ds}.csv", index=False)        

if __name__ == "__main__":
    run_snoopy_plot()

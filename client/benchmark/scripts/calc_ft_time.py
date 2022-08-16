import pandas as pd

train_ds_names = ['glue/cola', 'glue/sst2']
ft_full = True

df = pd.read_csv(".cache/ftrecord.csv")
models = pd.read_csv(".cache/shift_models/text_models.csv")
models = models['model_identifier'].values.tolist()
for train_ds_name in train_ds_names:
    subset = df[df['train_dataset_name']==train_ds_name]
    if ft_full:
        subset = subset[subset['train_slice']!=':800']
        subset = subset[subset['model_identifier'].isin(models)]
        # dedup
        subset = subset.drop_duplicates(subset=['model_identifier', 'train_split'])
        # sum the time
        total_secs = subset['elapsed_time'].sum()
        total_secs = total_secs * 65 / len(subset)
        print(f"{train_ds_name}: {total_secs}")
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

in_dir = ".cache/nlp"
ft_records = pd.read_csv(os.path.join(".cache", "ftrecord.csv"))

def sort_by_settings(df_series):
    queries = ['Q2', 'Q3', 'Q4']
    settings = ['Baseline', 'SHiFT']
    orders = []
    for each in queries:
        for each_settings in settings:
            orders.append(each+"_"+each_settings)
    order = df_series.apply(lambda x: orders.index(x))
    return order

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def get_best_model_from_shift(filename):
    with open(os.path.join(in_dir, filename),"r") as fp:
        if not 'linear' in filename:
            models = json.load(fp)['models']['Euclidean NN']
        else:
            models = json.load(fp)['models']['Linear(learning_rate=0.01 num_epochs=2000)']
    return models[0]['json_model']['hf_name']

def aggregate():
    tasks = ['cola', 'sst']
    draw_df = []
    for task in tasks:
        task_name = 'glue/cola' if task == 'cola' else 'glue/sst2'
        task_records = ft_records[(ft_records['train_dataset_name']==task_name) & (ft_records['train_slice']==":800")]

        NN_baseline=f'{task}_8gpu_nosh.json'
        NN_sh = f'{task}_8gpu_sh.json'
        linear_baseline=f'{task}_linear_nosh.json'
        linear_sh=f'{task}_linear_sh.json'
        hybrid_baseline=f'{task}_hybrid_nosh.json'
        hybrid_sh=f'{task}_hybrid_sh.json'

        best_nn_baseline = get_best_model_from_shift(NN_baseline)
        best_nn_sh = get_best_model_from_shift(NN_sh)
        best_linear_sh = get_best_model_from_shift(linear_sh)
        best_linear_baseline = get_best_model_from_shift(linear_baseline)
        best_hybrid_baseline = get_best_model_from_shift(hybrid_baseline)
        best_hybrid_sh = get_best_model_from_shift(hybrid_sh)
        
        best_nn_baseline_record = task_records[task_records['model_identifier']==best_nn_baseline]
        best_linear_baseline_record = task_records[task_records['model_identifier']==best_linear_baseline]
        best_nn_sh_record = task_records[task_records['model_identifier']==best_nn_sh]
        best_linear_sh_record = task_records[task_records['model_identifier']==best_linear_sh]
        best_hybrid_baseline_record = task_records[task_records['model_identifier']==best_hybrid_baseline]
        best_hybrid_sh_record = task_records[task_records['model_identifier']==best_hybrid_sh]
        
        draw_df.append({
            "reader": "GLUE/COLA" if task=='cola' else "GLUE/SST-2",
            "settings": "Baseline",
            "query": "Q2",
            "accuracy": best_nn_baseline_record['test_accuracy'].mean(),
        })
        draw_df.append({
            "reader": "GLUE/COLA" if task=='cola' else "GLUE/SST-2",
            "settings": "SHiFT",
            "query": "Q2",
            "accuracy": best_nn_sh_record['test_accuracy'].mean(),
        })
        draw_df.append({
            "reader": "GLUE/COLA" if task=='cola' else "GLUE/SST-2",
            "settings": "Baseline",
            "query": "Q3",
            "accuracy": best_linear_baseline_record['test_accuracy'].mean(),
        })
        draw_df.append({
            "reader": "GLUE/COLA" if task=='cola' else "GLUE/SST-2",
            "settings": "SHiFT",
            "query": "Q3",
            "accuracy": best_linear_sh_record['test_accuracy'].mean(),
        })
        draw_df.append({
            "reader": "GLUE/COLA" if task=='cola' else "GLUE/SST-2",
            "settings": "Baseline",
            "query": "Q4",
            "accuracy": best_hybrid_baseline_record['test_accuracy'].mean(),
        })
        draw_df.append({
            "reader": "GLUE/COLA" if task=='cola' else "GLUE/SST-2",
            "settings": "SHiFT",
            "query": "Q4",
            "accuracy": best_hybrid_sh_record['test_accuracy'].mean(),
        })

    return pd.DataFrame(draw_df)

def draw_byreaders(draw_df):
    counter = 0
    lw = 10
    f, axes = plt.subplots(1, 2, figsize=(8, 5))
    readers = ['GLUE/COLA', 'GLUE/SST-2']

    draw_sub_df = draw_df

    for reader in readers:
        task_name = 'glue/cola' if reader =='GLUE/COLA' else 'glue/sst2'
        task_records = ft_records[(ft_records['train_dataset_name']==task_name) & (ft_records['train_slice']==":800")]
        sub_df = draw_sub_df[draw_sub_df['reader'] == reader]
        sub_df['order'] = sub_df.apply(
            lambda x: x['query']+"_"+x['settings'], axis=1)
        sub_df = sub_df.sort_values("order", key=lambda x: sort_by_settings(x))

        ax = axes.flat[counter]
        ax = sns.barplot(
            ax=ax,
            data=sub_df,
            x="query",
            y='accuracy',
            hue='settings',
            palette=["C1", "C0"]
        )
        min_acc = task_records['test_accuracy'].min()
        max_acc = task_records['test_accuracy'].max()
        ax.set_ylim(0, 1)
        ax.set_title(reader)

        ax.axhline(min_acc, linestyle='--', label='FT Worst', lw=2, color='black')
        ax.axhline(max_acc, linestyle='--', label='FT Best', lw=2, color='C9')

        ax.set_xlabel("Method")
        # ax.set_xticklabels(ax.get_xticklabels(),rotation = 40)
        if counter == 0:
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("")
        elif counter == 1:
            ax.set_xlabel("Query")
            ax.xaxis.set_label_coords(-.15, -0.1)
            ax.set_ylabel("")
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
        counter = counter + 1
    handles, labels = axes.flat[0].get_legend_handles_labels()
    ncol = 4
    lgd = f.legend(flip(handles, ncol), flip(
        labels, ncol), loc='lower center', ncol=ncol)
    lgd_lines = lgd.get_lines()
    lgd_lines[0].set_linewidth(3)
    lgd_lines[1].set_linewidth(3)
    # lgd_lines[0].set_linestyle(":")

    for i, ax in enumerate(axes.flat):
        ax.legend([], [], frameon=False)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.2, bottom=0.2)
    plt.savefig("figures/nlp_finetune_vs_shift_accuracy_bar.pdf",
                bbox_inches='tight',
                bbox_extra_artists=(lgd,),
                pad_inches=0)
    plt.show()
if __name__=="__main__":
    # plot()
    draw_df = aggregate()
    draw_byreaders(draw_df)
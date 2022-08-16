import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


class_num = [8130, 720, 1579, 237, 240]
in_dir = ".cache/incremental"

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def aggregate():
    per_model_accuracy={}
    
    
    final_results = []
    with open(os.path.join(in_dir, "distributing.json"),"r") as fp:
        all_distributing = json.load(fp)
        all_models = all_distributing['models']['Euclidean NN']
        all_models = [x['json_model']['hf_name'] for x in all_models]
    all_results = []
    for m in all_models:
        per_model_accuracy[m] = []
    for i in range(len(class_num)):
        with open(os.path.join(in_dir,"perclass", f"{i}.json"),"r") as fp:
            all_results.append(json.load(fp)['models']['Euclidean NN'])
    
    for i in range(len(class_num)):
        results = all_results[i]
        for m in all_models:
            related_row = [x for x in results if x['json_model']['hf_name']==m]
            per_model_accuracy[m].append(related_row[0]['err']*class_num[i])
    
    for m in all_models:
        final_results.append(
            {"model":m, "err":sum(per_model_accuracy[m])/sum(class_num)}
        )
    df = pd.DataFrame(final_results)
    df.sort_values(by="err", inplace=True)
    df.to_csv(os.path.join(in_dir, "all_models_origin.csv"), index=False)

def plot():
    original_models = "Vemi/orchid219_ft_vit-large-patch16-224-in21k-finetuned-eurosat"
    with open(os.path.join(in_dir, "distributing.json"),"r") as fp:
        distributed_models = json.load(fp)
        distributed_model = distributed_models['models']['Euclidean NN'][0]['json_model']['hf_name']
    with open(os.path.join(in_dir, "non-distributing.json"),"r") as fp:
        nondistributed_models = json.load(fp)
        nondistributed_model = nondistributed_models['models']['Euclidean NN'][0]['json_model']['hf_name']

    ft_records = pd.read_csv(os.path.join(".cache", "ftrecord.csv"))
    diabetic_records = ft_records[(ft_records['train_dataset_name']=='diabetic_retinopathy_detection') & (ft_records['train_slice']==":100%")]

    original_ft_records = diabetic_records[(diabetic_records['model_identifier']==original_models) & (diabetic_records['train_slice']==":100%")]
    distributed_ft_records = diabetic_records[(diabetic_records['model_identifier']==distributed_model) & ((diabetic_records['train_slice']==":100%"))]
    nondistributed_ft_records = diabetic_records[(diabetic_records['model_identifier']==nondistributed_model) & ((diabetic_records['train_slice']==":100%"))]

    draw_df = [{
        "name": "Fully Shuffling",
        "accuracy": original_ft_records['test_accuracy'].mean(),
        
    },{
        "name": "w/o Distributing",
        "accuracy": nondistributed_ft_records['test_accuracy'].mean(),
        
    },{
        "name": "Uniform Distributing",
        "accuracy": distributed_ft_records['test_accuracy'].mean(),
        
    }]
    draw_df = pd.DataFrame(draw_df)
    ax = sns.barplot(
        data=draw_df,
        x="name",
        y='accuracy',
    )
    ax.set_xlabel("Method")
    ax.set_ylim(0.6, 0.8)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Query")
    plt.savefig("figures/shift/incremental.pdf",
                bbox_inches='tight',
                pad_inches=0)

if __name__=="__main__":
    plot()
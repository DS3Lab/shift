import os
import pandas as pd
from typing import List
from supabase import create_client, Client
from dstool.class_utils import singleton
from loguru import logger
from shift.tasks.mapping import shift_to_aftp_map

@singleton
class ShiftAPI():
    def __init__(self, fpath=None) -> None:
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_KEY")
        self.df = None
        self.load_cache(fpath)
        if self.url is not None:
            self.supabase: Client = create_client(self.url, self.key)

    def _cache_ftrecord(self):
        data = self.supabase.table("ftrecord").select(
            "*").limit(999999).execute()
        df = pd.DataFrame(data.data)
        df.to_csv(os.path.join(".cache", "ftrecord.csv"))
        self.df = df
        logger.info("Cached ftrecord refreshed, current size: {}".format(len(df)))

    def load_cache(self, fpath=None):
        if fpath is None:
            fpath = os.path.join(".cache", "ftrecord.csv")
        if self.df is None and os.path.exists(fpath):
            self.df = pd.read_csv(fpath)
        self.df.reset_index(inplace=True, drop=True)
        
    def get_ft_all_datasets(self):
        return list(self.df["train_dataset_name"].unique())

    def get_ft_single_model(self, dataset_name: str, model_name: str):
        subset = self.df[(self.df["train_dataset_name"] == dataset_name) & (
            self.df["model_name"] == model_name)]
        return subset

    def get_ft_multiple_models(
        self,
        dataset_name: str,
        model_name: List[str]
    ):
        if dataset_name.startswith("vtab"):
            target_dataset = shift_to_aftp_map[dataset_name]
            subset = self.get_finetune_performance(target_dataset)
            subset = subset[subset['model_identifier'].isin(model_name)]
        else:
            subset = self.df[(self.df["train_dataset_name"] == dataset_name) & (
                self.df["model_identifier"].isin(model_name))]
        return subset

    def get_known_models(self, target_dataset):
        if target_dataset['name'].startswith("vtab"):
            target_dataset = shift_to_aftp_map[target_dataset['name']]
        subset = self.get_finetune_performance(target_dataset)
        return list(subset["model_identifier"].unique())

    def get_total_ft_time(self, dataset_name, model_name):
        subset = self.get_ft_multiple_models(dataset_name, model_name)
        return subset["elapsed_time"].sum()

    def get_best_model(self, dataset_name, models):
        if dataset_name.startswith("vtab"):
            subset = self.get_aftp_task_df(shift_to_aftp_map[dataset_name])
            subset = subset[subset['model_identifier'].isin(models)]
        else:
            subset = self.get_ft_multiple_models(dataset_name, models)
        if len(subset) > 0:
            best_model_id = subset['test_accuracy'].idxmax()
            best_row = self.df.iloc[[best_model_id]]
            return best_row['model_identifier'].values[0], best_row['test_accuracy'].values[0]
        else:
            return None, None

    def get_best_model_on_task(self, task, top_k):
        model_names = self.get_known_models(task)
        subset = self.get_ft_multiple_models(task['name'], model_names)
        subset = subset.sort_values(by="test_accuracy", ascending=False)
        candidates = subset.iloc[:top_k]
        return candidates['model_identifier'].tolist(), candidates['test_accuracy'].tolist()

    def get_finetune_performance(self, target_dataset):
        if target_dataset == None:
            return list(self.df["model_identifier"].unique())
        else:
            condition = self.df["train_dataset_name"] == target_dataset['name']
            if 'train_split' in target_dataset and target_dataset['train_split'] != 'nan':
                condition = condition & (self.df["train_split"] == target_dataset['train_split'])
            else:
                # by default set it to be train
                condition = condition & (self.df["train_split"] == 'train')
            if 'test_split' in target_dataset and target_dataset['test_split'] != 'nan':
                condition = condition & (self.df["test_split"] == target_dataset['test_split'])
            else:
                condition = condition & (self.df["test_split"] == 'test')
            if 'test_slice' in target_dataset and target_dataset['test_slice'] != 'nan':
                condition = condition & (self.df["test_slice"] == target_dataset['test_slice'])
            subset = self.df[condition]
            return subset

    def get_aftp_task_df(self, task):
        config = task['config']
        ft_ds_name = task['name']
        ft_train_slice = task['train_slice']
        ft_df = self.df

        ft_subset = ft_df[
            (ft_df['train_dataset_name'] == ft_ds_name) & (
                ft_df['train_slice'] == ft_train_slice) & (ft_df['configs'] == str(config))]
        return ft_subset

if __name__=="__main__":
    api = ShiftAPI()
    api._cache_ftrecord()
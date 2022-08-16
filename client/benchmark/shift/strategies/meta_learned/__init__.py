import os
from shift.strategies._base import BaseSearchStrategy
from shift.tasks.mapping import shift_to_aftp_map
import json
from libparser.parser import Parser

class MetaLearnedSearchStrategy(BaseSearchStrategy):
    """
    MetaLearned Search Strategy
    
    Args:
        k_m: Number of best models from similiar tasks, default=1

    """
    def __init__(self, k_m=1) -> None:
        super().__init__()
        self.k_m = k_m
        self.name=f'MetaLearned (k_m={k_m})'
        self.shift_folder = os.path.join(".cache", "shift_vtab_nn")
        self.parser = Parser()
        self._load_shift_result()

    def _load_shift_result(self):
        shift_results = {}
        shift_results_files = [x for x in os.listdir(self.shift_folder) if x.endswith(".json")]
        for result in shift_results_files:
            with open(os.path.join(self.shift_folder, result), "r") as fp:
                result = json.load(fp)
                stmt = self.parser.parse(result['stmt'])
                train_ds_name = stmt['select']['trained_on']['datasets'][0]['name']
                shift_results[train_ds_name] = result['models']['Euclidean NN']
        self.shift_results = shift_results

    def search(self):
        if len(self.meta) == 0:
            meta_tasks = self.shift_api.get_ft_all_datasets()
            meta_tasks.remove(self.target['name'])
        else:
            meta_tasks = self.meta
        # the result of `dataset` is still unknown
        candidates = []
        for task in meta_tasks['tasks']:
            if task['name'].startswith('vtab'):
                aftp_task = shift_to_aftp_map[task['name']]
                ft_subset = self.shift_api.get_aftp_task_df(aftp_task)
                ft_subset = ft_subset.sort_values(by="test_accuracy", ascending=False)
                best_k_models = ft_subset['model_identifier'].values[:self.k_m]
                candidates.extend(best_k_models)
            else:
                model, acc = self.shift_api.get_best_model_on_task(task, self.k_m)
                candidates.extend(model)
        
        if not self.target['name'].startswith("vtab"):
            for key in shift_to_aftp_map.keys():
                if shift_to_aftp_map[key]['name'] == self.target['name']:
                    vtab_name = key
        else:
            vtab_name = self.target['name']
        shift_result = self.shift_results[vtab_name]
        related_rows = [x for x in shift_result if x['json_model']['hf_name'] in candidates]
        candidates = [x['json_model']['hf_name'] for x in related_rows][:1]
        return candidates, 0
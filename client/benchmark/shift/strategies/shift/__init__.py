import ast
import os
import json
import numpy as np
from typing import List
from shift.strategies._base import BaseSearchStrategy

class ShiftSearchStrategy(BaseSearchStrategy):
    """
    Random search strategy - enumerate all possible models, finetune them and find the most accurate one.
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.name=f"shift ({config['name']})"
        self.shift_config = config
        self.load_file()

    def load_file(self):
        with open(os.path.join(".cache", "shift", self.shift_config['id']+".json"), "r") as fp:
            content = json.load(fp)
        metrics = list(content['models'].keys())
        assert len(metrics) == 1, "Shift search strategy only supports one metric"
        self.metric = metrics[0]
        self.models = content['models'][self.metric]

    def search(self):
        results = self.models[:self.shift_config['k']]
        results = [x['json_model']['hf_name'] for x in results]
        return results, 0
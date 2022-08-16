import numpy as np
from shift.strategies._base import BaseSearchStrategy

class RandomSearchStrategy(BaseSearchStrategy):
    """
    Random search strategy - enumerate all possible models, finetune them and find the most accurate one.
    """
    def __init__(self, k) -> None:
        super().__init__()
        self.name=f'Random (k={k})'
        self.k = k

    def search(self):
        return np.random.choice(self.candidate_models, self.k, replace=False), 0
from typing import List
from shift.strategies._base import BaseSearchStrategy

class EnumerationSearchStrategy(BaseSearchStrategy):
    """
    Enumeration search strategy - enumerate all possible models, finetune them and find the most accurate one.
    """
    def __init__(self) -> None:
        super().__init__()
        self.name='Enumeration'

    def search(self):
        return self.candidate_models, 0
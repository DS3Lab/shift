from typing import Dict, List, Union

from pydantic import BaseModel
from schemas._base import ID

from .common import ClassifierWithParams, HyperbandRequest


class HyperbandStatus(BaseModel):
    id: ID
    arms: List
    errors: Dict
    current_index: Dict
    request: HyperbandRequest
    classifier: ClassifierWithParams
    pulls_performed: Union[Dict, None]
    partial_results: Union[Dict, None]
    total_size: int
    budget: int
    limit: int

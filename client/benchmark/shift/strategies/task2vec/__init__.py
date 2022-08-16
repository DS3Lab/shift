import os
import json
import numpy as np
from shift.strategies import shift
from shift.strategies._base import BaseSearchStrategy
from shift.strategies.task2vec.distance import _DISTANCES
from shift.tasks.mapping import dataset_names, ds_mapping, shift_to_aftp_map, ds_to_vtab_mapping
import itertools
from libparser.parser import Parser

class Embedding:
    def __init__(self, hessian, scale, meta=None):
        self.hessian = np.array(hessian)
        self.scale = np.array(scale)
        self.meta = meta
        
    def save(self, fpath):
        np.savez(fpath, hessian=self.hessian, scale=self.scale, meta=self.meta)

    def load(self, fpath):
        data = np.load(fpath, allow_pickle=True)
        self.hessian = data['hessian']
        self.scale = data['scale']
        self.meta = data['meta']

def pdist(embeddings, distance="cosine"):
    distance_fn = _DISTANCES[distance]
    n = len(embeddings)
    distance_matrix = np.zeros([n, n])
    if distance != "asymmetric_kl":
        for (i, e1), (j, e2) in itertools.combinations(enumerate(embeddings), 2):
            distance_matrix[i, j] = distance_fn(e1, e2)
            distance_matrix[j, i] = distance_matrix[i, j]
    else:
        for (i, e1) in enumerate(embeddings):
            for (j, e2) in enumerate(embeddings):
                distance_matrix[i, j] = distance_fn(e1, e2)
    return distance_matrix

def cdist(from_embeddings, to_embeddings, distance="cosine"):
    distance_fn = _DISTANCES[distance]
    distance_matrix = np.zeros([len(from_embeddings), len(to_embeddings)])
    for (i, e1) in enumerate(from_embeddings):
        for (j, e2) in enumerate(to_embeddings):
            if e1 is None or e2 is None:
                continue
            distance_matrix[i, j] = distance_fn(e1, e2)
    return distance_matrix

class Task2VecSearchStrategy(BaseSearchStrategy):
    """
    Task2Vec search strategy - enumerate all possible models, finetune them and find the most accurate one.
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.name=f"task2vec"
        self.task2vec_config = config
        self.cache_folder = os.path.join(".cache", "embeddings")
        self.neighbors = {}
        self.shift_folder = os.path.join(".cache", "shift_vtab_nn")
        self.parser = Parser()
        self._load_all_embeddings()
    
    def _load_all_embeddings(self):
        embeddings = []

        for i, ds in enumerate(dataset_names):
            ds = ds_mapping[i]
            embedding = Embedding(None, None, None)
            embedding.load(os.path.join(self.cache_folder, f"{ds}_embedding.npz"))
            embeddings.append(embedding)
        
        dist_matrix = pdist(embeddings, distance='normalized_cosine')
        for i, ds in enumerate(dataset_names):
            indices = np.argpartition(dist_matrix[i,:], 2)[:2]
            indices = [x for x in indices if x != i]
            assert len(indices) == 1
            nearest = indices[0]

            indices = np.argpartition(dist_matrix[i,:], 3)[:3]
            indices = [x for x in indices if x != i and x != nearest]
            assert len(indices) == 1
            second = indices[0]

            self.neighbors[ds_mapping[i]] = [ds_mapping[nearest], ds_mapping[second]]

    def search(self):
        results = []
        task_name = self.task2vec_config['task_name']
        nearest_task = self.neighbors[task_name][0]
        nearest_task = shift_to_aftp_map[ds_to_vtab_mapping[nearest_task]]
        best_model_on_nearest, acc = self.shift_api.get_best_model_on_task(nearest_task, 1)
        results.extend(best_model_on_nearest)
        return results, 70
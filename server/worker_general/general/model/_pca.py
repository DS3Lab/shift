"""
import numpy as np
from pipeline import DataType
from pipeline.model import Model, PreprocessingSpecs
from schemas.model import PCAModelConfig
from sklearn.decomposition import PCA

from .preprocessing import ImageCropResizeFlatten


class PCAModel(Model):
    def __init__(self, config: PCAModelConfig):
        self._target_image_size = config.target_image_size
        self._pca_dimension = config.pca_dimension
        self._pca = PCA(n_components=config.pca_dimension, svd_solver="full")

        # First batch of first reader is used for training
        self._train = True

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return ImageCropResizeFlatten(self._target_image_size)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        if self._train:
            self._pca.fit(features)
            self._train = False

        return self._pca.transform(features)
"""

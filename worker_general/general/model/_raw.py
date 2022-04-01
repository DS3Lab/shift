import numpy as np
from pipeline import DataType
from pipeline.model import Model, PreprocessingSpecs
from schemas.models import ImageNoOpModelConfig, ReshapeModelConfig

from .preprocessing import (
    ImageCropResize3Channels,
    ImageCropResizeFlatten,
    TextNoOpPreprocessing,
)


class ReshapeModel(Model):
    """Converts images into vectors of same size.

    Args:
        config (ReshapeModelConfig): Model configuration.
    """

    def __init__(self, config: ReshapeModelConfig):
        self._image_size = config.reshape_image_size

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return ImageCropResizeFlatten(self._image_size)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        return features


class ImageNoOpModel(Model):
    """Stores preprocessed images.

    Args:
        config (ImageNoOpModelConfig): Model configuration.
    """

    def __init__(self, config: ImageNoOpModelConfig):
        self._target_image_size = config.noop_target_image_size

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return ImageCropResize3Channels(self._target_image_size)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        return features


class TextNoOpModel(Model):
    """Stores preprocessed texts."""

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return TextNoOpPreprocessing()

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        return features

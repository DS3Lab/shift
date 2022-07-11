from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from schemas.models import FullModelConfig

from ._base import DataType, Device

__all__ = [
    "PreprocessingSpecs",
    "NullPreprocessingSpecs",
    "Model",
    "NullModel",
    "ModelFactory",
]


class PreprocessingSpecs(ABC):
    """Specifies how the data should be preprocessed by the reader such that it is
    suitable to be used by a model for inference."""

    @property
    def needs_disabled_multithreading(self) -> bool:
        """Specifies whether the preprocessing should be run in a single thread.

        Returns:
            bool: True if the preprocessing should not be run in multiple threads, False
            otherwise.
        """
        return False

    @abstractmethod
    def get_tf_preprocessing_fn(self) -> Optional[Callable]:
        """Returns the function that should be used for preprocessing in TensorFlow
        datasets.

        Returns:
            Callable, optional: TensorFlow dataset preprocessing function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_pt_preprocessing_fn(self) -> Optional[Callable]:
        """Returns the function that should be used for preprocessing in PyTorch
        datasets.

        Returns:
            Callable, optional: PyTorch dataset preprocessing function.
        """
        raise NotImplementedError


class NullPreprocessingSpecs(PreprocessingSpecs):
    """Used in places where preprocessing is not needed."""

    def get_tf_preprocessing_fn(self) -> Optional[Callable]:
        pass

    def get_pt_preprocessing_fn(self) -> Optional[Callable]:
        pass


class Model(ABC):
    """Interface that should be implemented by all models."""

    @abstractmethod
    def get_preprocessing_specs(
        self,
    ) -> PreprocessingSpecs:
        """Returns the preprocessing specification that preprocesses the data in a way
        that is suitable to be passed to the model.

        Returns:
            PreprocessingSpecs: Preprocessing specification.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Specifies what kind of data the model can embed.

        Returns:
            DataType: Type of data the model can embed.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        """Embeds the specified data.

        Args:
            features (np.ndarray): Data to be embedded.

        Returns:
            np.ndarray: Embedded data.
        """
        raise NotImplementedError


class NullModel(Model):
    """Used in places where the model is not needed."""

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return NullPreprocessingSpecs()

    @property
    def data_type(self) -> DataType:
        return DataType.UNKNOWN

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        return features


class ModelFactory(ABC):
    """Instantiates model instances given the specified parameters."""

    @staticmethod
    @abstractmethod
    def get_model(config: FullModelConfig, device: Device) -> Model:
        """Instantiates a model given the specified parameters.

        Args:
            config (FullModelConfig): Model configuration.
            device (Device): Target device, on which the model should be loaded.

        Returns:
            Model: Model instance.
        """
        raise NotImplementedError

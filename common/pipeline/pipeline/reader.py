from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from schemas.requests.reader import ReaderConfig

from ._base import DataType
from .model import PreprocessingSpecs

__all__ = ["Reader", "ReaderFactory"]


class Reader(ABC):
    """Interface that should be implemented by all readers."""

    @abstractmethod
    def __iter__(self):
        """Returns the iterator which can be iterated to get the data."""
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> Dict[str, np.ndarray]:
        """Returns next subset of the data (e.g. according to the specified batch size).
        All NumPy arrays should have the same length (= first dimension).

        Returns:
            Dict[str, np.ndarray]: Next subset of the data. Each key is a different part of the data (e.g. image, label, description, ...).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Specifies the data type of the feature with key 'READER_EMBED_FEATURE_NAME' even if this feature is not present in the dataset.

        Returns:
            DataType: Type of the feature with key 'READER_EMBED_FEATURE_NAME'.
        """
        raise NotImplementedError


class ReaderFactory(ABC):
    """Responsible for instantiating reader instances given the specified parameters."""

    @staticmethod
    @abstractmethod
    def get_reader(
        reader_config: ReaderConfig,
        batch_size: Optional[int],
        specs: PreprocessingSpecs,
    ) -> Reader:
        """Instantiates a reader given the specified parameters.

        Args:
            reader_config (ReaderConfig): Reader configuration.
            specs (PreprocessingSpecs): Preprocessing specification.
            batch_size (int, optional): Batch size; if not specified, maximal possible
                batch size (whole dataset) should be used.

        Returns:
            Reader: Reader instance.
        """
        raise NotImplementedError

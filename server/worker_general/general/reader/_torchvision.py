from time import sleep
from typing import Dict, Optional

import numpy as np
import torchvision
from pipeline import DataType
from pipeline.model import PreprocessingSpecs
from schemas.requests.reader import PTReaderConfig, QMNISTReaderConfig, USPSReaderConfig

from .._config import settings
from ._pytorch import PTReader, _DatasetWithLength, _extract_data_from_dataset


class _QMNISTDataset(_DatasetWithLength):
    def __init__(self, config: QMNISTReaderConfig, specs: PreprocessingSpecs):
        self._use_images = config.use_qmnist_images
        self._use_labels = config.use_qmnist_labels

        try:
            self._data = self._get_data(config, specs)
        except EOFError or FileNotFoundError:
            sleep(5)
            self._data = self._get_data(config, specs)

    def __getitem__(self, index: np.int64) -> Dict[str, np.ndarray]:
        return _extract_data_from_dataset(
            self._data, int(index), self._use_images, self._use_labels
        )

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def _get_data(config: QMNISTReaderConfig, specs: PreprocessingSpecs):
        return torchvision.datasets.QMNIST(
            root=settings.torch_dataset_location,
            what=config.split,
            compat=True,
            download=True,
            transform=specs.get_pt_preprocessing_fn(),
        )


class QMNISTReader(PTReader):
    """Loads and prepares the QMNIST dataset.

    Args:
        config (QMNISTReaderConfig): Reader configuration.
        specs (PreprocessingSpecs): Preprocessing specification.
        batch_size (int, optional): Batch size; if not specified, maximal possible batch size (whole dataset) is used.
    """

    def __init__(
        self,
        config: QMNISTReaderConfig,
        specs: PreprocessingSpecs,
        batch_size: Optional[int],
    ):
        self._config = config
        self._specs = specs
        super().__init__(config, batch_size)

    def _get_dataset(self) -> _DatasetWithLength:
        return _QMNISTDataset(self._config, self._specs)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


class _USPSDataset(_DatasetWithLength):
    def __init__(self, config: USPSReaderConfig, specs: PreprocessingSpecs):
        self._use_images = config.use_usps_images
        self._use_labels = config.use_usps_labels

        try:
            self._data = self._get_data(config, specs)
        except EOFError or FileNotFoundError:
            sleep(5)
            self._data = self._get_data(config, specs)

    def __getitem__(self, index: np.int64) -> Dict[str, np.ndarray]:
        return _extract_data_from_dataset(
            self._data, int(index), self._use_images, self._use_labels
        )

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def _get_data(config: USPSReaderConfig, specs: PreprocessingSpecs):
        return torchvision.datasets.USPS(
            root=settings.torch_dataset_location,
            train=config.train_split,
            transform=specs.get_pt_preprocessing_fn(),
            download=True,
        )


class USPSReader(PTReader):
    """Loads and prepares the USPS dataset.

    Args:
        config (USPSReaderConfig): Reader configuration.
        specs (PreprocessingSpecs): Preprocessing specification.
        batch_size (int, optional): Batch size; if not specified, maximal possible batch size (whole dataset) is used.
    """

    def __init__(
        self,
        config: USPSReaderConfig,
        specs: PreprocessingSpecs,
        batch_size: Optional[int],
    ):
        self._config = config
        self._specs = specs
        super().__init__(config, batch_size)

    def _get_dataset(self) -> _DatasetWithLength:
        return _USPSDataset(self._config, self._specs)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


def _get_torchvision_reader(
    config: PTReaderConfig, specs: PreprocessingSpecs, batch_size: Optional[int]
) -> PTReader:
    """Returns the appropriate Torchvision reader given the config.

    Args:
        config (PTReaderConfig): Reader configuration.
        specs (PreprocessingSpecs): Preprocessing specification.
        batch_size (int, optional): Batch size; if not specified, maximal possible batch size (whole dataset) is used.

    Returns:
        PTReader: torchvision reader corresponding to the specified config.
    """
    if isinstance(config, QMNISTReaderConfig):
        return QMNISTReader(config, specs, batch_size)
    if isinstance(config, USPSReaderConfig):
        return USPSReader(config, specs, batch_size)
    raise ValueError(f"Unknown torchvision config {config!r}")

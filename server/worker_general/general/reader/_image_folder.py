from typing import Dict, Optional

import numpy as np
from pipeline import DataType
from pipeline.model import PreprocessingSpecs
from schemas.requests.reader import ImageFolderReaderConfig
from torchvision.datasets import ImageFolder

from .._config import settings
from ._pytorch import PTReader, _DatasetWithLength, _extract_data_from_dataset


class _ImageFolderDataset(_DatasetWithLength):
    def __init__(self, config: ImageFolderReaderConfig, specs: PreprocessingSpecs):
        self._use_images = config.use_images
        self._use_labels = config.use_labels

        self._data = ImageFolder(
            root=settings.get_input_path_str(config.images_path),
            transform=specs.get_pt_preprocessing_fn(),
        )

    def __getitem__(self, index: np.int64) -> Dict[str, np.ndarray]:
        return _extract_data_from_dataset(
            self._data, int(index), self._use_images, self._use_labels
        )

    def __len__(self) -> int:
        return len(self._data)


class ImageFolderReader(PTReader):
    """Loads and prepares images stored in subfolders of a folder.

    Args:
        config (ImageFolderReaderConfig): Reader configuration.
        specs (PreprocessingSpecs): Preprocessing specification.
        batch_size (int, optional): Batch size; if not specified, maximal possible batch
            size (whole dataset) is used.
    """

    def __init__(
        self,
        config: ImageFolderReaderConfig,
        specs: PreprocessingSpecs,
        batch_size: Optional[int],
    ):
        self._config = config
        self._specs = specs
        super().__init__(config, batch_size)

    def _get_dataset(self) -> _DatasetWithLength:
        return _ImageFolderDataset(config=self._config, specs=self._specs)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE

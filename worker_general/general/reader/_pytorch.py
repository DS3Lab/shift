from abc import abstractmethod
from collections.abc import Sized
from typing import Dict, Optional, Tuple

import numpy as np
from pipeline.reader import Reader
from schemas.requests.reader import (
    READER_EMBED_FEATURE_NAME,
    READER_LABEL_FEATURE_NAME,
    PTReaderConfig,
)
from torch.utils import data as data
from torchvision.datasets import VisionDataset

from .._config import settings


class _DatasetWithLength(data.Dataset):
    """Defines PyTorch datasets that have a known length."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


def _extract_data_from_dataset(
    dataset: VisionDataset, index: int, include_embed_feature: bool, include_label: bool
) -> Dict[str, np.ndarray]:
    """Extracts a specific data point from the dataset and returns only the requested
    features. It must hold that the first returned value is the main feature (what can
    be embedded) and the second value is the label.

    Args:
        dataset (VisionDataset): Dataset from which the data points will be extracted.
        index (int): Index of the data point.
        include_embed_feature (bool): Specifies whether the result should include the
            feature that can be embedded.
        include_label (bool): Specifies whether the result should include the label.

    Returns:
        Dict[str, np.ndarray]: Requested extracted data.
    """
    tup = dataset[index]
    result = {}
    if include_embed_feature:
        result[READER_EMBED_FEATURE_NAME] = tup[0].numpy()
    if include_label:
        result[READER_LABEL_FEATURE_NAME] = (
            tup[1].numpy() if isinstance(tup[1], np.ndarray) else tup[1]
        )
    return result


class _RangeSampler(data.Sampler):
    """Constructs indices that define a subset of the dataset and the order in which
    dataset elements are accessed.

    Args:
        length (int): Length of the dataset.
        start_stop (Tuple[int, int], optional): First (inclusive) and last (exclusive)
            index which define a subset of the dataset to use. Value None means that
            the entire dataset is used.
        seed (int, optional): Seed to use for shuffling. The default value None means
            that shuffling is not performed.
        data_source (Sized, optional): Needed to call super constructor.
    """

    def __init__(
        self,
        length: int,
        start_stop: Optional[Tuple[int, int]],
        seed: Optional[int] = None,
        data_source: Optional[Sized] = None,
    ):
        super().__init__(data_source)

        # Sanity checks
        if start_stop is not None:
            start, stop = start_stop
            if start > stop:
                raise ValueError(
                    f"Invalid range: {start} (start) > {stop} (stop) holds"
                )
            if start < 0:
                raise ValueError(f"Invalid index {start}")
            if stop >= length:
                raise ValueError(f"Invalid index {stop} for length {length}")

        # Perform shuffling
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
            indices = rng.permutation(length)
        # No shuffling
        else:
            indices = np.arange(length)

        # Because of copy, the potentially enormous permutation array is freed from the
        # memory and only the relevant part is kept
        if start_stop is not None:
            self._indices = indices[start_stop[0] : start_stop[1]].copy()
        else:
            self._indices = indices

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


class PTReader(Reader):
    """Implements the functionality common to all PyTorch datasets.

    Note: when subclassing, make sure that the super __init__ call happens at the end of the :py:func:`__init__` method, so that the call to :py:func:`_get_dataset` can be executed.

    Args:
        config (PTReaderConfig): Reader configuration. Only contains configuration that
            is common to all PyTorch datasets.
        batch_size (int, optional): Batch size; if not specified, maximal possible batch
            size (whole dataset) is used.
    """

    def __init__(self, config: PTReaderConfig, batch_size: Optional[int]):
        dataset = self._get_dataset()
        sampler = _RangeSampler(
            length=len(dataset),
            start_stop=(config.slice.start, config.slice.stop)
            if config.slice is not None
            else None,
            seed=config.seed,
        )
        self._data = iter(
            data.DataLoader(
                dataset,
                batch_size=batch_size if batch_size is not None else len(dataset),
                num_workers=settings.pt_num_workers,
                prefetch_factor=settings.pt_prefetch_factor,
                sampler=sampler,
                # Inspired by the default_collate function
                # https://github.com/pytorch/pytorch/blob/58eb23378f2a376565a66ac32c
                # 93a316c45b6131/torch/utils/data/_utils/collate.py#L74
                # Idea:
                # 1. Get the keys from the first sample in a batch
                # 2. For each key: merge all the data in the batch
                collate_fn=lambda batch: {
                    key: np.stack([x[key] for x in batch], axis=0)
                    for key in batch[0].keys()
                },
            )
        )

    @abstractmethod
    def _get_dataset(self) -> _DatasetWithLength:
        """Returns the dataset that will be served by the reader. The reader takes care of loading samples from the dataset in a specific order using multiple workers."""
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        return next(self._data)

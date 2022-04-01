from unittest import mock

import numpy as np
import pytest
import torch as pt
from pipeline import DataType
from schemas import READER_EMBED_FEATURE_NAME, READER_LABEL_FEATURE_NAME
from schemas.requests.reader import Slice
from torchvision.datasets import VisionDataset

from .._pytorch import (
    PTReader,
    _DatasetWithLength,
    _extract_data_from_dataset,
    _RangeSampler,
)


class TestDataExtraction:
    @pytest.fixture(scope="class")
    def vision_dataset(self) -> VisionDataset:
        class MockDataset(VisionDataset):
            def __init__(self):
                super().__init__(".")

            def __getitem__(self, index: np.int64) -> tuple:
                return pt.as_tensor([1, 2, 3], dtype=pt.float32), pt.as_tensor(
                    0, dtype=pt.int64
                )

            def __len__(self) -> int:
                return 10

        return MockDataset()

    def test_extract_data_from_dataset(self, vision_dataset):
        result = _extract_data_from_dataset(vision_dataset, 2, True, True)
        assert READER_EMBED_FEATURE_NAME in result
        assert READER_LABEL_FEATURE_NAME in result


class TestRangeSampler:
    def test_non_subset_slice(self):
        """Test all cases when indices are not valid are caught beforehand."""
        # 1. Invalid indices
        with pytest.raises(ValueError):
            _ = _RangeSampler(length=10, start_stop=(8, 2))

        # 2. Not a subset (right side)
        with pytest.raises(ValueError):
            _ = _RangeSampler(length=10, start_stop=(5, 20))

        # 3. Not a subset (left side)
        with pytest.raises(ValueError):
            _ = _RangeSampler(length=10, start_stop=(-5, 10))

    def test_no_shuffling(self):
        sampler = _RangeSampler(length=10, start_stop=(5, 8))
        assert len(sampler) == 3
        assert list(sampler) == [5, 6, 7]

    def test_shuffling(self):
        sampler = _RangeSampler(length=10, start_stop=None, seed=0)
        assert len(sampler) == 10
        assert sorted(list(sampler)) == list(range(10))

        sampler_subset = _RangeSampler(length=10, start_stop=(2, 6), seed=0)
        assert len(sampler_subset) == 4
        assert list(sampler)[2:6] == list(sampler_subset)


class TestPTReader:
    @pytest.fixture(scope="class")
    def dataset(self) -> PTReader:
        class MockDataset(_DatasetWithLength):
            def __len__(self) -> int:
                return 10

            def __getitem__(self, index: np.int64):
                return {
                    READER_EMBED_FEATURE_NAME: np.array([1, 2, 3, 4], dtype=np.float32),
                    READER_LABEL_FEATURE_NAME: 5,
                }

        class MockReader(PTReader):
            def __init__(self):
                super().__init__(
                    mock.NonCallableMock(slice=Slice(start=2, stop=7), seed=None),
                    batch_size=4,
                )

            def _get_dataset(self) -> _DatasetWithLength:
                return MockDataset()

            @property
            def data_type(self) -> DataType:
                return DataType.UNKNOWN

        return MockReader()

    def test_data_expected_format(self, dataset):
        result = list(dataset)
        assert len(result) == 2

        first_batch = result[0]
        assert first_batch[READER_EMBED_FEATURE_NAME].shape == (4, 4)
        assert first_batch[READER_LABEL_FEATURE_NAME].shape == (4,)

        second_batch = result[1]
        assert second_batch[READER_EMBED_FEATURE_NAME].shape == (1, 4)
        assert second_batch[READER_LABEL_FEATURE_NAME].shape == (1,)

import pytest
from pydantic.error_wrappers import ValidationError

from ..response import NearestNeighborResult


def test_invalid_nearest_neighbor_result():
    # 0. Valid
    _ = NearestNeighborResult(
        test_labels=[0],
        test_indices_within_readers=[0],
        test_reader_indices=[0],
        train_labels=[0],
        train_indices_within_readers=[0],
        train_reader_indices=[0],
    )

    # 1. Invalid
    with pytest.raises(ValidationError):
        _ = NearestNeighborResult(
            test_labels=[0],
            test_indices_within_readers=[0, 1],
            test_reader_indices=[0],
            train_labels=[0],
            train_indices_within_readers=[0],
            train_reader_indices=[0],
        )

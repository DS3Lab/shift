from typing import Sequence
from unittest import mock

import faiss
import numpy as np
import pipeline.io as io
import pytest
from faiss import knn
from pipeline import Device
from schemas import get_hash
from schemas.classifier import Classifier
from schemas.requests.reader import READER_EMBED_FEATURE_NAME as REFN
from schemas.requests.reader import READER_LABEL_FEATURE_NAME as RLFN

from .. import _nearest_neighbors
from .._nearest_neighbors import (
    _apply_label_changes,
    _faiss_nn,
    _FaissOutOfMemoryError,
    _nn,
    nearest_neighbors,
)

euclidean_test = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32).reshape(-1, 1)
euclidean_train = np.array(
    [4.1, 2.1, 7.1, 5.1, 1.1, 3.1, 6.1], dtype=np.float32
).reshape(-1, 1)
euclidean_true_indices = np.array([4, 1, 5, 0, 3, 6, 2], dtype=np.int64).reshape(-1, 1)

cosine_test = np.array([[1, 1], [-1, 1]], dtype=np.float32)
cosine_train = np.array([[-2.3, 2], [1000, -1000], [0.5, 0.4]], dtype=np.float32)
cosine_true_indices = np.array([2, 0], dtype=np.int64).reshape(-1, 1)


@pytest.mark.parametrize(
    "train,test,distance,true_indices",
    [
        (
            euclidean_train,
            euclidean_test,
            Classifier.EUCLIDEAN_NN,
            euclidean_true_indices,
        ),
        [cosine_train, cosine_test, Classifier.COSINE_NN, cosine_true_indices],
    ],
)
def test_faiss_nn(
    train: np.ndarray, test: np.ndarray, distance: Classifier, true_indices: np.ndarray
):
    dist, ind = _faiss_nn(
        train=train,
        test=test,
        device=Device.CPU,
        distance=distance,
    )
    assert np.allclose(ind, true_indices)


def test_nn(monkeypatch):
    def faiss_knn_fails_above_3_points(xb, *args, **kwargs):
        # Fail if there are more than 3 points in the training set
        if xb.shape[0] > 3:
            raise _FaissOutOfMemoryError
        return knn(xb=xb, *args, **kwargs)

    # Simulate OOM for CPU even though this would never be thrown on CPU, only on GPU
    monkeypatch.setattr(faiss, "knn", faiss_knn_fails_above_3_points)

    dist, ind = _nn(
        train=euclidean_train,
        test=euclidean_test,
        device=Device.CPU,
        distance=Classifier.EUCLIDEAN_NN,
    )
    assert np.allclose(ind, euclidean_true_indices)


def test_not_enough_gpu_memory(monkeypatch):
    """Case when there is not enough GPU memory to compute distance between test points
    and a single training point."""

    def always_fail(*_, **__):
        raise _FaissOutOfMemoryError

    monkeypatch.setattr(faiss, "knn", always_fail)
    with pytest.raises(RuntimeError):
        _ = _nn(
            train=euclidean_train,
            test=euclidean_test,
            device=Device.CPU,
            distance=Classifier.EUCLIDEAN_NN,
        )


def test_apply_label_changes(monkeypatch):
    # Reading the change is mocked such that each mocked change object specifies both
    # the indices and the labels and when the change is read simply the specified labels
    # are returned
    def read_change_labels_mock(change_mock):
        return change_mock.labels

    monkeypatch.setattr(io, "read_change_labels", read_change_labels_mock)

    # Reader 0 (position -> label):
    # # Unsorted: 1 -> 3, 15 -> 1, 1 -> 3, 0 -> 2, 2 -> 4
    # # Sorted: 0 -> 2, 1 -> 3 (twice), 2 -> 4, 15 -> 1
    # Reader 1 (position -> label):
    # # Unsorted: 5 -> 6, 7 -> 5, 5 -> 6, 10 -> 1, 1 -> 0
    # # Sorted: 1 -> 0, 5 -> 6 (twice), 7 -> 5, 10 -> 1
    existing_labels = [3, 1, 6, 3, 5, 6, 1, 2, 0, 4]
    reader_indices = [0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
    indices_within_reader = [1, 15, 5, 1, 7, 5, 10, 0, 1, 2]

    changes = {
        0: [
            mock.NonCallableMock(indices=[2, 14], labels=[7, 0]),
            mock.NonCallableMock(indices=[1, 2, 15], labels=[5, 8, 10]),
        ],
        1: [mock.NonCallableMock(indices=[5, 10], labels=[8, 1])],
        # Changes to reader that is not present should be ignored
        2: [mock.NonCallableMock(indices=[10, 20], labels=[0, 0])],
    }

    # Expected result:
    # # Reader 0:
    # # 0 -> 2, 1 -> 5 (twice), 2 -> 8, 15 -> 10
    # # Reader 1:
    # # 1 -> 0, 5 -> 8 (twice), 7 -> 5, 10 -> 1

    expected_labels = [5, 10, 8, 5, 5, 8, 1, 2, 0, 8]
    existing_labels_copy = existing_labels.copy()
    obtained_labels = _apply_label_changes(
        existing_labels, reader_indices, indices_within_reader, changes
    )
    # Check that original labels stay the same
    assert existing_labels_copy == existing_labels
    assert expected_labels == obtained_labels


def test_nearest_neighbors(monkeypatch):
    def column_vector(seq: Sequence[int], dtype) -> np.ndarray:
        return np.array(seq, dtype=dtype).reshape(-1, 1)

    # 1. Prepare data
    test_labels = [1, 2, 3, 4, 5, 6]
    test_split_sizes = [3, 2, 1]

    # 2. Mocking test set
    mocked_test_set = {
        REFN: mock.NonCallableMock(),
        RLFN: np.array(test_labels, dtype=np.int64),
    }
    monkeypatch.setattr(
        io,
        "read_mutable_data_sequence",
        lambda *_, **__: (mocked_test_set, test_split_sizes),
    )

    # 3. Mock train set
    train_mutable_data = iter(
        [
            # First mocked train mutable data
            (
                {
                    REFN: mock.NonCallableMock(),
                    RLFN: np.array([4, 2, 3, 1], dtype=np.int64),
                },
                4,
            ),
            # Second mocked train mutable data
            (
                {
                    REFN: mock.NonCallableMock(),
                    RLFN: np.array([5, 1, 0, 2, 6, 4, 6, 7, 1], dtype=np.int64),
                },
                9,
            ),
        ]
    )

    monkeypatch.setattr(
        io, "read_mutable_data", lambda *_, **__: next(train_mutable_data)
    )

    # 4. Mock nn
    nn_results = iter(
        [
            (
                # First mocked train mutable data
                # Distances
                column_vector([0.5, 0.1, 6, 2, 1, 5], dtype=np.float32),
                # Index of closest point
                column_vector([2, 1, 3, 0, 2, 1], dtype=np.int64),
            ),
            (
                # Second mocked train mutable data
                # Distances
                column_vector([9, 5, 2, 6, 0.2, 6], dtype=np.float32),
                # Index of closest point
                column_vector([4, 0, 5, 2, 3, 6], dtype=np.int64),
            ),
        ]
    )

    monkeypatch.setattr(_nearest_neighbors, "_nn", lambda *_, **__: next(nn_results))

    # # Expected result:
    # # Compare 0.5 and 9   -> reader 0 closer -> index within reader 2 -> label 3
    # # Compare 0.1 and 5   -> reader 0 closer -> index within reader 1 -> label 2
    # # Compare 6   and 2   -> reader 1 closer -> index within reader 5 -> label 4
    # # Compare 2   and 6   -> reader 0 closer -> index within reader 0 -> label 4
    # # Compare 1   and 0.2 -> reader 1 closer -> index within reader 3 -> label 2
    # # Compare 5   and 6   -> reader 0 closer -> index within reader 1 -> label 2

    # 5. Run
    hash_ = get_hash("hash")
    cr_base = mock.NonCallableMock(
        train=[mock.NonCallableMock(), mock.NonCallableMock()],
        test=[mock.NonCallableMock()],
        hash=hash_,
    )
    cr = mock.NonCallableMock(
        get_request_without_closing_label_changes=lambda: cr_base,
        get_closing_train_label_changes=lambda: [],
        get_closing_test_label_changes=lambda: [],
        hash=hash_,
        hash_without_closing_label_changes="hash",
    )
    result = nearest_neighbors(request=cr, nn_result=None, device=Device.CPU)

    # 6. Compare result
    assert list(result.keys()) == [hash_]
    nn_result = result[hash_]
    assert nn_result.test_labels == test_labels
    assert nn_result.test_reader_indices == [0, 0, 0, 1, 1, 2]
    assert nn_result.test_indices_within_readers == [0, 1, 2, 0, 1, 0]
    assert nn_result.train_labels == [3, 2, 4, 4, 2, 2]
    assert nn_result.train_reader_indices == [0, 0, 1, 0, 1, 0]
    assert nn_result.train_indices_within_readers == [2, 1, 5, 0, 3, 1]
    assert np.allclose(nn_result.error, 2 / 3)

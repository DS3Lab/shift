from collections import defaultdict
from itertools import chain
import timeit
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pipeline.io as io
from pipeline import Device
from schemas import READER_EMBED_FEATURE_NAME as REFN
from schemas import READER_LABEL_FEATURE_NAME as RLFN
from schemas import Hash
from schemas.classifier import Classifier
from schemas.requests.common import Change, ClassifierRequest, MutableData
from schemas.response import NearestNeighborResult
from common.telemetry.telemetry import add_event


class _FaissOutOfMemoryError(BaseException):
    """Signals that Faiss was unable to perform 1NN, because there was not enough GPU memory."""

    pass


def _faiss_nn(
    train: np.ndarray, test: np.ndarray, device: Device, distance: Classifier
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds a single nearest neighbor among training points for each test point.

    Args:
        train (np.ndarray): Training points with shape (number of points x dimension).
        test (np.ndarray): Test points with shape (number of points x dimension).
        device (Device): Device used by Faiss for 1NN.
        distance (Classifier): Type of distance used for nearest neighbors.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Distance to closest training point for each test point and the indices of the closest training points. Both have shape (number of test point, 1).
    """
    assert distance in {
        Classifier.EUCLIDEAN_NN,
        Classifier.COSINE_NN,
    }, f"Classifier {distance} is not valid for nearest neighbors"
    if distance == Classifier.EUCLIDEAN_NN:
        transformed_train = train
        transformed_test = test
    else:
        transformed_train = train / np.linalg.norm(train, axis=1).reshape(-1, 1)
        transformed_test = test / np.linalg.norm(test, axis=1).reshape(-1, 1)

    if device == Device.GPU:
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()

        try:
            dist, ind = faiss.knn_gpu(
                res=res,
                xb=transformed_train,
                xq=transformed_test,
                k=1,
                metric=faiss.METRIC_L2,
            )
        except RuntimeError as e:
            raise _FaissOutOfMemoryError from e if "out of memory" in str(e) else e

    else:
        dist, ind = faiss.knn(
            xb=transformed_train,
            xq=transformed_test,
            k=1,
            metric=faiss.METRIC_L2,
        )

    return dist, ind


def _nn(
    train: np.ndarray, test: np.ndarray, device: Device, distance: Classifier
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds a single nearest neighbor among training points for each test point. If the computation of nearest neighbors cannot be performed directly on the supplied data, training data is split into pieces such that the nearest neighbor from each piece can be inferred for each test point without running out of GPU memory (if GPU device is used).

    Args:
        train (np.ndarray): Training points with shape (number of points x dimension).
        test (np.ndarray): Test points with shape (number of points x dimension).
        device (Device): Device used by Faiss for 1NN.
        distance (Classifier): Type of distance used for nearest neighbors.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Distance to the closest training point and its index for each test point. Both arrays have shape (number of test points, 1).
    """

    # Distances and indices for splits of training points
    partial_results: List[Tuple[np.ndarray, np.ndarray]] = []

    nn_fits_into_memory = False
    num_train_points = train.shape[0]
    num_train_splits = 1

    # Produce smaller and smaller training splits until nn can be run
    while not nn_fits_into_memory:
        train_splits: Sequence[np.ndarray] = np.array_split(train, num_train_splits)
        try:
            # np.array_split will make first split the biggest - if nn does not fail for this split, it also should not for remaining ones
            d, i = _faiss_nn(
                train=train_splits[0], test=test, device=device, distance=distance
            )

        # _faiss_nn failed
        except _FaissOutOfMemoryError:
            if num_train_splits < num_train_points:
                num_train_splits += 1
                continue
            raise RuntimeError(
                "Distance between test points and a single training point cannot be computed, because there is not enough GPU memory"
            )

        # faiss_nn did not fail
        else:
            partial_results.append((d, i))
            nn_fits_into_memory = True

        # Number of training points processed in each of the iterations
        cumulative_split_sizes: Sequence[int] = np.cumsum(
            [t.shape[0] for t in train_splits]
        ).tolist()

        for index_split in range(1, len(train_splits)):
            d, i = _faiss_nn(
                train=train_splits[index_split],
                test=test,
                device=device,
                distance=distance,
            )
            # Indices always start from 0 - fix that to account for training points from previous splits
            partial_results.append((d, i + cumulative_split_sizes[index_split - 1]))

    # Both matrices have shape: (test set size, num_train_splits)
    # For each test point: distance to closest training point from each split
    dist_per_split = np.concatenate([p[0] for p in partial_results], axis=1)
    # For each test point: index of closest training point from each split
    # Index was already corrected and can be thus directly used on the original data
    ind_per_split = np.concatenate([p[1] for p in partial_results], axis=1)

    # 1. Find for each test point the split that contains the nearest neighbor
    # Shape: (test set size, 1)
    ind_of_closest_split = np.expand_dims(np.argmin(dist_per_split, axis=1), axis=1)

    # 2. Return for each test point the distance to the closest training point and its index - Shape of both: (test set size, 1)
    return np.take_along_axis(
        dist_per_split,
        ind_of_closest_split,
        axis=1,
    ), np.take_along_axis(
        ind_per_split,
        ind_of_closest_split,
        axis=1,
    )


def _apply_label_changes(
    labels: Sequence[int],
    reader_indices: Sequence[int],
    indices_within_reader: Sequence[int],
    changes: Dict[int, Sequence[Change]],
) -> List[int]:
    """Applies label changes to the labels. Each label has a corresponding reader index and index within that reader. Those two numbers tell where the label was taken from.
    Reader index here denotes the index of the mutable data which consists of the base reader/data and changes applied to it.

    Changes are specified per reader, where each reader index is a key of the dictionary. Indices within a reader are indices specified by the change.

    Changes here can also alter the embed feature, however those changes will be
    ignored.

    Args:
        labels (Sequence[int]): Original labels.
        reader_indices (Sequence[int]): Specifies for each label its origin with respect to the reader (mutable data).
        indices_within_reader (Sequence[int]): Specifies for each label its position within the reader of its origin.
        changes (Dict[int, Sequence[Change]]): Specifies for each reader index the label changes that need to be applied to that reader.

    Returns:
        List[int]: New labels obtained by applying changes to the existing labels.
    """

    # Performs a copy of a list
    new_labels: List[int] = list(labels)

    # 1. Create a mapping: label origin -> index of label within the sequence
    # Because same label (and its origin) can appear multiple times, the map points to a sequence indices instead of a single index
    map_ = defaultdict(list)
    index_tuples = zip(reader_indices, indices_within_reader)
    for overall_index, index_tuple in enumerate(index_tuples):
        map_[index_tuple].append(overall_index)

    # 2. Iterate through all changes
    for reader_index in changes:
        for change in changes[reader_index]:
            change_labels = io.read_change_labels(change)

            for index_within_reader, change_index in zip(
                change.base_indices, change.change_indices
            ):

                # Check if label change position (origin) has a match
                index_tuple = (reader_index, index_within_reader)
                if index_tuple in map_:
                    # If there is a match, change the label
                    for overall_index in map_[index_tuple]:
                        new_labels[overall_index] = change_labels[change_index]
    return new_labels


def _apply_label_changes_to_nn_result(
    result: NearestNeighborResult,
    train_changes: Dict[int, Sequence[Change]],
    test_changes: Dict[int, Sequence[Change]],
) -> NearestNeighborResult:

    result_copy = result.copy(deep=True)
    result_copy.train_labels = _apply_label_changes(
        labels=result.train_labels,
        reader_indices=result.train_reader_indices,
        indices_within_reader=result.train_indices_within_readers,
        changes=train_changes,
    )
    result_copy.test_labels = _apply_label_changes(
        labels=result.test_labels,
        reader_indices=result.test_reader_indices,
        indices_within_reader=result.test_indices_within_readers,
        changes=test_changes,
    )

    return result_copy


def _get_nn_result(
    train_mutable_data_sequence: Sequence[MutableData],
    test_mutable_data_sequence: Sequence[MutableData],
    device: Device,
    distance: Classifier,
) -> NearestNeighborResult:
    """Constructs a NN result from the raw train and test data. The result contains everything needed to recalculate the error if only labels are changed afterwards in the training and/or test set.

    Args:
        train_mutable_data_sequence (Sequence[MutableData]): Train data.
        test_mutable_data_sequence (Sequence[MutableData]): Test data.
        device (Device): Device used for 1NN.
        distance (Classifier): Type of distance used for nearest neighbors.

    Returns:
        NearestNeighborResult: A summary of labels for each point from the test set and the corresponding label of the nearest point from the training set.
    """

    # Load test data
    start = timeit.default_timer()
    test_dict, test_md_lengths = io.read_mutable_data_sequence(
        test_mutable_data_sequence
    )
    stop = timeit.default_timer()
    # add_event(
    #     'load_inference_result',
    #     {

    #         'device': 'GPU' if device == Device.GPU else 'CPU',
    #     },
    #     round(1000 * (stop - start))
    # )
    # Each list contains a column vector for each element of train mutable data
    # sequence
    # # For each test point: distance to the closest training point
    md_dists: List[np.ndarray] = []
    # # For each test point: index within reader/mutable data of the closest training
    # # point
    md_indices: List[np.ndarray] = []
    # # For each test point: label of the closest training point
    md_labels: List[np.ndarray] = []

    # 1. Go through sequence
    for md_index, mutable_data in enumerate(train_mutable_data_sequence):
        # 1.1 Load data
        start = timeit.default_timer()
        train_md_dict, train_md_length = io.read_mutable_data(mutable_data)
        stop = timeit.default_timer()
        # add_event(
        #     'load_inference_result',
        #     {

        #         'device': 'GPU' if device == Device.GPU else 'CPU',
        #     },
        #     round(1000 * (stop - start))
        # )
        # 1.2 Determine for each test point the closest training point and its index
        dist_closest_points, ind_closest_points = _nn(
            train=train_md_dict[REFN],
            test=test_dict[REFN],
            device=device,
            distance=distance,
        )

        # 1.3 Determine for each test point the label of closest training point
        # Shape after first and second indexing: (number of test points, 1)
        # Shape after first indexing: (number of test points,)
        closest_training_point_labels = train_md_dict[RLFN][ind_closest_points]

        # 1.4 Store results
        md_dists.append(dist_closest_points)
        md_indices.append(ind_closest_points)
        md_labels.append(closest_training_point_labels)

    # 2. Merge results into matrices of shape
    # (number of test points, train mutable data sequence length)
    md_dists_matrix = np.concatenate(md_dists, axis=1)
    md_indices_matrix = np.concatenate(md_indices, axis=1)
    md_labels_matrix = np.concatenate(md_labels, axis=1)

    # 3. Find for each test point the element of a sequence that contains the nearest neighbor - shape: (test set size, 1)
    ind_closest_md = np.expand_dims(np.argmin(md_dists_matrix, axis=1), axis=1)

    # 4. Calculate error
    return NearestNeighborResult(
        test_labels=test_dict[RLFN].tolist(),
        test_indices_within_readers=list(
            chain.from_iterable([range(x) for x in test_md_lengths])
        ),
        test_reader_indices=list(
            chain.from_iterable(
                [[i] * length for i, length in enumerate(test_md_lengths)]
            )
        ),
        # Selects for each test point the label of the nearest neighbor
        train_labels=np.take_along_axis(md_labels_matrix, ind_closest_md, axis=1)
        .reshape(-1)
        .tolist(),
        # Selects for each test point the index within reader of the nearest neighbor
        train_indices_within_readers=np.take_along_axis(
            md_indices_matrix, ind_closest_md, axis=1
        )
        .reshape(-1)
        .tolist(),
        train_reader_indices=ind_closest_md.reshape(-1).tolist(),
    )


def nearest_neighbors(
    request: ClassifierRequest,
    nn_result: Optional[NearestNeighborResult],
    device: Device,
) -> Dict[Hash, NearestNeighborResult]:
    """Constructs the nearest neighbor result that corresponds to the passed classifier request. If the nearest neighbor partial result already exists, it is used as a starting point.

    Args:
        request (ClassifierRequest): Classifier request for which the nearest neighbor result will be constructed.
        nn_result (NearestNeighborResult, optional): Partial nearest neighbor result which if specified, corresponds to the classifier request without closing label changes.
        device (Device): Device used for 1NN.

    Returns:
        NearestNeighborResult: A summary of labels for each point from the test set and the corresponding label of the nearest point from the training set.
    """

    return_value = {}
    # TODO: NN - handle different cases depending on whether a partial result was passed or not. If the partial result was passed, compute only what is missing, otherwise compute everything. NN computation is done with _get_nn_result.
    # When computing the missing parts, make sure to preserve the enumeration of MutableData (see 'enumerate' in _get_nn_result) by using the correct offset value.
    if nn_result is None:
        base_request = request.get_request_without_closing_label_changes()
        nn_result = _get_nn_result(
            base_request.train,
            base_request.test,
            device=device,
            distance=request.classifier.name,
        )
        return_value[base_request.hash] = nn_result

    # TODO: NN - write another function that merges the partial result (if it exists) together with the new result (missing training points).
    # Both results are instances of the NearestNeighborResult.
    nn_result_with_applied_changes = _apply_label_changes_to_nn_result(
        nn_result,
        request.get_closing_train_label_changes(),
        request.get_closing_test_label_changes(),
    )

    return_value[request.hash] = nn_result_with_applied_changes

    return return_value

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from schemas import READER_EMBED_FEATURE_NAME as REFN
from schemas import READER_LABEL_FEATURE_NAME as RLFN
from schemas.requests.common import Change, MutableData
from schemas.requests.reader import ResultsNumPyReaderConfig

from ._numpy_io import NumPyReader


def _check_embed_data(data: np.ndarray):
    """Checks that a NumPy array contains valid embedded images/texts.

    Args:
        data (np.ndarray): NumPy array to check.
    """
    if len(data.shape) != 2 or data.shape[1] == 0:
        raise ValueError(
            f"Embedded data should be a vector "
            f"(provided samples have shape {data.shape[1:]})"
        )

    if data.dtype != np.float32:
        raise TypeError(f"Unsupported embed type {data.dtype}, please use np.float32")


def _check_label_data(data: np.ndarray):
    """Checks that a NumPy array contains valid labels.

    Args:
        data (np.ndarray): NumPy array to check.
    """
    if len(data.shape) != 1 or data.shape[0] == 0:
        raise ValueError(
            f"Labels should be scalars (provided labels have shape {data.shape[1:]})"
        )

    if data.dtype != np.int64 and data.dtype != np.int32:
        raise TypeError(f"Unsupported label type {data.dtype}, please use np.int64")


def _check_shape_compatibility(data1: np.ndarray, data2: np.ndarray):
    """Checks that two NumPy arrays have the same shape.

    Args:
        data1 (np.ndarray): First array.
        data2 (np.ndarray): Second array.
    """
    if data1.shape[1:] != data2.shape[1:]:
        raise ValueError(
            f"Data should have same shape ignoring the first dimension: "
            f"{data1.shape[1:]!r} != {data2.shape[1:]!r}"
        )


def _check_indices_compatibility(
    base_length: Optional[int], change_length: int, change_indices: Sequence[int]
):
    """Checks that supplied change indices are valid.

    If a change supplies more data than there are indices, this is not considered to be an error. This way, the same reader can be reused multiple times with different indices in order to avoid running inference multiple times for the same data.
    If there are more indices than there is change data, an error is raised since for some indices it is then not clear which data to use.

    Args:
        base_length (int, optional): Length of the base data. The value None means that the length of the base data is not known.
        change_length (int): Length of the data defined by the change.
        change_indices (int): Indices that define which data point of base data should be replaced with the data defined by the change. All indices are expected to be non-negative.
    """
    if len(change_indices) > change_length:
        raise ValueError(
            f"There are more change indices specified ({len(change_indices)}) than "
            f"there are changes available ({change_length})"
        )

    if base_length is not None:
        for i in change_indices:
            if i >= base_length:
                raise IndexError(
                    f"Change with index {i} is not valid for base data with length "
                    f"{base_length}"
                )


def _read_change(change: Change) -> Dict[str, np.ndarray]:
    """Reads data from a change."""
    change_reader = NumPyReader(
        config=ResultsNumPyReaderConfig(
            embed_feature=REFN if change.embed_feature_present else None,
            label_feature=RLFN if change.label_feature_present else None,
            job_hash=change.inference_request_hash,
        ),
        batch_size=None,
    )

    result = next(change_reader)
    if change.embed_feature_present:
        # Check if we have a dimension like (1,1,..., N)
        # Just squeeze it to (N)
        result[REFN] = np.squeeze(result[REFN])
        _check_embed_data(result[REFN])
    if change.label_feature_present:
        _check_label_data(result[RLFN])

    return result


def read_change_labels(change: Change) -> Sequence[int]:
    """Reads label data specified by a change. This is are all labels that can be read (labels do not depend on specified indices).

    Args:
        change (Change): Change to read labels from.

    Returns:
        Sequence[int]: Label data specified by the change.
    """
    if not change.label_feature_present:
        raise ValueError("Cannot read labels, because labels are not present")
    result = _read_change(change)
    labels = result[RLFN].tolist()
    _check_indices_compatibility(None, len(labels), change.indices)
    return labels


def read_mutable_data(
    mutable_data_spec: MutableData,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Reads the base data and applies changes to it.

    Args:
        mutable_data_spec (MutableData): Specification of the base data and changes applied to it.

    Returns:
        Tuple[Dict[str, np.ndarray], int]: A dictionary containing the embed and label feature, and length of the read data.
    """

    # 1. Read base data
    base_data_dict: Dict[str, np.ndarray] = next(
        NumPyReader(
            config=ResultsNumPyReaderConfig(
                embed_feature=REFN,
                label_feature=RLFN,
                job_hash=mutable_data_spec.inference_request_hash,
            ),
            batch_size=None,
        )
    )

    base_embed_data, base_label_data = base_data_dict[REFN], base_data_dict[RLFN]
    # squeeze from (1,1,...,N) to (N)
    base_embed_data = np.squeeze(base_embed_data)
    base_length = base_embed_data.shape[0]
    _check_embed_data(base_embed_data)
    _check_label_data(base_label_data)

    # 2. Read and apply changes
    for change in mutable_data_spec.changes:
        # 2.1 Read change
        change_data_dict = _read_change(change)

        # 2.2 Apply change to the embed feature
        if change.embed_feature_present:
            change_embed_data = change_data_dict[REFN]
            _check_shape_compatibility(base_embed_data, change_embed_data)
            # _check_indices_compatibility(
            #    base_length, change_embed_data.shape[0], change.indices
            # )

            for base_index, change_index in zip(
                change.base_indices, change.change_indices
            ):
                base_embed_data[base_index] = change_embed_data[change_index]

        # 2.3 Apply change to the label feature
        if change.label_feature_present:
            change_label_data = change_data_dict[RLFN]
            # _check_indices_compatibility(
            #    base_length, change_label_data.shape[0], change.indices
            # )

            for base_index, change_index in zip(
                change.base_indices, change.change_indices
            ):
                base_label_data[base_index] = change_label_data[change_index]

    return {REFN: base_embed_data, RLFN: base_label_data}, base_length


def read_mutable_data_sequence(
    mutable_data_specs: Sequence[MutableData],
) -> Tuple[Dict[str, np.ndarray], Sequence[int]]:
    """Reads the base data (multiple parts) and applies changes to it.

    Args:
        mutable_data_specs (Sequence[int]): Specifications of the base data and changes applied to them.

    Returns:
        Tuple[Dict[str, np.ndarray], Sequence[int]]: A dictionary containing the embed and label feature, and lengths of the read data.
    """

    if len(mutable_data_specs) == 0:
        raise ValueError("There should be at least of mutable data to read")

    parts: List[Dict[str, np.ndarray]] = []
    lengths: List[int] = []
    for i in range(len(mutable_data_specs)):
        part, length = read_mutable_data(mutable_data_specs[i])
        parts.append(part)
        lengths.append(length)

        if i > 0:
            _check_shape_compatibility(parts[0][REFN], parts[i][REFN])

    # Merge the data
    return {
        key: np.concatenate([part[key] for part in parts]) for key in [REFN, RLFN]
    }, lengths

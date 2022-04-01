import secrets

import numpy as np
import pytest
from schemas import (
    READER_EMBED_FEATURE_NAME,
    READER_LABEL_FEATURE_NAME,
    Hash,
    generate_id,
    get_hash,
)
from schemas.requests.reader import Change, MutableData

from .._numpy_io import NumPyWriter
from .._numpy_overlapping_reader import (
    _check_embed_data,
    _check_indices_compatibility,
    _check_label_data,
    _check_shape_compatibility,
    read_change_labels,
    read_mutable_data,
    read_mutable_data_sequence,
)

DEFAULT_FEATURE_LENGTH = 5


def _write_data(
    path: str,
    include_embed_feature: bool,
    include_label_feature: bool,
    length: int,
    fill_value: int,
    feature_length: int = DEFAULT_FEATURE_LENGTH,
):
    writer = NumPyWriter(path)
    to_add = dict()
    if include_embed_feature:
        to_add[READER_EMBED_FEATURE_NAME] = np.full(
            shape=(length, feature_length), fill_value=fill_value, dtype=np.float32
        )
    if include_label_feature:
        to_add[READER_LABEL_FEATURE_NAME] = np.full(
            shape=(length,), fill_value=fill_value, dtype=np.int64
        )
    writer.add(to_add)
    writer.finalize()


@pytest.fixture
def gen_hash():
    def _internal():
        return get_hash(secrets.token_hex(5))

    return _internal


# Each fixture has structure:
# <data type>_<data length>_<feature length>_<default value>
@pytest.fixture
def base_1(data_path, gen_hash) -> Hash:
    hash_ = gen_hash()
    _write_data(
        path=str((data_path / hash_).resolve()),
        include_embed_feature=True,
        include_label_feature=True,
        length=10,
        fill_value=1,
    )
    return hash_


@pytest.fixture
def base_2(data_path, gen_hash) -> Hash:
    hash_ = gen_hash()
    _write_data(
        path=str((data_path / hash_).resolve()),
        include_embed_feature=True,
        include_label_feature=True,
        length=5,
        fill_value=2,
    )
    return hash_


@pytest.fixture
def embed_change_3(data_path, gen_hash) -> Change:
    hash_ = gen_hash()
    _write_data(
        path=str((data_path / hash_).resolve()),
        include_embed_feature=True,
        include_label_feature=False,
        length=10,
        fill_value=3,
    )
    return Change(
        inference_request_id=generate_id(),
        inference_request_hash=hash_,
        indices=[0, 2, 5],
        embed_feature_present=True,
        label_feature_present=False,
    )


@pytest.fixture
def full_change_4(data_path, gen_hash) -> Change:
    hash_ = gen_hash()
    _write_data(
        path=str((data_path / hash_).resolve()),
        include_embed_feature=True,
        include_label_feature=True,
        length=10,
        fill_value=4,
    )
    return Change(
        inference_request_id=generate_id(),
        inference_request_hash=hash_,
        indices=[1, 2, 5, 6, 8],
        embed_feature_present=True,
        label_feature_present=True,
    )


@pytest.fixture
def label_change_5(data_path, gen_hash) -> Change:
    hash_ = gen_hash()
    _write_data(
        path=str((data_path / hash_).resolve()),
        include_embed_feature=False,
        include_label_feature=True,
        length=10,
        fill_value=5,
    )
    return Change(
        inference_request_id=generate_id(),
        inference_request_hash=hash_,
        indices=[1, 2, 3, 4],
        embed_feature_present=False,
        label_feature_present=True,
    )


@pytest.fixture
def mutable_data_1(
    base_1,
    embed_change_3,
    full_change_4,
    label_change_5,
) -> MutableData:
    return MutableData(
        inference_request_id=generate_id(),
        inference_request_hash=base_1,
        changes=[embed_change_3, full_change_4, label_change_5],
    )


@pytest.fixture
def mutable_data_2(base_2) -> MutableData:
    return MutableData(
        inference_request_id=generate_id(), inference_request_hash=base_2, changes=[]
    )


def test_read_change_labels(embed_change_3, full_change_4, label_change_5):
    # 1. Label not present
    with pytest.raises(ValueError):
        read_change_labels(embed_change_3)

    # 2. Label present
    assert read_change_labels(label_change_5) == [5] * 10
    assert read_change_labels(full_change_4) == [4] * 10


class TestChecks:
    def test_check_embed_data(self):
        # 0. Valid
        _check_embed_data(np.zeros(shape=(10, 2), dtype=np.float32))

        # 1. Wrong shape
        with pytest.raises(ValueError):
            _check_embed_data(np.zeros(shape=(10, 2, 3), dtype=np.float32))

        with pytest.raises(ValueError):
            _check_embed_data(np.zeros(shape=(10,), dtype=np.float32))

        with pytest.raises(ValueError):
            _check_embed_data(np.zeros(shape=(10, 0), dtype=np.float32))

        # 2. Wrong type
        with pytest.raises(TypeError):
            _check_embed_data(np.zeros(shape=(10, 2), dtype=np.float64))

    def test_check_label_data(self):
        # 0. Valid
        _check_label_data(np.zeros(shape=(10,), dtype=np.int64))

        # 1. Wrong shape
        with pytest.raises(ValueError):
            _check_label_data(np.zeros(shape=(10, 1), dtype=np.int64))

        with pytest.raises(ValueError):
            _check_label_data(np.zeros(shape=(0,), dtype=np.int64))

        # 2. Wrong type
        with pytest.raises(TypeError):
            _check_label_data(np.zeros(shape=(10,), dtype=np.int32))

    def test_check_shape_compatibility(self):
        # 0. Valid
        _check_shape_compatibility(np.zeros(shape=(10, 2)), np.zeros(shape=(10, 2)))

        # 1. Incompatible shape
        with pytest.raises(ValueError):
            _check_shape_compatibility(np.zeros(shape=(10, 2)), np.zeros(shape=(10, 3)))

        with pytest.raises(ValueError):
            _check_shape_compatibility(
                np.zeros(shape=(10, 2, 1)), np.zeros(shape=(10, 2))
            )

    def test_check_indices_compatibility(self):
        # 0. Valid
        _check_indices_compatibility(
            base_length=10, change_length=6, change_indices=[0, 1, 2]
        )
        _check_indices_compatibility(10, 20, [0, 1, 2])
        _check_indices_compatibility(3, 5, [0, 1, 2])
        _check_indices_compatibility(None, 5, [0, 1, 2])

        # 1. More indices than data
        with pytest.raises(ValueError):
            _check_indices_compatibility(5, 3, [0, 1, 2, 3])

        # 2. Unreachable base data
        with pytest.raises(IndexError):
            _check_indices_compatibility(5, 10, [5])


class TestReadMutableData:
    def test_changes_applied_correctly(self, mutable_data_1):
        data_dictionary, length = read_mutable_data(mutable_data_1)
        assert length == 10

        # 1. Embed feature
        assert READER_EMBED_FEATURE_NAME in data_dictionary
        expected_embed_features = (
            np.array([[3, 4, 4, 1, 1, 4, 4, 1, 4, 1]], dtype=np.float32)
            .repeat(DEFAULT_FEATURE_LENGTH, 0)
            .T
        )
        assert np.allclose(
            data_dictionary[READER_EMBED_FEATURE_NAME], expected_embed_features
        )

        # 2. Label feature
        assert READER_LABEL_FEATURE_NAME in data_dictionary
        expected_label_features = np.array(
            [1, 5, 5, 5, 5, 4, 4, 1, 4, 1], dtype=np.int64
        )
        assert np.allclose(
            data_dictionary[READER_LABEL_FEATURE_NAME], expected_label_features
        )

    def test_read_mutable_data_sequence(self, mutable_data_2):
        with pytest.raises(ValueError):
            read_mutable_data_sequence([])

        data_dictionary, lengths = read_mutable_data_sequence(
            [mutable_data_2, mutable_data_2]
        )
        assert lengths == [5, 5]

        # 1. Embed feature
        assert READER_EMBED_FEATURE_NAME in data_dictionary
        assert np.allclose(
            data_dictionary[READER_EMBED_FEATURE_NAME],
            np.full(shape=(10, DEFAULT_FEATURE_LENGTH), fill_value=2, dtype=np.float32),
        )

        # 2. Label feature
        assert READER_LABEL_FEATURE_NAME in data_dictionary
        assert np.allclose(
            data_dictionary[READER_LABEL_FEATURE_NAME],
            np.full(shape=(10,), fill_value=2, dtype=np.int64),
        )

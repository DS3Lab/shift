from random import shuffle
from typing import Callable

import pytest
from pydantic.error_wrappers import ValidationError
from schemas.requests.reader import (
    READER_EMBED_FEATURE_NAME,
    READER_LABEL_FEATURE_NAME,
    Feature,
    FeaturesMixin,
    HFReaderConfig,
    ReaderConfig,
    ResultsNumPyReaderConfig,
    Slice,
    TFReaderConfig,
)

from .._base import get_hash


def check_invariant_json(params: dict, gen: Callable[[dict], ReaderConfig]):
    """Checks whether JSONs generated from readers are same when the order of parameters
    is different. Implicitly also checks that reader configs work with correct
    parameters.

    Args:
        params (dict): Parameters passed to the reader config.
        gen (Callable[[dict], ReaderConfig]): Function that given the parameters
            creates a reader config
    """
    # Extract keys
    keys = list(params.keys())

    # Shuffle keys and see whether that affects the output JSON
    jsons = []
    for _ in range(10):
        shuffle(keys)
        reader_config: ReaderConfig = gen({key: params[key] for key in keys})
        jsons.append(reader_config.invariant_json)

    # All JSONs should be the same
    assert len(set(jsons)) == 1


def test_slice_validation():
    slice_ = Slice(start=10, stop=20)

    with pytest.raises(ValidationError):
        slice_.stop = 5

    with pytest.raises(ValidationError):
        slice_.start = 25

    with pytest.raises(ValidationError):
        _ = Slice(start=20, stop=10)


def test_features_mixin_validation_unique_feature_names():
    sample_feature = Feature(store_name="example", path=["example"])

    # 0. Valid
    _ = FeaturesMixin(other_features=[sample_feature])

    # 1. Duplicate feature (duplicate name)
    with pytest.raises(ValidationError):
        _ = FeaturesMixin(other_features=[sample_feature, sample_feature])

    # 2. Use of reserved names
    with pytest.raises(ValidationError):
        _ = FeaturesMixin(
            other_features=[
                Feature(store_name=READER_LABEL_FEATURE_NAME, path=["label"])
            ]
        )

    with pytest.raises(ValidationError):
        _ = FeaturesMixin(
            other_features=[
                Feature(store_name=READER_EMBED_FEATURE_NAME, path=["image"])
            ]
        )


class TestTFReaderConfig:
    @pytest.fixture(scope="function")
    def tf_reader_config_dict(self) -> dict:
        return {
            "slice": Slice(start=10, stop=20),
            "split": "train",
            "tf_dataset_name": "cifar100:3.0.2",
            "embed_feature_path": ["features", "image"],
            "label_feature_path": ["label"],
            "other_features": [Feature(store_name="description", path=["description"])],
        }

    def test_validation_dataset_pinned_version(self, tf_reader_config_dict: dict):
        # 0. Valid
        _ = TFReaderConfig.parse_obj(tf_reader_config_dict)

        # 1. No version at all
        tf_reader_config_dict["tf_dataset_name"] = "cifar100"
        with pytest.raises(ValidationError):
            _ = TFReaderConfig.parse_obj(tf_reader_config_dict)

        # 2. Version partially specified
        tf_reader_config_dict["tf_dataset_name"] = "cifar100:3.*.*"
        with pytest.raises(ValidationError):
            _ = TFReaderConfig.parse_obj(tf_reader_config_dict)

        # 3. Incorrect number version specified
        tf_reader_config_dict["tf_dataset_name"] = "cifar100:3:01:2"
        with pytest.raises(ValidationError):
            _ = TFReaderConfig.parse_obj(tf_reader_config_dict)

    def test_invariant_json(self, tf_reader_config_dict: dict):
        """Testing to see whether there are any problems because of multiple
        inheritance."""
        check_invariant_json(
            tf_reader_config_dict, lambda x: TFReaderConfig.parse_obj(x)
        )

    def test_validation_reader_cannot_be_empty(self):
        # No features specified
        with pytest.raises(ValueError):
            _ = TFReaderConfig(tf_dataset_name="data:0.0.1", split="train")

        # Features specified
        reader = TFReaderConfig(
            tf_dataset_name="data:0.0.1", split="train", embed_feature_path=["image"]
        )

        # Feature removed
        with pytest.raises(ValueError):
            reader.embed_feature_path = None


def test_hf_reader_config_invariant_json():
    check_invariant_json(
        {
            "slice": Slice(start=10, stop=20),
            "split": "train",
            "hf_dataset_name": "glue",
            "configuration": "mrpc",
            "embed_feature_path": ["text"],
            "label_feature_path": ["label"],
            "other_features": [Feature(store_name="description", path=["description"])],
        },
        lambda x: HFReaderConfig.parse_obj(x),
    )


def test_numpy_reader_config_validation_job_id():
    # Valid ID
    _ = ResultsNumPyReaderConfig(embed_feature="image", job_hash=get_hash("test"))

    # Invalid IDs
    # 1. Invalid length
    with pytest.raises(ValidationError):
        _ = ResultsNumPyReaderConfig(embed_feature="image", job_hash="a1234")

    # 2. Invalid characters
    with pytest.raises(ValidationError):
        _ = ResultsNumPyReaderConfig(embed_feature="image", job_hash="A" * 64)


"""
class TestCSVReaderConfig:
    @pytest.fixture(scope="function")
    def csv_reader_config_dict(self) -> dict:
        return {
            "csv_path": "Documents/data.csv",
            "num_columns": 4,
            "num_records": 10_000,
            "embed_column": Column(position=0, type=ColumnType.STRING, name="text"),
            "other_columns": [Column(position=1, type=ColumnType.INT64, name="label")],
        }

    def test_valid_embed_column_position(self, csv_reader_config_dict: dict):
        csv_reader_config_dict["embed_column"].position = csv_reader_config_dict[
            "num_columns"
        ]
        with pytest.raises(ValidationError):
            _ = CSVReaderConfig.parse_obj(csv_reader_config_dict)

    def test_valid_embed_column_type(self, csv_reader_config_dict: dict):
        # csv_reader_config_dict["embed_column"].type = ColumnType.INT64
        csv_reader_config_dict["embed_column"] = Column(
            position=0, type=ColumnType.INT64, name="bla"
        )
        with pytest.raises(ValidationError):
            _ = CSVReaderConfig.parse_obj(csv_reader_config_dict)

    def test_valid_other_columns_position(self, csv_reader_config_dict: dict):
        csv_reader_config_dict["other_columns"][0].position = csv_reader_config_dict[
            "num_columns"
        ]
        with pytest.raises(ValidationError):
            _ = CSVReaderConfig.parse_obj(csv_reader_config_dict)

    def test_compatible_columns_position(self, csv_reader_config_dict: dict):
        csv_reader_config_dict["embed_column"].position = csv_reader_config_dict[
            "other_columns"
        ][0].position
        with pytest.raises(ValidationError):
            _ = CSVReaderConfig.parse_obj(csv_reader_config_dict)

    def test_compatible_columns_name(self, csv_reader_config_dict: dict):
        csv_reader_config_dict["other_columns"][0].name = csv_reader_config_dict[
            "embed_column"
        ].name
        with pytest.raises(ValidationError):
            _ = CSVReaderConfig.parse_obj(csv_reader_config_dict)

    def test_valid_slice(self, csv_reader_config_dict):
        csv_reader_config_dict["slice"] = Slice(start=100, stop=10_001)
        with pytest.raises(ValidationError):
            _ = CSVReaderConfig.parse_obj(csv_reader_config_dict)

    def test_invariant_json(self, csv_reader_config_dict: dict):
        check_invariant_json(
            csv_reader_config_dict, lambda x: CSVReaderConfig.parse_obj(x)
        )
"""

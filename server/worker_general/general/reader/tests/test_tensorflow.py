import functools
from typing import Callable, Optional, Sequence

import pytest
import tensorflow as tf
import tensorflow_datasets.core.features as tf_features
from pipeline import DataType
from pipeline.model import PreprocessingSpecs
from schemas.reader import (
    READER_EMBED_FEATURE_NAME,
    READER_LABEL_FEATURE_NAME,
    ShuffleParams,
    Slice,
    TFReaderConfig,
)

from .._base import check_paths
from .._tensorflow import TFReader, _check_path_and_determine_type, _prepare_dataset


@pytest.fixture(scope="module")
def features_dictionary() -> tf_features.FeaturesDict:
    return tf_features.FeaturesDict(
        {
            "data": tf_features.FeaturesDict(
                {
                    "img_sequence": tf_features.Sequence(
                        feature={
                            "img": tf_features.Image(shape=(28, 28, 3), dtype=tf.uint8)
                        },
                        length=100,
                    ),
                    "sequence": tf_features.Sequence(
                        feature=tf_features.FeaturesDict({"item": tf_features.Text()})
                    ),
                    "img": tf_features.Image(shape=(32, 32, 3), dtype=tf.uint8),
                    "label": tf_features.ClassLabel(num_classes=10),
                    "count": tf.int64,
                }
            ),
            "other": tf_features.FeaturesDict(
                {
                    "additional": tf_features.FeaturesDict(
                        {
                            "extra": tf_features.FeaturesDict(
                                {
                                    "description": tf_features.Text(),
                                    "bbox": tf_features.BBoxFeature(),
                                }
                            )
                        }
                    )
                }
            ),
        }
    )


class TestPathChecking:
    @pytest.fixture(scope="class")
    def check_path_fn(self, features_dictionary) -> Callable[[Sequence[str]], DataType]:
        return functools.partial(
            _check_path_and_determine_type, features_dictionary=features_dictionary
        )

    def test_image_correct_type(self, features_dictionary):
        inferred_type = _check_path_and_determine_type(
            ["data", "img"], features_dictionary
        )
        assert inferred_type == DataType.IMAGE

    def test_text_correct_type(self, features_dictionary):
        inferred_type = _check_path_and_determine_type(
            ["other", "additional", "extra", "description"], features_dictionary
        )
        assert inferred_type == DataType.TEXT

    def test_other_feature_type(self, features_dictionary):
        inferred_type = _check_path_and_determine_type(
            ["other", "additional", "extra", "bbox"], features_dictionary
        )
        assert inferred_type == DataType.OTHER

    def test_non_existent_path(self, features_dictionary):
        # Request for non-existent key before a 'leaf' is reached
        with pytest.raises(KeyError):
            _check_path_and_determine_type(
                ["data", "additional", "img"], features_dictionary
            )

        # Reaches a 'leaf', but additional keys supplied
        with pytest.raises(KeyError):
            _check_path_and_determine_type(["data", "img", "img"], features_dictionary)

    def test_sequence_elements_not_allowed(self, features_dictionary):
        with pytest.raises(ValueError):
            _check_path_and_determine_type(
                ["data", "sequence", "item"], features_dictionary
            )

    def test_sequence_not_allowed(self, features_dictionary):
        with pytest.raises(ValueError):
            _check_path_and_determine_type(
                ["data", "img_sequence"], features_dictionary
            )

    def test_dictionary_not_allowed(self, features_dictionary):
        with pytest.raises(ValueError):
            _check_path_and_determine_type(["other", "additional"], features_dictionary)

    def test_valid_specification(self, check_path_fn):
        embed_type = check_paths(
            check_path_fn,
            ["data", "img"],
            ["data", "label"],
            [["other", "additional", "extra", "description"]],
        )
        assert embed_type == DataType.IMAGE

    def test_invalid_specification(self, check_path_fn):
        with pytest.raises(ValueError):
            check_paths(check_path_fn, ["data", "count"], ["data", "label"], None)


class TestPrepareDataset:
    @pytest.fixture(scope="class")
    def range_10_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(list(range(10))).map(
            lambda x: {READER_EMBED_FEATURE_NAME: x, "other": "..."}
        )

    @pytest.fixture(scope="class")
    def add_1_specs(self) -> PreprocessingSpecs:
        class ExampleSpecs(PreprocessingSpecs):
            def get_tf_preprocessing_fn(self) -> Optional[Callable]:
                return lambda x: x + 1

            def get_pt_preprocessing_fn(self) -> Optional[Callable]:
                pass

        return ExampleSpecs()

    @pytest.mark.parametrize(
        "run_preprocessing,expected", [(False, [[2, 3], [4]]), (True, [[3, 4], [5]])]
    )
    def test_prepare_dataset(
        self,
        range_10_dataset: tf.data.Dataset,
        add_1_specs: PreprocessingSpecs,
        run_preprocessing: bool,
        expected: Sequence[Sequence[int]],
    ):
        iterator = _prepare_dataset(
            range_10_dataset,
            extraction_fn=lambda x: {
                READER_EMBED_FEATURE_NAME: x[READER_EMBED_FEATURE_NAME]
            },
            specs=add_1_specs,
            run_preprocessing=run_preprocessing,
            batch_size=2,
            slice_=Slice(start=2, stop=5),
        )
        assert expected == [
            item[READER_EMBED_FEATURE_NAME].tolist() for item in iterator
        ]


# NOTE: TFReader is altered in conftest.py!
class TestTFReader:
    def test_reader(self, null_specs):
        reader = TFReader(
            TFReaderConfig(
                embed_feature_path=["image"],
                label_feature_path=["label"],
                tf_dataset_name="data:0.0.1",
                split="train",
                shuffle=ShuffleParams(buffer_size=3, seed=5),
            ),
            specs=null_specs,
            batch_size=None,
        )

        assert reader.data_type == DataType.IMAGE
        results = list(reader)
        assert len(results) == 1
        assert (
            READER_EMBED_FEATURE_NAME in results[0]
            and READER_LABEL_FEATURE_NAME in results[0]
        )

    def test_reader_label_only(self, null_specs):
        reader = TFReader(
            TFReaderConfig(
                label_feature_path=["label"],
                tf_dataset_name="data:0.0.1",
                split="train",
            ),
            specs=null_specs,
            batch_size=None,
        )

        assert reader.data_type == DataType.UNKNOWN
        results = list(reader)
        assert len(results) == 1
        assert (
            READER_EMBED_FEATURE_NAME not in results[0]
            and READER_LABEL_FEATURE_NAME in results[0]
        )

    def test_reader_custom_extraction_fn(self, null_specs):
        reader = TFReader(
            TFReaderConfig(
                tf_dataset_name="data:0.0.1", embed_feature_path=["image"], split="test"
            ),
            specs=null_specs,
            batch_size=3,
            # Overrides the retrieval of 'image'
            custom_extraction_fn=lambda x: {READER_LABEL_FEATURE_NAME: x["label"]},
            custom_extraction_fn_run_preprocessing=False,
        )
        assert reader.data_type == DataType.UNKNOWN
        results = list(reader)
        for result in results:
            assert READER_EMBED_FEATURE_NAME not in result
            assert READER_LABEL_FEATURE_NAME in result
        assert [item[READER_LABEL_FEATURE_NAME].tolist() for item in results] == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0],
        ]

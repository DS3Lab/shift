import datasets
import datasets.features as hf_features
import numpy as np
import pytest
from pipeline import DataType
from schemas.reader import (
    READER_EMBED_FEATURE_NAME,
    READER_LABEL_FEATURE_NAME,
    HFReaderConfig,
)

from .._huggingface import HFReader, _check_path_and_determine_type


@pytest.fixture(scope="module")
def features_dictionary() -> hf_features.Features:
    return hf_features.Features(
        {
            "data": {
                "txt_sequence": hf_features.Sequence(
                    feature={"txt": hf_features.Value(dtype="string")},
                    length=100,
                ),
                "txt_list_sequence": [{"txt": hf_features.Value(dtype="string")}],
                "txt": hf_features.Value(dtype="string"),
                "label": hf_features.ClassLabel(num_classes=10),
                "count": hf_features.Value(dtype="int64"),
            },
            "other": {
                "additional": {
                    "extra": {
                        "translation": hf_features.Translation(languages=["en", "de"]),
                        "translation_vl": hf_features.TranslationVariableLanguages(
                            languages=["en", "de"]
                        ),
                        "description": hf_features.Value(dtype="string"),
                    }
                }
            },
        }
    )


class TestPathChecking:
    def test_text_correct_type(self, features_dictionary):
        inferred_type = _check_path_and_determine_type(
            ["data", "txt"], features_dictionary
        )
        assert inferred_type == DataType.TEXT

    def test_translation_correct_type(self, features_dictionary):
        inferred_type_1 = _check_path_and_determine_type(
            ["other", "additional", "extra", "translation", "en"], features_dictionary
        )
        assert inferred_type_1 == DataType.TEXT

        inferred_type_2 = _check_path_and_determine_type(
            ["other", "additional", "extra", "translation_vl", "en"],
            features_dictionary,
        )
        assert inferred_type_2 == DataType.TEXT

    def test_other_feature_type(self, features_dictionary):
        inferred_type_1 = _check_path_and_determine_type(
            ["data", "label"], features_dictionary
        )
        assert inferred_type_1 == DataType.OTHER

        inferred_type_2 = _check_path_and_determine_type(
            ["data", "count"], features_dictionary
        )
        assert inferred_type_2 == DataType.OTHER

    def test_non_existent_path(self, features_dictionary):
        # Request for non-existent key before a 'leaf' is reached
        with pytest.raises(KeyError):
            _check_path_and_determine_type(
                ["data", "additional", "img"], features_dictionary
            )

        # Reaches a 'leaf', but additional keys supplied
        with pytest.raises(KeyError):
            _check_path_and_determine_type(["data", "txt", "txt"], features_dictionary)

    def test_sequence_elements_not_allowed(self, features_dictionary):
        # Actual sequence
        with pytest.raises(ValueError):
            _check_path_and_determine_type(
                ["data", "txt_sequence", "txt"], features_dictionary
            )

        # A list
        with pytest.raises(ValueError):
            _check_path_and_determine_type(
                ["data", "txt_list_sequence", "txt"], features_dictionary
            )

    def test_sequence_not_allowed(self, features_dictionary):
        # Actual sequence
        with pytest.raises(ValueError):
            _check_path_and_determine_type(
                ["data", "txt_sequence"], features_dictionary
            )

        # A list
        with pytest.raises(ValueError):
            _check_path_and_determine_type(
                ["data", "txt_list_sequence"], features_dictionary
            )

    def test_dictionary_not_allowed(self, features_dictionary):
        with pytest.raises(ValueError):
            _check_path_and_determine_type(["other", "additional"], features_dictionary)

    def test_invalid_translation_keys(self, features_dictionary):
        # Invalid language
        with pytest.raises(KeyError):
            _check_path_and_determine_type(
                ["other", "additional", "extra", "translation", "fr"],
                features_dictionary,
            )

        # Additional keys
        with pytest.raises(KeyError):
            _check_path_and_determine_type(
                ["other", "additional", "extra", "translation", "en", "text"],
                features_dictionary,
            )


class TestHFReader:
    @pytest.fixture(autouse=True)
    def patch_load_dataset(self, monkeypatch, features_dictionary):
        """Patches the datasets library such that the mocked dataset is always returned
        (NumPy format)."""

        class MockedDataset:
            def __init__(self):
                self._extraction_fn = None
                self.features = features_dictionary

            def map(self, extraction_fn, *_, **__):
                self._extraction_fn = extraction_fn
                return self

            def set_format(self, _):
                pass

            def __len__(self):
                return 10

            def __getitem__(self, _):
                return self._extraction_fn(
                    {
                        "data": {
                            "txt": np.array("Some text"),
                            "label": np.array(0),
                            "count": np.array(10),
                        },
                        "other": {
                            "additional": {
                                "extra": {
                                    "translation": {
                                        "en": np.array("English text"),
                                        "de": np.array("German text"),
                                    },
                                    "translation_vl": {
                                        "en": np.array("English text"),
                                        "de": np.array("German text"),
                                    },
                                    "description": np.array("description"),
                                }
                            }
                        },
                    }
                )

        def mocked_load_dataset(*_, **__):
            return MockedDataset()

        monkeypatch.setattr(datasets, "load_dataset", mocked_load_dataset)

    def test_reader(self, null_specs):
        reader = HFReader(
            HFReaderConfig(
                embed_feature_path=["data", "txt"],
                label_feature_path=["data", "label"],
                hf_dataset_name="data",
                split="train",
            ),
            specs=null_specs,
            batch_size=None,
        )

        results = list(reader)
        assert len(results) == 1
        assert (
            READER_EMBED_FEATURE_NAME in results[0]
            and READER_LABEL_FEATURE_NAME in results[0]
        )

    def test_reader_label_only(self, null_specs):
        reader = HFReader(
            HFReaderConfig(
                label_feature_path=["data", "label"],
                hf_dataset_name="data",
                split="train",
            ),
            specs=null_specs,
            batch_size=None,
        )

        results = list(reader)
        assert len(results) == 1
        assert (
            READER_EMBED_FEATURE_NAME not in results[0]
            and READER_LABEL_FEATURE_NAME in results[0]
        )

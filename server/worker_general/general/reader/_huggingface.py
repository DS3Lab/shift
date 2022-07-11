import functools
from typing import Dict, Optional, Sequence

import datasets
import datasets.features as hf_features
import numpy as np
from pipeline import DataType
from pipeline.model import PreprocessingSpecs
from schemas import READER_EMBED_FEATURE_NAME
from schemas.requests.reader import HFReaderConfig

from ._base import check_paths, get_extraction_fn
from ._pytorch import PTReader, _DatasetWithLength


def _check_path_and_determine_type(
    feature_path: Sequence[str], features_dictionary: hf_features.Features
) -> DataType:
    """Checks whether the specified path is valid for the selected dataset and infers
    the data type of the feature selected by the path. See
    https://huggingface.co/docs/datasets/features.html for more information.

    Args:
        feature_path (Sequence[str]): Path (sequence of keys) that leads to the target
            feature.
        features_dictionary (hf_features.Features): A specification of the data
            stored withing the HuggingFace dataset.

    Returns:
        DataType: Type of the target feature.
    """
    current_feature = features_dictionary
    for part in feature_path:
        if isinstance(current_feature, hf_features.Sequence) or isinstance(
            current_feature, list
        ):
            raise ValueError("Using parts of Sequence not supported!")
        elif isinstance(current_feature, dict):
            if part not in current_feature:
                raise KeyError(f"Invalid part of path {part!r}")
            current_feature = current_feature[part]

        elif isinstance(current_feature, hf_features.Translation) or isinstance(
            current_feature, hf_features.TranslationVariableLanguages
        ):
            if part not in current_feature.languages:
                raise KeyError(f"Invalid part of path {part!r}")

            # We have reached text
            # If there are additional parts present, an error will be raised
            current_feature = DataType.TEXT
        else:
            raise KeyError(f"Redundant index {part!r} (path too long)")

    is_string = (
        isinstance(current_feature, hf_features.Value)
        and current_feature.dtype == "string"
    ) or current_feature == DataType.TEXT
    if is_string:
        return DataType.TEXT

    if isinstance(current_feature, hf_features.Value) or isinstance(
        current_feature, hf_features.ClassLabel
    ):
        return DataType.OTHER

    raise ValueError(f"Invalid feature {current_feature!r}")


# Note: if the same dataset is downloaded in two processes at the same time, this does
# not crash any of the processes (like for instance TFDS)
class _HuggingFaceDataset(_DatasetWithLength):
    def __init__(self, config: HFReaderConfig, specs: PreprocessingSpecs):
        # 1. Load dataset
        dataset = datasets.load_dataset(
            path=config.hf_dataset_name,
            name=config.configuration,
            split=config.split,
        )

        # 2. Check that specified features are valid
        # For a given path (sequence of keys) checks the path for the SPECIFIC
        # features dictionary returned by the loaded dataset
        check_path_fn = functools.partial(
            _check_path_and_determine_type, features_dictionary=dataset.features
        )
        inferred_data_type = check_paths(
            check_path_fn,
            config.embed_feature_path,
            config.label_feature_path,
            [f.path for f in config.other_features]
            if config.other_features is not None
            else None,
        )
        if inferred_data_type not in {DataType.TEXT, DataType.UNKNOWN}:
            raise ValueError(f"Invalid embed feature type {inferred_data_type!r}")

        # 3. Run basic preprocessing (feature extraction) and prepare preprocessing
        self._preprocessing_fn = specs.get_pt_preprocessing_fn()

        # New columns are added in get_extraction_fn
        # All existing columns are removed via remove_columns
        # Note: this generates a new file!
        self._dataset = dataset.map(
            get_extraction_fn(config.get_features()),
            remove_columns=list(dataset.features.keys()),
        )
        self._dataset.set_format("numpy")
        self._embed_feature_present = config.embed_feature_present

    # Preprocessing happens here rather in map above, so that it is performed while
    # inference is running
    def __getitem__(self, index: np.int64) -> Dict[str, np.ndarray]:
        dictionary = self._dataset[int(index)]

        # Apply preprocessing
        if self._preprocessing_fn is not None and self._embed_feature_present:
            dictionary[READER_EMBED_FEATURE_NAME] = self._preprocessing_fn(
                dictionary[READER_EMBED_FEATURE_NAME]
            )

        return dictionary

    def __len__(self) -> int:
        return len(self._dataset)


class HFReader(PTReader):
    """Loads and prepares a HuggingFace dataset.

    Note: When using HuggingFace Transformers model with the tokenizers, make sure
    that tokenization does not happen in the main process if you use multiple workers.
    There is a warning if that happens and the tokenizer parallelism is turned off.
    See: https://github.com/huggingface/tokenizers/issues/187#issuecomment-635692450
    This limitation does not affect the current code, but must be kept in mind when
    altering the code.

    Args:
        config (HFReaderConfig): Reader configuration.
        specs (PreprocessingSpecs): Preprocessing specification.
        batch_size (int, optional): Batch size; if not specified, maximal possible batch
            size (whole dataset) is used.
    """

    def __init__(
        self,
        config: HFReaderConfig,
        specs: PreprocessingSpecs,
        batch_size: Optional[int],
    ):
        self._config = config
        self._specs = specs
        super().__init__(config, batch_size)

    def _get_dataset(self) -> _DatasetWithLength:
        return _HuggingFaceDataset(config=self._config, specs=self._specs)

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT

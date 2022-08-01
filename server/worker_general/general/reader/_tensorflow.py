import functools
from time import sleep
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets.core.features as tf_features
from celery.utils.log import get_task_logger
from pipeline import DataType
from pipeline.model import PreprocessingSpecs
from pipeline.reader import Reader
from schemas import READER_EMBED_FEATURE_NAME
from schemas.requests.reader import Slice, TFReaderConfig

from .._config import settings
from ._base import check_paths, get_extraction_fn

_logger = get_task_logger(__name__)


def _prepare_dataset(
    data: tf.data.Dataset,
    extraction_fn: Callable,
    specs: PreprocessingSpecs,
    run_preprocessing: bool,
    batch_size: int,
    slice_: Optional[Slice] = None,
) -> Iterator:
    """Prepares the input pipeline in a requested way by chaining operations to the supplied tf.data.Dataset object.

    Args:
        data (tf.data.Dataset): The input dataset.
        extraction_fn (Callable): Function that extracts the features from the dataset such that after the extraction features are present in a dictionary with one level.
        specs (PreprocessingSpecs): Preprocessing specification.
        run_preprocessing (bool): Specifies whether the preprocessing needs to be run. Meant to be set to False when the prepared dataset will not include any feature that will be embedded.
        batch_size (int): Specifies how dataset should be batched.
        slice_ (Slice, optional): Specifies which part of the dataset to use. Default value None denotes the entire dataset.

    Returns:
        Iterator: A prepared dataset that can be iterated over and returns in each iteration a dictionary of NumPy arrays.
    """
    if slice_ is not None:
        data = data.skip(slice_.start).take(slice_.stop - slice_.start)

    def preparation_fn(dictionary: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # Why OK to ignore warning? preprocessing fn will only get executed if it
        # is not None and if embed_feature_name is not None
        # See: https://github.com/python/mypy/issues/2608
        dictionary[READER_EMBED_FEATURE_NAME] = preprocessing_fn(  # type: ignore
            dictionary[READER_EMBED_FEATURE_NAME]  # type: ignore
        )
        return dictionary

    # Disable parallelism if required by preprocessing
    parallelism = 1 if specs.needs_disabled_multithreading else tf.data.AUTOTUNE

    preprocessing_fn = specs.get_tf_preprocessing_fn()
    if preprocessing_fn is not None and run_preprocessing:
        return (
            # Extract features
            data.map(extraction_fn, num_parallel_calls=tf.data.AUTOTUNE)
            # Preprocess the rest feature in parallel
            .map(preparation_fn, num_parallel_calls=parallelism)
            # Batch
            .batch(batch_size)
            # Ensure that GPU does not wait on CPU by preprocessing in advance
            .prefetch(settings.tf_prefetch_size)
            # Return NumPy
            .as_numpy_iterator()
        )
    else:
        return (
            data.map(extraction_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(settings.tf_prefetch_size)
            .as_numpy_iterator()
        )


def _check_path_and_determine_type(
    feature_path: Sequence[str],
    features_dictionary: tf_features.FeaturesDict,
) -> DataType:
    """Checks whether the specified path is valid for the selected dataset and infers
    the data type of the feature selected by the path.

    Args:
        feature_path (Sequence[str]): Path (sequence of keys) that leads to the target
            feature.
        features_dictionary (tf_features.FeaturesDict): A specification of the data
            stored withing the TFDS dataset.

    Returns:
        DataType: Type of the target feature.
    """
    current_feature = features_dictionary
    for part in feature_path:
        if isinstance(current_feature, tf_features.Sequence):
            raise ValueError("Using parts of Sequence not supported!")
        if isinstance(current_feature, tf_features.FeaturesDict):
            if part not in current_feature:
                raise KeyError(f"Invalid part of path {part!r}")
            current_feature = current_feature[part]
        else:
            raise KeyError(f"Redundant path part {part!r} (path too long)")

    if isinstance(current_feature, tf_features.Image):
        return DataType.IMAGE
    if isinstance(current_feature, tf_features.Text):
        return DataType.TEXT
    if isinstance(current_feature, tf_features.FeaturesDict) or isinstance(
        current_feature, tf_features.Sequence
    ):
        raise ValueError(
            "Cannot use a dictionary/list of features as a feature - "
            "specify a path to a specific feature!"
        )
    return DataType.OTHER


class TFReader(Reader):
    """Loads and prepares data using TFDS.

    Args:
        config (TFReaderConfig): Reader configuration.
        specs (PreprocessingSpecs): Preprocessing specification.
        batch_size (int, optional): Batch size; if not specified, maximal possible batch size (whole dataset) is used.
        custom_extraction_fn (Callable, optional): Custom function that overrides the extraction that would use features supplied in the config. Features supplied in the config are ignored. The default value None means that the regular extraction function is used.
        custom_extraction_fn_run_preprocessing (bool): True if custom_extraction_fn returns a READER_EMBED_FEATURE_NAME and preprocessing should be run on that field, False otherwise. This value is used only if custom_extraction_fn is not None. The default value False means that preprocessing is not run.
    """

    def __init__(
        self,
        config: TFReaderConfig,
        specs: PreprocessingSpecs,
        batch_size: Optional[int],
        custom_extraction_fn: Optional[Callable] = None,
        custom_extraction_fn_run_preprocessing: bool = False,
    ):
        try:
            _logger.info("Loading TFDS dataset with config %r", config)
            data, info = self._load_data(config.tf_dataset_name, config.split)

        # Triggered when multiple downloads of same dataset are triggered by different processes. Why is this enough? When there are multiple processes downloading the same data, the first one to finish will succeed, all others will fail when they finish downloading and the files already exist. Tested was the case
        # where the first process is the first to starts downloading, but the second
        # finishes first by throttling the first one. In this case, the first one fails
        # and the second succeeds.
        except tf.errors.AlreadyExistsError:
            _logger.warning(
                "While loading TFDS dataset with config %r, the same dataset was already being downloaded so the loading failed, retrying to load the dataset in 5s",
                config,
            )
            sleep(5)
            _logger.info("Retrying loading TFDS dataset with config %r", config)
            data, info = self._load_data(config.tf_dataset_name, config.split)
        
        # Check that specified features are valid, determine embed feature type
        if custom_extraction_fn is None:
            # For a given path (sequence of keys) checks the path for the SPECIFIC features dictionary returned by the loaded dataset
            check_path_fn = functools.partial(
                _check_path_and_determine_type, features_dictionary=info.features
            )
            self._data_type = check_paths(
                check_path_fn,
                config.embed_feature_path,
                config.label_feature_path,
                [f.path for f in config.other_features]
                if config.other_features is not None
                else None,
            )
        else:
            self._data_type = DataType.UNKNOWN
        self._num_classes = info.features[config.label_feature_path[0]].num_classes

        # Shuffle dataset, never
        """
        data = data.shuffle(
            buffer_size=100000,
            reshuffle_each_iteration=False,
        )
        """

        # Determine batch size
        if batch_size is not None:
            batch_size_to_use = batch_size

        # Use an upper bound of the size which is simply the size of the dataset to
        # have a single batch that holds the entire dataset
        # Note: using tf.int64.max can throw an error
        else:
            batch_size_to_use = info.splits[config.split].num_examples

        # Determine whether preprocessing should be run or not
        # 1. Custom extraction function -> use additional supplied data
        if custom_extraction_fn is not None:
            run_preprocessing = custom_extraction_fn_run_preprocessing
        # 2. Default extraction function -> look whether embed feature was specified
        else:
            run_preprocessing = config.embed_feature_present
        self._data = _prepare_dataset(
            data=data,
            extraction_fn=custom_extraction_fn
            if custom_extraction_fn is not None
            else get_extraction_fn(config.get_features()),
            specs=specs,
            run_preprocessing=run_preprocessing,
            batch_size=batch_size_to_use,
            slice_=config.slice,
        )

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        return next(self._data)

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @staticmethod
    def _load_data(
        dataset_name: str, split: str
    ) -> Tuple[tf.data.Dataset, tfds.core.DatasetInfo]:
        """Loads data using TFDS library."""
        data, info = tfds.load(
            name=dataset_name,
            split=split,
            data_dir=settings.tfds_location,
            shuffle_files=False,
            with_info=True,
            as_supervised=False,
        )
        return data, info

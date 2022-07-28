from typing import Callable, Dict, NamedTuple, Optional

import numpy as np
import tensorflow as tf
from celery.utils.log import get_task_logger
from pipeline import DataType
from pipeline.model import PreprocessingSpecs
from pipeline.reader import Reader
from schemas.requests.reader import (
    READER_EMBED_FEATURE_NAME,
    READER_LABEL_FEATURE_NAME,
    TFReaderConfig,
    VTABNames,
    VTABReaderConfig,
    VTABSplits,
)
from schemas.mapping.vtab import _VTABMapping, _SplitsCalculator, _VTABExtractionFunctions,_VTABSpecs,_VTABPreprocessing
from ._tensorflow import TFReader

_logger = get_task_logger(__name__)

class VTABReader(Reader):
    """Loads and prepares one of the VTAB datasets, which are modified TFDS datasets.
    NOTE: Returned VTAB datasets are not shuffled (they are sometimes in the
    original implementation).

    Args:
        config (VTABReaderConfig): Reader configuration.
        specs (PreprocessingSpecs): Preprocessing specification.
        batch_size (int, optional): Batch size; if not specified, maximal possible batch
            size (whole dataset) is used.
    """

    def __init__(
        self,
        config: VTABReaderConfig,
        specs: PreprocessingSpecs,
        batch_size: Optional[int],
    ):
        try:
            vtab_mapping = _VTABMapping(config.use_feature, config.use_label)
            vtab_specs = vtab_mapping.get_specs(config.vtab_name)
        except ValueError:
            raise ValueError(f"Unable to get VTAB specs for {config.vtab_name!r}")

        _logger.info("Initializing VTAB dataset with config %r", config)
        self._reader = TFReader(
            config=TFReaderConfig(
                tf_dataset_name=vtab_specs.name,
                split=vtab_specs.splits[config.split],
                slice=config.slice,
                # If custom_extraction_fn is not None the two paths below are completely
                # ignored
                # They provide defaults when the custom_extraction_fn does not exist
                embed_feature_path=["image"] if config.use_feature else None,
                label_feature_path=["label"] if config.use_label else None,
            ),
            specs=_VTABPreprocessing(model_preprocessing_specs=specs),
            batch_size=batch_size,
            # None for datasets that do not have a custom extraction function
            custom_extraction_fn=vtab_specs.extraction_fn,
            # VTAB dataset always provides a feature and a label, whether preprocessing
            # is run or not depends solely on the config - whether to use a feature
            custom_extraction_fn_run_preprocessing=config.use_feature,
        )

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        return next(self._reader)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE
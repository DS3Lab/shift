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

_logger = get_task_logger(__name__)


# TODO: license
# TODO: check all params are correct
class _SplitsCalculator:
    """Provides helper functions for calculating how splits should look like depending
    on the specified parameters. The splits are constructed by dividing sets present
    in the dataset into multiple splits (e.g. only training set present, but also test
    and validation sets are required)."""

    @staticmethod
    def train_test_validation() -> Dict[VTABSplits, str]:
        """Splits when training, test and validations sets (splits) are present in the
        dataset.

        Returns:
            Dict[VTABSplits, str]: Constructed splits.
        """
        return {
            VTABSplits.TRAIN: "train",
            VTABSplits.VAL: "validation",
            VTABSplits.TRAINVAL: "train+validation",
            VTABSplits.TEST: "test",
            VTABSplits.TRAIN800: "train[:800]",
            VTABSplits.VAL200: "validation[:200]",
            VTABSplits.TRAIN800VAL200: "train[:800]+validation[:200]",
        }

    @staticmethod
    def train_test(train_size: int, train_percent: int) -> Dict[VTABSplits, str]:
        """Splits when training and test sets (splits) are present in the dataset.

        Returns:
            Dict[VTABSplits, str]: Constructed splits.
        """
        num_train = train_percent * train_size // 100

        return {
            VTABSplits.TRAIN: f"train[:{num_train}]",
            VTABSplits.VAL: f"train[{num_train}:]",
            VTABSplits.TRAINVAL: "train",
            VTABSplits.TEST: "test",
            VTABSplits.TRAIN800: "train[:800]",
            VTABSplits.VAL200: f"train[{num_train}:{num_train + 200}]",
            VTABSplits.TRAIN800VAL200: f"train[:800]+"
            f"train[{num_train}:{num_train + 200}]",
        }

    @staticmethod
    def train(
        train_size: int, train_percent: int, validation_percent: int
    ) -> Dict[VTABSplits, str]:
        """Splits when only the training set (split) is present in the dataset.

        Returns:
            Dict[VTABSplits, str]: Constructed splits.
        """
        num_train = train_percent * train_size // 100
        num_validation = validation_percent * train_size // 100

        return {
            VTABSplits.TRAIN: f"train[:{num_train}]",
            VTABSplits.VAL: f"train[{num_train}:{num_train + num_validation}",
            VTABSplits.TRAINVAL: f"train[:{num_train + num_validation}]",
            VTABSplits.TEST: f"train[{num_train + num_validation}:]",
            VTABSplits.TRAIN800: "train[:800]",
            VTABSplits.VAL200: f"train[{num_train}:{num_train + 200}]",
            VTABSplits.TRAIN800VAL200: f"train[:800]+"
            f"train[{num_train}:{num_train + 200}]",
        }

    @staticmethod
    def smallnorb() -> Dict[VTABSplits, str]:
        """Splits specific to the 'smallnorm' dataset.

        Returns:
            Dict[VTABSplits, str]: Constructed splits.
        """
        return {
            VTABSplits.TRAIN: "train",
            VTABSplits.VAL: "test[:50%]",
            VTABSplits.TRAINVAL: "train+test[:50%]",
            VTABSplits.TEST: "test[50%:]",
            VTABSplits.TRAIN800: "train[:800]",
            VTABSplits.VAL200: "test[:200]",
            VTABSplits.TRAIN800VAL200: "train[:800]+test[:200]",
        }


class _VTABExtractionFunctions:
    """Provides custom extraction functions needed by some of the datasets.

    Args:
        include_embed_feature (bool): Specifies whether the result should include the
            feature that can be embedded.
        include_label (bool): Specifies whether the result should include the label.
    """

    def __init__(self, include_embed_feature: bool, include_label: bool):
        self._include_feature = include_embed_feature
        self._include_label = include_label

    def _filter(self, data: dict):
        if self._include_feature and self._include_label:
            return data
        if self._include_feature:
            return {READER_EMBED_FEATURE_NAME: data[READER_EMBED_FEATURE_NAME]}

        return {READER_LABEL_FEATURE_NAME: data[READER_LABEL_FEATURE_NAME]}

    def clevr_distance(self, x):
        dist = tf.reduce_min(x["objects"]["pixel_coords"][:, 2])
        thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
        label = tf.reduce_max(tf.where((thrs - dist) < 0))
        return self._filter(
            {READER_EMBED_FEATURE_NAME: x["image"], READER_LABEL_FEATURE_NAME: label}
        )

    def clevr_counting(self, x):
        return self._filter(
            {
                READER_EMBED_FEATURE_NAME: x["image"],
                READER_LABEL_FEATURE_NAME: tf.size(x["objects"]["size"]) - 3,
            }
        )

    def dsprites_location(self, x):
        image = tf.tile(x["image"], [1, 1, 3]) * 255
        label = tf.cast(
            tf.math.floordiv(tf.cast(x["label_x_position"], tf.float32), 32.0 / 16),
            tf.int64,
        )
        return self._filter(
            {READER_EMBED_FEATURE_NAME: image, READER_LABEL_FEATURE_NAME: label}
        )

    def dsprites_orientation(self, x):
        image = tf.tile(x["image"], [1, 1, 3]) * 255
        label = tf.cast(
            tf.math.floordiv(tf.cast(x["label_orientation"], tf.float32), 40.0 / 16),
            tf.int64,
        )
        return self._filter(
            {READER_EMBED_FEATURE_NAME: image, READER_LABEL_FEATURE_NAME: label}
        )

    def kitti(self, x):
        vehicles = tf.where(x["objects"]["type"] < 3)  # Car, Van, Truck.
        vehicle_z = tf.gather(params=x["objects"]["location"][:, 2], indices=vehicles)
        vehicle_z = tf.concat([vehicle_z, tf.constant([[1000.0]])], axis=0)
        dist = tf.reduce_min(vehicle_z)
        thrs = np.array([-100.0, 8.0, 20.0, 999.0])
        label = tf.reduce_max(tf.where((thrs - dist) < 0))
        return self._filter(
            {READER_EMBED_FEATURE_NAME: x["image"], READER_LABEL_FEATURE_NAME: label}
        )

    def smallnorb_azimuth(self, x):
        return self._filter(
            {
                READER_EMBED_FEATURE_NAME: tf.tile(x["image"], [1, 1, 3]),
                READER_LABEL_FEATURE_NAME: x["label_azimuth"],
            }
        )

    def smallnorb_elevation(self, x):
        return self._filter(
            {
                READER_EMBED_FEATURE_NAME: tf.tile(x["image"], [1, 1, 3]),
                READER_LABEL_FEATURE_NAME: x["label_elevation"],
            }
        )


class _VTABSpecs(NamedTuple):
    """Specifies the details of a VTAB dataset."""

    name: str
    splits: Dict[VTABSplits, str]
    extraction_fn: Optional[Callable] = None


class _VTABMapping:
    """Provides a mapping from a VTAB name to details needed to properly load and
    prepare the dataset.

    Args:
        include_embed_feature (bool): Specifies whether the result should include the
            feature that can be embedded.
        include_label (bool): Specifies whether the result should include the label.
    """

    def __init__(self, include_embed_feature: bool, include_label: bool):
        self._extraction_fns = _VTABExtractionFunctions(
            include_embed_feature, include_label
        )

    def get_specs(self, name: VTABNames) -> _VTABSpecs:
        if name == VTABNames.CALTECH_101:
            return _VTABSpecs(
                name="caltech101:3.0.1",
                splits=_SplitsCalculator.train_test(train_size=3_060, train_percent=90),
            )

        if name == VTABNames.CIFAR_100:
            return _VTABSpecs(
                name="cifar100:3.0.2",
                splits=_SplitsCalculator.train_test(
                    train_size=50_000, train_percent=90
                ),
            )

        if name == VTABNames.CLEVR_DISTANCE_PREDICTION:
            return _VTABSpecs(
                name="clevr:3.1.0",
                splits=_SplitsCalculator.train_test(
                    train_size=70_000, train_percent=90
                ),
                extraction_fn=self._extraction_fns.clevr_distance,
            )

        if name == VTABNames.CLEVR_COUNTING:
            return _VTABSpecs(
                name="clevr:3.1.0",
                splits=_SplitsCalculator.train_test(
                    train_size=70_000, train_percent=90
                ),
                extraction_fn=self._extraction_fns.clevr_counting,
            )
        if name == VTABNames.CIFAR_10:
            return _VTABSpecs(
                name="cifar10:3.0.2",
                splits=_SplitsCalculator.train_test(
                    train_size=50_000, train_percent=90
                ),
            )
        if name == VTABNames.DIABETIC_RETHINOPATHY:
            return _VTABSpecs(
                name="diabetic_retinopathy_detection/btgraham-300:3.0.0",
                splits=_SplitsCalculator.train_test_validation(),
            )

        if name == VTABNames.DMLAB:
            return _VTABSpecs(
                name="dmlab:2.0.1", splits=_SplitsCalculator.train_test_validation()
            )

        if name == VTABNames.DSPRITES_ORIENTATION_PREDICTION:
            return _VTABSpecs(
                name="dsprites:2.0.0",
                splits=_SplitsCalculator.train(
                    train_size=737_280, train_percent=80, validation_percent=10
                ),
                extraction_fn=self._extraction_fns.dsprites_orientation,
            )

        if name == VTABNames.DSPRITES_LOCATION_PREDICTION:
            return _VTABSpecs(
                name="dsprites:2.0.0",
                splits=_SplitsCalculator.train(
                    train_size=737_280, train_percent=80, validation_percent=10
                ),
                extraction_fn=self._extraction_fns.dsprites_location,
            )

        if name == VTABNames.DTD:
            return _VTABSpecs(
                name="dtd:3.0.1", splits=_SplitsCalculator.train_test_validation()
            )

        if name == VTABNames.EUROSAT:
            return _VTABSpecs(
                name="eurosat/rgb:2.0.0",
                splits=_SplitsCalculator.train(
                    train_size=27_000, train_percent=60, validation_percent=20
                ),
            )

        if name == VTABNames.KITTI_DISTANCE_PREDICTION:
            return _VTABSpecs(
                name="kitti:3.2.0",
                splits=_SplitsCalculator.train_test_validation(),
                extraction_fn=self._extraction_fns.kitti,
            )

        if name == VTABNames.OXFORD_FLOWERS:
            return _VTABSpecs(
                name="oxford_flowers102:2.1.1",
                splits=_SplitsCalculator.train_test_validation(),
            )

        if name == VTABNames.OXFORD_PET:
            return _VTABSpecs(
                name="oxford_iiit_pet:3.2.0",
                splits=_SplitsCalculator.train_test(train_size=3_680, train_percent=80),
            )

        if name == VTABNames.PATCH_CAMELYON:
            return _VTABSpecs(
                name="patch_camelyon:2.0.0",
                splits=_SplitsCalculator.train_test_validation(),
            )

        if name == VTABNames.RESISC45:
            return _VTABSpecs(
                name="resisc45:3.0.0",
                splits=_SplitsCalculator.train(
                    train_size=31_500, train_percent=60, validation_percent=20
                ),
            )

        if name == VTABNames.SMALLNORB_AZIMUTH_PREDICTION:
            return _VTABSpecs(
                name="smallnorb:2.0.0",
                splits=_SplitsCalculator.smallnorb(),
                extraction_fn=self._extraction_fns.smallnorb_azimuth,
            )

        if name == VTABNames.SMALLNORB_ELEVATION_PREDICTION:
            return _VTABSpecs(
                name="smallnorb:2.0.0",
                splits=_SplitsCalculator.smallnorb(),
                extraction_fn=self._extraction_fns.smallnorb_elevation,
            )

        if name == VTABNames.SUN397:
            return _VTABSpecs(
                name="sun397/tfds:4.0.0",
                splits=_SplitsCalculator.train_test_validation(),
            )

        if name == VTABNames.SVHN:
            return _VTABSpecs(
                name="svhn_cropped:3.0.0",
                splits=_SplitsCalculator.train_test(
                    train_size=73_257, train_percent=90
                ),
            )
        # below are customized datasets
        if name == VTABNames.CIFAR_9:
            return _VTABSpecs(
                name="cifarn:3.0.3",
                splits=_SplitsCalculator.train_test(
                    train_size=45_000, train_percent=90
                ),
            )
        if name == VTABNames.CIFAR_10Ordered:
            return _VTABSpecs(
                name="cifar10_ordered:1.0.0",
                splits=_SplitsCalculator.train_test(
                    train_size=50_000, train_percent=90
                ),
            )
        if name == VTABNames.CIFAR_9TH:
            return _VTABSpecs(
                name="cifar9_th:1.0.0",
                splits=_SplitsCalculator.train_test(
                    train_size=5000, train_percent=90
                ),
            )
        raise ValueError(f"Unknown key {name!r}")


class _VTABPreprocessing(PreprocessingSpecs):
    """Preprocesses the data such that it is suitable for a VTAB dataset. If model
    preprocessing is specified as well, the preprocessing is composed of the VTAB
    preprocessing and the model preprocessing.

    Args:
        model_preprocessing_specs (PreprocessingSpecs): Preprocessing specified by the
            model. This preprocessing is applied after the VTAB preprocessing and MUST
            BE INVARIANT with respect to image type (tf.uint8 or tf.float32), since the
            images are already preprocessed and thus of type tf.float32.
    """

    def __init__(self, model_preprocessing_specs: PreprocessingSpecs):
        self._model_tf_preprocessing_fn = (
            model_preprocessing_specs.get_tf_preprocessing_fn()
        )

    # https://github.com/google-research/task_adaptation/blob/
    # ed9b417768c38c38e1fea8194d23442bd78eeafd/task_adaptation/data_loader.py#L61
    @staticmethod
    def _preprocessing_fn(image):
        return tf.cast(tf.image.resize(image, [224, 224]), tf.float32) / 255.0

    def _nested_preprocess_fn(self, image):
        return self._model_tf_preprocessing_fn(self._preprocessing_fn(image))

    def get_tf_preprocessing_fn(self) -> Callable:
        if self._model_tf_preprocessing_fn is None:
            return self._preprocessing_fn

        return self._nested_preprocess_fn

    # Never used
    def get_pt_preprocessing_fn(self) -> Optional[Callable]:
        pass

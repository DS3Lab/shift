from typing import Callable, Optional, Sequence

import pytest
import tensorflow as tf
from pipeline.model import PreprocessingSpecs
from schemas.model import ImageSize
from schemas.requests.reader import (
    READER_EMBED_FEATURE_NAME,
    READER_LABEL_FEATURE_NAME,
    VTABNames,
    VTABReaderConfig,
    VTABSplits,
)
from tensorflow.python.framework import random_seed

from ...model.preprocessing import ImageCropResize3Channels, ImageCropResizeFlatten
from .._vtab import VTABReader, _VTABMapping, _VTABPreprocessing


class TestVTABPreprocessingCompatibility:
    """Test whether using VTABPreprocessing in combination with preprocessing requested
    by a model yields correct results. The main goal is to test whether preprocessing
    functions are invariant when used in combination with additional preprocessing that
    happens before. The invariance is tested by checking after the image has
    been resized to size (224, 224, *) according to VTAB whether applying additional
    preprocessing (where the target size is again (224, 224, *)) alters the result.
    Non-square images are not tested, because the regular preprocessing crops the
    images, whereas VTAB preprocessing stretches/shrinks them. Resizing to sizes
    different than (224, 224, *) is not tested, since resizing an image multiple times
    does not give the same result.
    """

    _target_image_size = ImageSize(height=224, width=224)

    @pytest.fixture(scope="class")
    def random_images(self) -> Sequence[tf.Tensor]:
        """Generates a random sequence of square TensorFlow images of different size.

        Returns:
            Sequence[tf.Tensor]: A sequence of random square images.
        """
        random_seed.set_seed(0)
        random_images = []
        for _ in range(20):
            size = tf.random.uniform(shape=(1,), minval=1, maxval=500, dtype=tf.int32)[
                0
            ]
            random_images.append(
                tf.cast(
                    tf.random.uniform(
                        shape=(size, size, 3), minval=0, maxval=255, dtype=tf.int32
                    ),
                    dtype=tf.uint8,
                )
            )
        return random_images

    @staticmethod
    def check_functions_identical(
        fn_1: Callable, fn_2: Callable, images: Sequence[tf.Tensor]
    ):
        """Checks whether two functions process the provided images in a same way.

        Args:
            fn_1 (Callable): First function.
            fn_2 (Callable): Second function.
            images (Sequence[tf.Tensor]): Images to process.
        """
        for image in images:
            tf.debugging.assert_near(fn_1(image), fn_2(image))

    def test_icr3c_no_normalization_invariance(self, random_images):
        icr3c_no_normalization = ImageCropResize3Channels(
            self._target_image_size, normalize=False
        )
        vtab = _VTABPreprocessing(icr3c_no_normalization)
        self.check_functions_identical(
            icr3c_no_normalization.get_tf_preprocessing_fn(),
            vtab.get_tf_preprocessing_fn(),
            random_images,
        )

    def test_icr3c_normalization_invariance(self, random_images):
        icr3c_normalization = ImageCropResize3Channels(
            self._target_image_size, normalize=True
        )
        vtab = _VTABPreprocessing(icr3c_normalization)
        self.check_functions_identical(
            icr3c_normalization.get_tf_preprocessing_fn(),
            vtab.get_tf_preprocessing_fn(),
            random_images,
        )

    def test_icrf_invariance(self, random_images):
        icrf = ImageCropResizeFlatten(self._target_image_size)
        vtab = _VTABPreprocessing(icrf)
        self.check_functions_identical(
            icrf.get_tf_preprocessing_fn(),
            vtab.get_tf_preprocessing_fn(),
            random_images,
        )

    def test_noop_invariance(self, random_images):
        """Tests expected behaviour of default VTAB preprocessing."""

        class NoOp(PreprocessingSpecs):
            def get_tf_preprocessing_fn(self) -> Optional[Callable]:
                pass

            def get_pt_preprocessing_fn(self) -> Optional[Callable]:
                pass

        vtab = _VTABPreprocessing(NoOp())
        icr3c = ImageCropResize3Channels(
            required_image_size=ImageSize(height=224, width=224)
        )
        self.check_functions_identical(
            icr3c.get_tf_preprocessing_fn(),
            vtab.get_tf_preprocessing_fn(),
            random_images,
        )


class TestVTABSpecs:
    @pytest.fixture(scope="class")
    def data(self) -> dict:
        length = 10
        data = {
            "image": tf.zeros(shape=(10, 10, 1), dtype=tf.uint8),
            "objects": {
                "pixel_coords": tf.zeros(shape=(length, 3), dtype=tf.float32),
                "location": tf.zeros(shape=(length, 3), dtype=tf.float32),
                "size": tf.ones(shape=(length,), dtype=tf.int64),
                "type": tf.ones(shape=(length,), dtype=tf.int64),
            },
            "label_azimuth": tf.ones(shape=(length,), dtype=tf.int64),
            "label_elevation": tf.ones(shape=(length,), dtype=tf.int64),
            "label_x_position": tf.ones(shape=(length,), dtype=tf.int64),
            "label_orientation": tf.ones(shape=(length,), dtype=tf.int64),
        }

        return data

    @pytest.mark.parametrize(
        "include_embed_feature,include_label",
        [(True, True), (True, False), (False, True)],
    )
    def test_all_specs(self, include_embed_feature, include_label, data):
        mapping = _VTABMapping(include_embed_feature, include_label)
        for name in VTABNames:
            specs = mapping.get_specs(VTABNames(name))
            if specs.extraction_fn is not None:
                result = specs.extraction_fn(data)
                # Check that expected data is present
                assert (READER_EMBED_FEATURE_NAME in result) == include_embed_feature
                assert (READER_LABEL_FEATURE_NAME in result) == include_label


# NOTE: TFReader is altered in conftest.py!
class TestVTABReader:
    def test_reader_custom_extraction_fn(self, null_specs):
        reader = VTABReader(
            config=VTABReaderConfig(
                vtab_name=VTABNames.CLEVR_COUNTING,
                split=VTABSplits.TRAIN800VAL200,
                use_feature=True,
                use_label=False,
            ),
            specs=null_specs,
            batch_size=3,
        )
        for batch in reader:
            assert READER_EMBED_FEATURE_NAME in batch
            assert READER_LABEL_FEATURE_NAME not in batch

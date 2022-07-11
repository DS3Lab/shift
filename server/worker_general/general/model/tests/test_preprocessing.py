import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from pipeline.model import PreprocessingSpecs
from schemas.models.image_model import ImageSize

from ..preprocessing import ImageCropResize3Channels, ImageCropResizeFlatten

np.random.seed(0)

# NumPy images
square_image_ndarray = np.random.randint(
    low=0, high=255, size=(10, 10, 3), dtype=np.uint8
)
horizontal_image_ndarray = np.random.randint(
    low=0, high=255, size=(15, 10, 3), dtype=np.uint8
)
vertical_image_ndarray = np.random.randint(
    low=0, high=255, size=(5, 10, 3), dtype=np.uint8
)
one_channel_image_ndarray = np.random.randint(
    low=0, high=255, size=(10, 12), dtype=np.uint8
)

# PIL images
square_image_pil = Image.fromarray(square_image_ndarray)
horizontal_image_pil = Image.fromarray(horizontal_image_ndarray)
vertical_image_pil = Image.fromarray(vertical_image_ndarray)
one_channel_image_pil = Image.fromarray(one_channel_image_ndarray)

# TensorFlow images
square_image_tf = tf.convert_to_tensor(square_image_ndarray)
horizontal_image_tf = tf.convert_to_tensor(horizontal_image_ndarray)
vertical_image_tf = tf.convert_to_tensor(vertical_image_ndarray)
one_channel_image_tf = tf.convert_to_tensor(
    np.expand_dims(one_channel_image_ndarray, axis=2)
)

image_pairs = [
    (square_image_pil, square_image_tf),
    (horizontal_image_pil, horizontal_image_tf),
    (vertical_image_pil, vertical_image_tf),
    (one_channel_image_pil, one_channel_image_tf),
]


def assert_functions_equal(
    specs: PreprocessingSpecs, pil_image: Image, tf_image: tf.Tensor
):
    tf_fun = specs.get_tf_preprocessing_fn()
    pt_fun = specs.get_pt_preprocessing_fn()
    assert (
        tf_fun is not None and pt_fun is not None
    ), "Preprocessing functions should not be None"

    tf_result = tf_fun(tf_image).numpy()
    pt_result = pt_fun(pil_image)

    np.allclose(tf_result, pt_result)


@pytest.mark.parametrize("pil_image,tf_image", image_pairs)
def test_icr3c_square_resize(pil_image: Image, tf_image: tf.Tensor):
    icr3c = ImageCropResize3Channels(required_image_size=ImageSize(height=10, width=10))
    assert_functions_equal(icr3c, pil_image, tf_image)


@pytest.mark.parametrize("pil_image,tf_image", image_pairs)
def test_icr3c_rectangular_resize(pil_image: Image, tf_image: tf.Tensor):
    icr3c = ImageCropResize3Channels(required_image_size=ImageSize(height=20, width=30))
    assert_functions_equal(icr3c, pil_image, tf_image)


@pytest.mark.parametrize("pil_image,tf_image", image_pairs)
def test_icr3c_normalization(pil_image: Image, tf_image: tf.Tensor):
    icr3c = ImageCropResize3Channels(
        required_image_size=ImageSize(height=20, width=30), normalize=True
    )
    assert_functions_equal(icr3c, pil_image, tf_image)


@pytest.mark.parametrize("pil_image,tf_image", image_pairs)
def test_icrf_square_resize(pil_image: Image, tf_image: tf.Tensor):
    icr3c = ImageCropResizeFlatten(target_image_size=ImageSize(height=10, width=10))
    assert_functions_equal(icr3c, pil_image, tf_image)


@pytest.mark.parametrize("pil_image,tf_image", image_pairs)
def test_icrf_rectangular_resize(pil_image: Image, tf_image: tf.Tensor):
    icr3c = ImageCropResizeFlatten(target_image_size=ImageSize(height=20, width=30))
    assert_functions_equal(icr3c, pil_image, tf_image)

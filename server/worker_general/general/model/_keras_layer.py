import numpy as np
import tensorflow as tf
from pipeline import DataType, Device
from pipeline.model import Model, PreprocessingSpecs
from schemas.models.image_model import ImageKerasLayerConfig
from schemas.models.text_model import TextKerasLayerConfig
from tensorflow import keras
from loguru import logger

from .._config import settings
from .preprocessing import ImageCropResize3Channels, TextNoOpPreprocessing

class ImageKerasLayer(Model):
    """Runs inference with an image Keras layer.

    Args:
        config (ImageKerasLayerConfig): Model configuration.
        device (Device): Device used for the inference.
    """

    def __init__(self, config: ImageKerasLayerConfig, device: Device):
        self._device_string = "GPU:0" if device == Device.GPU else "CPU"
        with tf.device(self._device_string):
            self._layer = keras.models.load_model(
                settings.get_input_path_str(config.image_layer_path)
            )
        self._required_image_size = config.required_image_size

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return ImageCropResize3Channels(self._required_image_size)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        with tf.device(self._device_string):
            tf_tensor = tf.convert_to_tensor(features)
            return self._layer(tf_tensor).numpy()


class TextKerasLayer(Model):
    """Runs inference with a text Keras layer.

    Args:
        config (ImageKerasLayerConfig): Model configuration.
        device (Device): Device used for the inference.
    """

    def __init__(self, config: TextKerasLayerConfig, device: Device):
        self._device_string = "GPU:0" if device == Device.GPU else "CPU"
        with tf.device(self._device_string):
            self._layer = keras.models.load_model(
                settings.get_input_path_str(config.text_layer_path)
            )

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return TextNoOpPreprocessing()

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        with tf.device(self._device_string):
            tf_tensor = tf.convert_to_tensor(features)
            logger.info(f"tf tensor shape: {tf_tensor.shape}")
            return self._layer(tf_tensor).numpy()

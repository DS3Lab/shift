from typing import Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from celery.utils.log import get_task_logger
from pipeline import DataType, Device
from pipeline.model import Model, PreprocessingSpecs
from schemas.models import TFFullImageModelConfig, TFFullTextModelConfig
from schemas.models.image_model import FinetunedTFFullImageModelConfig

from .preprocessing import ImageCropResize3Channels, TextNoOpPreprocessing

logger = get_task_logger(__name__)


class _TFModel(Model):
    def __init__(
        self,
        url: str,
        output_key: Optional[str],
        device: Device,
    ):
        # Note: OK to download it in parallel - does not cause an error like TFDS
        self._device_string = "GPU:0" if device == Device.GPU else "CPU"
        logger.info("Loading model from {}...".format(url))
        # Load a model
        with tf.device(self._device_string):
            self._model = hub.KerasLayer(
                url,
                trainable=False,
                # Used only in TF1 Hub format
                output_key=output_key,
            )

        # Determine the output shape
        # output_shape_tensor = self._model.compute_output_shape(None)
        # if self._output_signature:
        #     output_shape_tensor = output_shape_tensor[self._output_signature]
        # self._output_dimension = self._get_output_dimension(
        #     output_shape_tensor.as_list()
        # )

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        with tf.device(self._device_string):
            tf_tensor = tf.convert_to_tensor(features)
            tf_result = self._model(tf_tensor).numpy()
        return self._after_inference_hook(tf_result)

    # # Meant to be overridden
    # @staticmethod
    # def _get_output_dimension(shape: List[int]) -> int:
    #     if len(shape) != 2 or shape[0] is not None or not isinstance(shape[1], int):
    #         raise RuntimeError(
    #             f"Embedding outputs an embedding of wrong shape! "
    #             f"Expected is a one-dimensional embedding, found: {shape}"
    #         )
    #     return shape[1]

    # Meant for cases when one needs to process the obtained embedding
    # For instance when you get embedding per token instead of pooled version
    @staticmethod
    def _after_inference_hook(x: np.ndarray) -> np.ndarray:
        return x


class TFImageModel(_TFModel):
    """Runs inference with a TensorFlow image model.

    Args:
        config (TFFullImageModelConfig or FinetunedTFFullImageModelConfig): Model configuration.
        device (Device): Device used for the inference.
    """

    def __init__(
        self,
        config: Union[TFFullImageModelConfig, FinetunedTFFullImageModelConfig],
        device: Device,
    ):
        super().__init__(
            url=str(config.tf_image_model_url),
            output_key=config.output_key,
            device=device,
        )
        self._required_image_size = config.required_image_size

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return ImageCropResize3Channels(self._required_image_size)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


class TFTextModel(_TFModel):
    """Runs inference with a TensorFlow text model.

    Args:
        config (TFFullTextModelConfig): Model configuration.
        device (Device): Device used for the inference.
    """

    def __init__(self, config: TFFullTextModelConfig, device: Device):
        super().__init__(
            url=str(config.tf_text_model_url),
            output_key=config.output_key,
            device=device,
        )

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return TextNoOpPreprocessing()

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT

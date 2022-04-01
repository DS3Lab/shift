from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import torch as pt
import torchvision.transforms as transforms
from PIL.Image import Image
from pipeline.model import PreprocessingSpecs
from schemas.models.image_model import ImageSize
from torchvision.transforms.functional import center_crop
from transformers import BertTokenizerFast, XLNetTokenizerFast

__all__ = [
    "ImageCropResizeFlatten",
    "ImageCropResize3Channels",
    "TextNoOpPreprocessing",
    "HFPreprocessing",
]


class _ImagePreparer:
    """Prepares an image tensor by performing a center crop and optionally replicating
    channels if there is only one channel.

    Args:
        enforce_3_channels (bool): True if there should always be 3 channels, False
            otherwise.
    """

    def __init__(self, enforce_3_channels: bool = False):
        self._enforce_3_channels = enforce_3_channels

    def __call__(self, image: pt.Tensor):
        num_channels, height, width = image.shape
        min_dim = min(height, width)
        intermediate = center_crop(image, min_dim)

        if num_channels not in {1, 3}:
            raise RuntimeError(
                f"Incorrect number of image channels. "
                f"Found: {num_channels}, expected: 1 or 3"
            )

        # Create additional channels if needed
        if num_channels == 1 and self._enforce_3_channels:
            return intermediate.repeat((3, 1, 1))

        return intermediate


class _DimensionSwitcher:
    """Switches dimensions of an image tensor from (#channels, height, width) to
    (height, width, #channels)."""

    def __call__(self, image: pt.Tensor):
        return image.permute(1, 2, 0)


class _TFImageHelper:
    """Contains methods for manipulating images with TensorFlow."""

    @staticmethod
    def central_crop_with_resize_to_float(
        feature: tf.Tensor, required_image_size: Tuple[int, int]
    ) -> tf.Tensor:
        converted_img = tf.image.convert_image_dtype(
            feature, dtype=tf.float32, saturate=False
        )
        converted_img /= 255.0
        shape = tf.shape(converted_img)
        min_dim = tf.minimum(shape[0], shape[1])
        cropped_img = tf.image.resize_with_crop_or_pad(converted_img, min_dim, min_dim)
        return tf.image.resize(cropped_img, required_image_size)

    @staticmethod
    def central_crop_with_resize(
        feature: tf.Tensor, required_image_size: Tuple[int, int]
    ) -> tf.Tensor:
        converted_img = tf.image.convert_image_dtype(
            feature, dtype=tf.float32, saturate=False
        )
        shape = tf.shape(converted_img)
        min_dim = tf.minimum(shape[0], shape[1])
        cropped_img = tf.image.resize_with_crop_or_pad(converted_img, min_dim, min_dim)
        return tf.image.resize(cropped_img, required_image_size)

    @staticmethod
    def central_crop_with_resize_3_channels(
        feature: tf.Tensor, required_image_size: Tuple[int, int]
    ) -> tf.Tensor:
        resized_img = _TFImageHelper.central_crop_with_resize_to_float(
            feature, required_image_size
        )
        # For 1 channel, repeats 3 times; for 3 channels, repeats 1 time
        return tf.repeat(resized_img, 3 - tf.shape(resized_img)[2] + 1, axis=2)

    @staticmethod
    def central_crop_with_resize_3_channels_normalized(
        feature: tf.Tensor, required_image_size: Tuple[int, int]
    ):
        intermediate = _TFImageHelper.central_crop_with_resize_3_channels(
            feature, required_image_size
        )
        return tf.divide(
            tf.subtract(
                intermediate,
                tf.constant([0.485, 0.456, 0.406], dtype=tf.float32),
            ),
            tf.constant([0.229, 0.224, 0.225], dtype=tf.float32),
        )

    @staticmethod
    def raw_image_with_central_crop_and_resize(
        feature: tf.Tensor, required_image_size: Tuple[int, int]
    ) -> tf.Tensor:
        resized_img = _TFImageHelper.central_crop_with_resize(
            feature, required_image_size
        )
        # Must be a tuple!
        return tf.reshape(resized_img, (-1,))


class ImageCropResizeFlatten(PreprocessingSpecs):
    """Performs a central crop, a resize and flattening. All images are transformed into vectors of the same length, since after crop and resize operations all images are of same size.

    Args:
        target_image_size (ImageSize): Image size to which images will be resized.
    """

    def __init__(self, target_image_size: ImageSize):
        self._target_image_size = (target_image_size.height, target_image_size.width)

    def get_tf_preprocessing_fn(self) -> Callable[[tf.Tensor], tf.Tensor]:
        return lambda x: _TFImageHelper.raw_image_with_central_crop_and_resize(
            x, self._target_image_size
        )

    def get_pt_preprocessing_fn(self) -> Callable[[Image], pt.Tensor]:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                _ImagePreparer(enforce_3_channels=False),
                transforms.Resize(self._target_image_size),
                _DimensionSwitcher(),
                pt.flatten,
            ]
        )


class ImageCropResize3Channels(PreprocessingSpecs):
    """Creates 3 channels if there is only one channel, performs a central crop,
    a resize and optionally also normalization.

    Args:
        required_image_size (ImageSize): Image size to which images will be resized.
        normalize (bool): True if normalization should be performed, False otherwise.
            The default value False means that normalization is not performed.
    """

    def __init__(self, required_image_size: ImageSize, normalize: bool = False):
        self._required_image_size = (
            required_image_size.height,
            required_image_size.width,
        )
        self._normalize = normalize

    def get_tf_preprocessing_fn(self) -> Callable[[tf.Tensor], tf.Tensor]:
        if self._normalize:
            return (
                lambda x: _TFImageHelper.central_crop_with_resize_3_channels_normalized(
                    x, self._required_image_size
                )
            )

        return lambda x: _TFImageHelper.central_crop_with_resize_3_channels(
            x, self._required_image_size
        )

    def get_pt_preprocessing_fn(self) -> Callable[[Image], pt.Tensor]:
        if self._normalize:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    _ImagePreparer(enforce_3_channels=True),
                    transforms.Resize(self._required_image_size),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    _DimensionSwitcher(),
                ]
            )

        return transforms.Compose(
            [
                transforms.ToTensor(),
                _ImagePreparer(enforce_3_channels=True),
                transforms.Resize(self._required_image_size),
                _DimensionSwitcher(),
            ]
        )


class TextNoOpPreprocessing(PreprocessingSpecs):
    """Performs no preprocessing."""

    def get_tf_preprocessing_fn(self) -> None:
        pass

    def get_pt_preprocessing_fn(self) -> None:
        pass


class HFPreprocessing(PreprocessingSpecs):
    """Preprocessing (tokenization) for the HuggingFace Transformers models
    (BERT and XLNet)."""

    def __init__(self, name: str, max_length: int, tokenizer_params: dict):
        self._max_length = max_length
        if "bert" in name:
            self._tokenizer = BertTokenizerFast.from_pretrained(
                name, **tokenizer_params
            )
        elif "xlnet" in name:
            self._tokenizer = XLNetTokenizerFast.from_pretrained(
                name, **tokenizer_params
            )
        else:
            raise NotImplementedError(f"Cannot find tokenizer for model {name!r}")

    # See: https://github.com/huggingface/tokenizers/issues/537#issuecomment-733118900
    # In future this might be resolved, but for now TensorFlow dataset must not
    # preprocess in parallel
    # The alternative seems to be to use the regular tokenizers (not fast)
    @property
    def needs_disabled_multithreading(self) -> bool:
        return True

    def _tokenize(self, feature: str) -> np.ndarray:
        """Performs tokenization using HuggingFace tokenizers.

        Args:
            feature (str): String that needs to be tokenized.

        Returns:
            np.ndarray: Tokenized input with shape ``3 x max_length`` specified in the
            :py:func:`__init__`.
        """
        tokenizer_dict = self._tokenizer(
            feature,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self._max_length,  # Pad & truncate all sentences.
            padding="max_length",
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_token_type_ids=True,
            return_tensors="np",
        )

        return np.vstack(
            (
                tokenizer_dict["input_ids"].reshape(-1),
                tokenizer_dict["attention_mask"].reshape(-1),
                tokenizer_dict["token_type_ids"].reshape(-1),
            )
        )

    def _tokenize_tf(self, feature: tf.Tensor) -> tf.Tensor:
        return tf.constant(
            self._tokenize(feature.numpy().decode("UTF-8")), dtype=tf.int64
        )

    def get_tf_preprocessing_fn(self) -> Callable[[tf.Tensor], tf.Tensor]:
        return lambda feature: tf.py_function(
            func=self._tokenize_tf, inp=[feature], Tout=tf.int64
        )

    def _tokenize_pt(self, feature: np.ndarray) -> np.ndarray:
        return self._tokenize(str(feature.item()))

    def get_pt_preprocessing_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._tokenize_pt

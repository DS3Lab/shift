from abc import ABC
from typing import Callable, List, Optional

from pydantic import BaseModel, Field, HttpUrl
from schemas._base import _DefaultConfig
from schemas.models.common import (
    FullModelConfig,
    ModelInfo,
    Source,
    _layer_path_field,
)

class ImageSize(BaseModel):
    height: int = Field(
        ...,
        title="Image height",
        description="To what height images should be resized",
        example=224,
        ge=1,
    )
    width: int = Field(
        ...,
        title="Image width",
        description="To what width images should be resized",
        example=224,
        ge=1,
    )

    class Config(_DefaultConfig):
        pass


_image_resize_field = Field(
    ...,
    title="Image resize",
    description="To what sizes images should be resized",
)


class ImageFullModelConfig(FullModelConfig, ABC):
    @property
    def invariant_json(self) -> str:
        return self.json(exclude={"info"})


class ImageModelInfo(ModelInfo):
    image_size: Optional[int] = Field(
        None,
        title="Image size",
        description="Target image size for the model",
        example=224,
        ge=1,
        le=10 ** 15,
    )

    class Config(_DefaultConfig):
        title = "Information about the image model"


class ReshapeModelConfig(ImageFullModelConfig):
    reshape_image_size: ImageSize = _image_resize_field

    @property
    def _source(self) -> Source:
        return Source.OTHER

    class Config(_DefaultConfig):
        title = "Reshape Model"


"""
Framework-specific field below
"""

_optional_tf2_output_key_field = Field(
    None,
    title="Output key (optional for TF1 models, should not be set for TF2 models)",
    description="Output key that will cause the TF1 model to output an alternative "
    "representation, which might have more than 1 dimension",
)


class TFFullImageModelConfig(ImageFullModelConfig):
    tf_image_model_url: HttpUrl = Field(
        ...,
        title="TensorFlow URL",
        description="TensorFlow URL that points to an image model that is compatible "
        "with TF2 https://www.tensorflow.org/hub/model_compatibility (should be all)",
        example="https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4",
    )
    id: Optional[str] = Field(
        "0",
        title="id",
        description="To distinguish the same models in different versions",
        example="1",
    )
    output_key: Optional[str] = _optional_tf2_output_key_field
    required_image_size: ImageSize = _image_resize_field

    @property
    def _source(self) -> Source:
        return Source.TENSORFLOW

    class Config(_DefaultConfig):
        title = "TensorFlow Image Model"


class FinetunedTFFullImageModelConfig(ImageFullModelConfig):
    tf_image_model_url: str = Field(
        ...,
        title="Image Model Path",
        description="Path to Tensorflow Model",
        example="",
    )
    base_model: TFFullImageModelConfig = Field(
        ..., title="BaseModel used for training..."
    )
    train_readers: List = Field(..., title="Readers used for training")
    lr: float = Field(
        ...,
        title="Learning Rate",
    )
    epochs: int = Field(..., title="Epochs")
    output_key: Optional[str] = _optional_tf2_output_key_field
    required_image_size: ImageSize = _image_resize_field

    @property
    def _source(self) -> Source:
        return Source.TENSORFLOW

    class Config(_DefaultConfig):
        title = "TensorFlow Image Model"


class TorchvisionInternalModelConfig(BaseModel):
    id_name: str = Field(
        ...,
        title="Model name",
        description="Model identifier passed to torchvision",
        example="alexnet",
    )
    layer_extractor: Callable = Field(
        ...,
        title="Feature vector layer extractor",
        description="A function that given a model extracts a layer that corresponds to the feature vector",
        example=lambda x: x.classifier_1[5],
    )
    required_image_size: ImageSize = _image_resize_field

    class Config(_DefaultConfig):
        pass


class ImageNoOpModelConfig(ImageFullModelConfig):
    noop_target_image_size: ImageSize = _image_resize_field

    @property
    def _source(self) -> Source:
        return Source.OTHER

    class Config(_DefaultConfig):
        title = "Image Store (no-op) Model"


class ImageKerasLayerConfig(ImageFullModelConfig):
    image_layer_path: str = _layer_path_field
    required_image_size: ImageSize = _image_resize_field

    @property
    def _source(self) -> Source:
        return Source.KERAS_LAYER

    class Config(_DefaultConfig):
        title = "Image Keras Layer"


class HFImageModelConfig(ImageFullModelConfig):
    hf_name: str = Field(
        ...,
        title="Image Model Identifier",
        description="Huggingface Image Model Identifier",
        example="google/vit-base-patch16-224",
    )
    required_image_size: ImageSize = _image_resize_field

    @property
    def _source(self) -> Source:
        return Source.HUGGINGFACE_TRANSFORMERS

    class Config(_DefaultConfig):
        title = "Huggingface Image Model"
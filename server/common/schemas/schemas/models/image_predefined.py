from enum import Enum
from typing import Sequence, Tuple

from pydantic import Field
from pydantic.networks import HttpUrl
from pydantic.tools import parse_obj_as
from schemas._base import _DefaultConfig
from schemas.models.common import ProxyModelConfig, Source
from schemas.models.image_model import (
    ImageFullModelConfig,
    ImageModelInfo,
    ImageSize,
    TFFullImageModelConfig,
    TorchvisionInternalModelConfig,
)


class PredefinedTFImageModelName(str, Enum):
    INCEPTION_V3 = "Inception V3"
    RESNET_V2_50 = "ResNet V2 50"
    RESNET_V2_101 = "ResNet V2 101"
    RESNET_V2_152 = "ResNet V2 152"
    EFFICIENTNET_B0 = "EfficientNet B0"
    EFFICIENTNET_B1 = "EfficientNet B1"
    EFFICIENTNET_B2 = "EfficientNet B2"
    EFFICIENTNET_B3 = "EfficientNet B3"
    EFFICIENTNET_B4 = "EfficientNet B4"
    EFFICIENTNET_B5 = "EfficientNet B5"
    EFFICIENTNET_B6 = "EfficientNet B6"
    EFFICIENTNET_B7 = "EfficientNet B7"


class PredefinedTorchvisionModelName(str, Enum):
    ALEXNET = "AlexNet"
    VGG16 = "VGG16"
    VGG19 = "VGG19"
    GOOGLENET = "GoogLeNet"
    RESNET18 = "ResNet18"
    RESNET34 = "ResNet34"


predefinedTFImageModels = {
    PredefinedTFImageModelName.INCEPTION_V3: {
        "url": "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
        "image_size": ImageSize(height=299, width=299),
        "batch_size": 256,
    },
    PredefinedTFImageModelName.RESNET_V2_50: {
        "url": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4",
        "image_size": ImageSize(height=224, width=224),
        "batch_size": 512,
    },
    PredefinedTFImageModelName.RESNET_V2_101: {
        "url": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4",
        "image_size": ImageSize(height=224, width=224),
        "batch_size": 512,
    },
    PredefinedTFImageModelName.RESNET_V2_152: {
        "url": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4",
        "image_size": ImageSize(height=224, width=224),
        "batch_size": 512,
    },
    PredefinedTFImageModelName.EFFICIENTNET_B0: {
        "url": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
        "image_size": ImageSize(height=224, width=224),
        "batch_size": 128,
    },
    PredefinedTFImageModelName.EFFICIENTNET_B1: {
        "url": "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
        "image_size": ImageSize(height=240, width=240),
        "batch_size": 128,
    },
    PredefinedTFImageModelName.EFFICIENTNET_B2: {
        "url": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
        "image_size": ImageSize(height=260, width=260),
        "batch_size": 256,
    },
    PredefinedTFImageModelName.EFFICIENTNET_B3: {
        "url": "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
        "image_size": ImageSize(height=300, width=300),
        "batch_size": 128,
    },
    PredefinedTFImageModelName.EFFICIENTNET_B4: {
        "url": "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
        "image_size": ImageSize(height=380, width=380),
        "batch_size": 64,
    },
    PredefinedTFImageModelName.EFFICIENTNET_B5: {
        "url": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
        "image_size": ImageSize(height=456, width=456),
        "batch_size": 64,
    },
    PredefinedTFImageModelName.EFFICIENTNET_B6: {
        "url": "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
        "image_size": ImageSize(height=528, width=528),
        "batch_size": 8,
    },
    PredefinedTFImageModelName.EFFICIENTNET_B7: {
        "url": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
        "image_size": ImageSize(height=600, width=600),
        "batch_size": 16,
    },
}


class TFImageModelConfig(ProxyModelConfig):
    tf_image_name: PredefinedTFImageModelName = Field(
        ...,
        title="Model name",
        description="Name of the predefined TensorFlow image model",
        example=PredefinedTFImageModelName.EFFICIENTNET_B5,
    )

    @property
    def full_config(self) -> TFFullImageModelConfig:
        return _get_tf_full_image_model_config_and_info(self.tf_image_name)[0]

    class Config(_DefaultConfig):
        title = "Predefined TensorFlow Image Model"


def _get_tf_full_image_model_config_and_info(
    name: PredefinedTFImageModelName,
) -> Tuple[TFFullImageModelConfig, ImageModelInfo]:
    if name in predefinedTFImageModels:
        return (
            TFFullImageModelConfig(
                tf_image_model_url=parse_obj_as(
                    HttpUrl, predefinedTFImageModels[name]["url"]
                ),
                required_image_size=predefinedTFImageModels[name]["image_size"],
            ),
            ImageModelInfo(
                batch_size=predefinedTFImageModels[name]["batch_size"],
                image_size=predefinedTFImageModels[name]["image_size"].height,
            ),
        )
    else:
        raise NotImplementedError("Unknown Tensorflow Model name {}".format(name))


def get_all_tf_full_image_model_configs_with_info() -> Sequence[
    Tuple[TFFullImageModelConfig, ImageModelInfo]
]:
    """Generates a sequence of all predefined TensorFlow image models.

    Returns:
        Sequence[Tuple[TFFullImageModelConfig, ImageModelInfo]]: A sequence of all
        predefined TensorFlow image models together with information about them.
    """
    return [
        _get_tf_full_image_model_config_and_info(PredefinedTFImageModelName(name))
        for name in PredefinedTFImageModelName
    ]


def _get_torchvision_internal_model_config(
    name: PredefinedTorchvisionModelName,
) -> TorchvisionInternalModelConfig:
    if name == PredefinedTorchvisionModelName.ALEXNET:
        return TorchvisionInternalModelConfig(
            id_name="alexnet",
            layer_extractor=lambda x: x.classifier[5],
            required_image_size=ImageSize(height=224, width=224),
        )

    if name == PredefinedTorchvisionModelName.VGG16:
        return TorchvisionInternalModelConfig(
            id_name="vgg16",
            layer_extractor=lambda x: x.classifier[5],
            required_image_size=ImageSize(height=224, width=224),
        )

    if name == PredefinedTorchvisionModelName.VGG19:
        return TorchvisionInternalModelConfig(
            id_name="vgg19",
            layer_extractor=lambda x: x.classifier[5],
            required_image_size=ImageSize(height=224, width=224),
        )

    if name == PredefinedTorchvisionModelName.GOOGLENET:
        return TorchvisionInternalModelConfig(
            id_name="googlenet",
            layer_extractor=lambda x: x.dropout,
            required_image_size=ImageSize(height=224, width=224),
        )
    if name == PredefinedTorchvisionModelName.RESNET18:
        return TorchvisionInternalModelConfig(
            id_name="resnet18",
            layer_extractor=lambda x: x.avgpool,
            required_image_size=ImageSize(height=224, width=224),
        )
    if name == PredefinedTorchvisionModelName.RESNET34:
        return TorchvisionInternalModelConfig(
            id_name="resnet34",
            layer_extractor=lambda x: x.avgpool,
            required_image_size=ImageSize(height=224, width=224),
        )
    raise NotImplementedError(f"Unknown torchvision model name: {name}")


class TorchvisionFullModelConfig(ImageFullModelConfig):
    torchvision_name: PredefinedTorchvisionModelName = Field(
        ...,
        title="Model name",
        description="Name of the predefined torchvision model",
        example=PredefinedTorchvisionModelName.ALEXNET,
    )

    @property
    def internal_config(self) -> TorchvisionInternalModelConfig:
        return _get_torchvision_internal_model_config(self.torchvision_name)

    @property
    def _source(self) -> Source:
        return Source.TORCHVISION

    class Config(_DefaultConfig):
        title = "Predefined torchvision Model"


def _get_torchvision_full_model_config(
    name: PredefinedTorchvisionModelName,
) -> Tuple[TorchvisionFullModelConfig, ImageModelInfo]:
    # Can be this simple (for now), because image size for all models is 224

    model = TorchvisionFullModelConfig(torchvision_name=name)

    if name == PredefinedTorchvisionModelName.ALEXNET:
        return model, ImageModelInfo(batch_size=2048, image_size=224)

    if name == PredefinedTorchvisionModelName.VGG16:
        return model, ImageModelInfo(batch_size=128, image_size=224)

    if name == PredefinedTorchvisionModelName.VGG19:
        return model, ImageModelInfo(batch_size=128, image_size=224)

    if name == PredefinedTorchvisionModelName.GOOGLENET:
        return model, ImageModelInfo(batch_size=512, image_size=224)

    if name == PredefinedTorchvisionModelName.RESNET18:
        return model, ImageModelInfo(batch_size=64, image_size=224)
    if name == PredefinedTorchvisionModelName.RESNET34:
        return model, ImageModelInfo(batch_size=64, image_size=224)
    raise NotImplementedError(f"Unknown torchvision model name: {name}")


class TorchvisionFullModelConfig(ImageFullModelConfig):
    torchvision_name: PredefinedTorchvisionModelName = Field(
        ...,
        title="Model name",
        description="Name of the predefined torchvision model",
        example=PredefinedTorchvisionModelName.ALEXNET,
    )

    @property
    def internal_config(self) -> TorchvisionInternalModelConfig:
        return _get_torchvision_internal_model_config(self.torchvision_name)

    @property
    def _source(self) -> Source:
        return Source.TORCHVISION

    class Config(_DefaultConfig):
        title = "Predefined torchvision Model"


def get_all_torchvision_model_configs() -> Sequence[
    Tuple[TorchvisionFullModelConfig, ImageModelInfo]
]:
    """Generates a sequence of all predefined torchvision models.

    Returns:
        Sequence[Tuple[TorchvisionFullModelConfig, ImageModelInfo]]: A sequence of all
        predefined torchvision models together with info about them.
    """
    return [
        _get_torchvision_full_model_config(PredefinedTorchvisionModelName(name))
        for name in PredefinedTorchvisionModelName
    ]

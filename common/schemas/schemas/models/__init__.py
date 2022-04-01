# here we defined the exposed configs for REST api response
from typing import Sequence, Tuple, Union

from schemas.models.common import FullModelConfig, ProxyModelConfig
from schemas.models.image_model import (
    FinetunedTFFullImageModelConfig,
    ImageFullModelConfig,
    ImageKerasLayerConfig,
    ImageModelInfo,
    ImageNoOpModelConfig,
    ReshapeModelConfig,
    TF1FullImageModelConfig,
    TFFullImageModelConfig,
)
from schemas.models.image_predefined import (
    PredefinedTorchvisionModelName,
    TFImageModelConfig,
    TorchvisionFullModelConfig,
    get_all_tf_full_image_model_configs_with_info,
    get_all_torchvision_model_configs,
)
from schemas.models.text_model import (
    HFModelConfig,
    TextFullModelConfig,
    TextKerasLayerConfig,
    TextModelInfo,
    TextNoOpModelConfig,
    TF1FullTextModelConfig,
    TFFullTextModelConfig,
)
from schemas.models.text_predefined import (
    TFTextModelConfig,
    get_all_tf_full_text_model_configs_with_info,
)

ImageFullModelConfigsU = Union[
    ImageKerasLayerConfig,
    ImageNoOpModelConfig,
    ReshapeModelConfig,
    TF1FullImageModelConfig,
    TFFullImageModelConfig,
    TorchvisionFullModelConfig,
    FinetunedTFFullImageModelConfig,
]

TextFullModelConfigsU = Union[
    HFModelConfig,
    TextKerasLayerConfig,
    TextNoOpModelConfig,
    TF1FullTextModelConfig,
    TFFullTextModelConfig,
]

ImageModelConfigsU = Union[ImageFullModelConfigsU, TFImageModelConfig]
TextModelConfigsU = Union[TextFullModelConfigsU, TFTextModelConfig]

# Contains:
# 1. ImageFullModelConfig
# 2. TextFullModelConfig
# 3. ProxyModelConfig
# Does not contain:
# 1. TorchvisionInternalModelConfig -> Unsafe for users to pass arbitrary functions
AllModelConfigsU = Union[ImageModelConfigsU, TextModelConfigsU]


def _convert_proxy_model_config(config: AllModelConfigsU) -> FullModelConfig:
    """Performs type conversion from a Union type to a regular type by converting
    'shortcut' configs to 'full' configs.

    Args:
        config (AllModelConfigsU): Union type model config.

    Returns:
        FullModelConfig: Regular type model config.
    """
    full_config: FullModelConfig
    if isinstance(config, ProxyModelConfig):
        return config.full_config
    else:
        return config


def _get_image_full_model_config(config: ImageModelConfigsU) -> ImageFullModelConfig:
    full_config = _convert_proxy_model_config(config)
    if isinstance(full_config, ImageFullModelConfig):
        return full_config
    raise RuntimeError(f"Unable to get an image full model config from {config!r}")


def _get_text_full_model_config(config: TextModelConfigsU) -> TextFullModelConfig:
    """Performs type conversion from Union type (text models) to a regular 'full' text
    model type.

    Args:
        config (TextModelConfigsU): Union type text model config.

    Returns:
        TextFullModelConfig: Regular 'full' type text model config.
    """
    full_config = _convert_proxy_model_config(config)
    if isinstance(full_config, TextFullModelConfig):
        return full_config
    raise RuntimeError(f"Unable to get a text full model config from {config!r}")


def get_full_model_config(
    config: AllModelConfigsU,
) -> Union[ImageFullModelConfig, TextFullModelConfig]:
    """Performs type conversion from Union type (all models) to a union of regular 'full'
    image model and text model types.

    Args:
        config (AllModelConfigsU): Union type model config.

    Returns:
        Union[ImageFullModelConfig, TextFullModelConfig]: Regular 'full' type model
        config.
    """
    full_config = _convert_proxy_model_config(config)
    if isinstance(full_config, ImageFullModelConfig) or isinstance(
        full_config, TextFullModelConfig
    ):
        return full_config
    raise RuntimeError(f"Unable to get a full model config from {config!r}")


_alexnet_example = TorchvisionFullModelConfig(
    torchvision_name=PredefinedTorchvisionModelName.ALEXNET
)


def get_predefined_image_model_configs_with_info() -> Sequence[
    Tuple[ImageFullModelConfig, ImageModelInfo]
]:
    """Generates a sequence of all predefined image models.

    Returns:
        Sequence[Tuple[ImageFullModelConfig, ImageModelInfo]]: A sequence of predefined
        image models together with info about them.
    """
    return [
        *get_all_tf_full_image_model_configs_with_info(),
        *get_all_torchvision_model_configs(),
    ]


def get_predefined_text_model_configs_with_info() -> Sequence[
    Tuple[TextFullModelConfig, TextModelInfo]
]:
    """Generates a sequence of all predefined text models.

    Returns:
        Sequence[Tuple[TextFullModelConfig, TextModelInfo]]: A sequence of predefined
        text models together with info about them.
    """
    return get_all_tf_full_text_model_configs_with_info()

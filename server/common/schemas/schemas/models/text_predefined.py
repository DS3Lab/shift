from enum import Enum
from typing import Sequence, Tuple

from pydantic import Field
from pydantic.networks import HttpUrl
from pydantic.tools import parse_obj_as
from schemas._base import _DefaultConfig
from schemas.models.common import ProxyModelConfig
from schemas.models.text_model import TextModelInfo, TFFullTextModelConfig


class PredefinedTFTextModelName(str, Enum):
    ELMO = "ELMo"
    NNLM_50 = "NNLM 50"
    NNLM_50_NORMALIZATION = "NNLM 50 Normalization"
    NNLM_128 = "NNLM 128"
    NNLM_128_NORMALIZATION = "NNLM 128 Normalization"
    USE = "Universal Sentence Encoder"
    USE_LARGE = "Universal Sentence Encoder Large"


predefinedTextModels = {
    PredefinedTFTextModelName.ELMO: {
        "url": "https://tfhub.dev/google/elmo/3",
        "batch_size": 2,
    },
    PredefinedTFTextModelName.NNLM_50: {
        "url": "https://tfhub.dev/google/nnlm-en-dim50/2",
        "batch_size": 2048,
    },
    PredefinedTFTextModelName.NNLM_50_NORMALIZATION: {
        "url": "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2",
        "batch_size": 2048,
    },
    PredefinedTFTextModelName.NNLM_128: {
        "url": "https://tfhub.dev/google/nnlm-en-dim128/2",
        "batch_size": 2048,
    },
    PredefinedTFTextModelName.NNLM_128_NORMALIZATION: {
        "url": "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2",
        "batch_size": 2048,
    },
    PredefinedTFTextModelName.USE: {
        "url": "https://tfhub.dev/google/universal-sentence-encoder/4",
        "batch_size": 2048,
    },
    PredefinedTFTextModelName.USE_LARGE: {
        "url": "https://tfhub.dev/google/universal-sentence-encoder-large/5",
        "batch_size": 16,
    },
}


def _get_tf_full_text_model_config_and_info(
    name: PredefinedTFTextModelName,
) -> Tuple[TFFullTextModelConfig, TextModelInfo]:
    if name in PredefinedTFTextModelName:
        return (
            TFFullTextModelConfig(
                tf_text_model_url=parse_obj_as(
                    HttpUrl, predefinedTextModels[name]["url"]
                ),
            ),
            TextModelInfo(batch_size=predefinedTextModels[name]["batch_size"]),
        )
    else:
        raise NotImplementedError(f"Unknown TensorFlow model name: {name}")


def get_all_tf_full_text_model_configs_with_info() -> Sequence[
    Tuple[TFFullTextModelConfig, TextModelInfo]
]:
    """Generates a sequence of all predefined TensorFlow text models.

    Returns:
        Sequence[Tuple[TFFullTextModelConfig, TextModelInfo]]: A sequence of all
        predefined TensorFlow text models together with info about them.
    """
    return [
        _get_tf_full_text_model_config_and_info(PredefinedTFTextModelName(name))
        for name in PredefinedTFTextModelName
    ]


class TFTextModelConfig(ProxyModelConfig):
    tf_text_name: PredefinedTFTextModelName = Field(
        ...,
        title="Model name",
        description="Name of the predefined TensorFlow text model",
        example=PredefinedTFTextModelName.USE_LARGE,
    )

    @property
    def full_config(self) -> TFFullTextModelConfig:
        return _get_tf_full_text_model_config_and_info(self.tf_text_name)[0]

    class Config(_DefaultConfig):
        title = "Predefined TensorFlow Text Model"

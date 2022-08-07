from abc import ABC
from typing import Optional

from pydantic import root_validator
from pydantic.fields import Field
from pydantic.networks import HttpUrl
from schemas._base import _DefaultConfig
from schemas.models.common import (
    FullModelConfig,
    ModelInfo,
    Source,
    _layer_path_field,
    _optional_tf2_output_key_field,
)
from typing_extensions import Literal


class TextFullModelConfig(FullModelConfig, ABC):
    @property
    def invariant_json(self) -> str:
        return self.json(exclude={"info"})


class TextModelInfo(ModelInfo):
    token_length: Optional[int] = Field(
        None,
        title="Token Length",
        description="Number of tokens used in the inference",
        example=500,
        ge=1,
        le=10 ** 15,
    )

    class Config(_DefaultConfig):
        title = "Information about the text model"


"""
TensorFlow Related Model Config/Info
"""


class TFFullTextModelConfig(TextFullModelConfig):
    tf_text_model_url: HttpUrl = Field(
        ...,
        title="TensorFlow URL",
        description="TensorFlow URL that points to a text model that is compatible "
        "with TF2 https://www.tensorflow.org/hub/model_compatibility (should be all)",
        example="https://tfhub.dev/google/universal-sentence-encoder/4",
    )
    output_key: Optional[str] = _optional_tf2_output_key_field

    @property
    def _source(self) -> Source:
        return Source.TENSORFLOW

    class Config(_DefaultConfig):
        title = "TensorFlow Text Model"


class TextNoOpModelConfig(TextFullModelConfig):
    noop_text: str = Field(
        ...,
        title="Model selector",
        description="Can be set to any value - is used just to distinguish the model "
        "from others",
        example="yes",
    )

    @property
    def _source(self) -> Source:
        return Source.OTHER

    @property
    def invariant_json(self) -> str:
        copy = self.copy(deep=True)
        copy.noop_text = ""
        return copy.json()

    class Config(_DefaultConfig):
        title = "Text Store (no-op) Model"


class TextKerasLayerConfig(TextFullModelConfig):
    text_layer_path: str = _layer_path_field

    @property
    def _source(self) -> Source:
        return Source.KERAS_LAYER

    class Config(_DefaultConfig):
        title = "Text Keras Layer"

class HFTextModelConfig(TextFullModelConfig):
    hf_name: str = Field(
        ...,
        title="HuggingFace Model Identifier",
        description="Model Identifier that is passed directly to HuggingFace",
        example="bert-base-cased",
    )
    max_length: int = Field(
        ...,
        title="Max token length",
        description="Maximal number of tokens used when calculating the embedding",
        example=512,
        ge=1,
    )
    pooled_output: bool = Field(
        ...,
        title="Use pooled output",
        description="True if pooled output should be used, False if not "
        "(pooled output is not available for all models)",
        example=False,
    )
    tokenizer_params: Optional[dict] = Field(
        None,
        title="Tokenizer parameters",
        description="Additional parameters passed to the tokenizer that corresponds "
        "to the model",
    )

    @property
    def _source(self) -> Source:
        return Source.HUGGINGFACE_TRANSFORMERS

    @root_validator
    def pooled_output_available(cls, values: dict) -> dict:
        if "hf_name" in values and "pooled_output" in values:
            hf_name: str = values["hf_name"]
            pooled_output: bool = values["pooled_output"]

            if pooled_output and "bert" not in hf_name:
                raise ValueError(
                    f"Model {values['hf_name']} does not support pooled output"
                )

        return values

    class Config(_DefaultConfig):
        title = "HuggingFace Model"

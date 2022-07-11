from abc import ABC, abstractmethod
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TargetEnvironment(str, Enum):
    GENERAL = "general"
    TF_1 = "tf_1"


class Source(str, Enum):
    TENSORFLOW = "TensorFlow"
    TORCHVISION = "torchvision"
    HUGGINGFACE_TRANSFORMERS = "HuggingFace Transformers"
    KERAS_LAYER = "Keras layer"
    VTAB_MODEL = "vtab"
    OTHER = "other"


class FullModelConfig(BaseModel, ABC):
    @property
    @abstractmethod
    def invariant_json(self) -> str:
        """Gets an invariant JSON from which a model config can be constructed.
        The invariance here means that two models that produce the same output for a given input should have a same JSON.

        Returns:
            str: Invariant JSON representation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _source(self) -> Source:
        raise NotImplementedError

    @property
    def source_str(self) -> str:
        return self._source.value

    @property
    def target_environment(self) -> TargetEnvironment:
        return TargetEnvironment.GENERAL


class ModelInfo(BaseModel):
    batch_size: Optional[int] = Field(
        None, title="Batch size", description="Batch size when running inference", ge=1
    )
    date_added: Optional[date] = Field(
        date.today(),
        title="Date added",
        description="Date when model was registered in sh¡ft!",
    )
    num_params: Optional[int] = Field(
        None,
        title="Number of parameters",
        description="Number of parameters present in the model",
        example=10_000_000,
        ge=1,
        le=10 ** 15,
    )


class ImageFullModelConfig(FullModelConfig, ABC):
    @property
    def invariant_json(self) -> str:
        return self.json(exclude={"info"})


class ProxyModelConfig(BaseModel, ABC):
    """Classes that do not directly define the model, but instead provide a shortcut
    to the model for commonly used models.
    """

    @property
    @abstractmethod
    def full_config(self) -> FullModelConfig:
        raise NotImplementedError


_optional_tf2_output_key_field = Field(
    None,
    title="Output key (optional for TF1 models, should not be set for TF2 models)",
    description="Output key that will cause the TF1 model to output an alternative "
    "representation, which might have more than 1 dimension",
)

_optional_tf1_output_key_field = Field(
    None,
    title="Output key",
    description="Output key that will cause the model to output an alternative "
    "representation, which might have more than 1 dimension",
)

_layer_path_field = Field(
    ...,
    title="Path to the stored Keras layer",
    description="Path to the stored Keras layer relative to the specified mount "
    "location",
    example="Documents/keras_layers/my_model",
)

from datetime import date
from typing import List, Optional, Union

from pydantic import Field
from schemas._base import _DefaultConfig
from schemas.common import DateRange, IntegerRange
from schemas.models import (
    AllModelConfigsU,
    ImageFullModelConfig,
    ImageModelConfigsU,
    ImageModelInfo,
    TextFullModelConfig,
    TextModelConfigsU,
    TextModelInfo,
    TFTextModelConfig,
    _alexnet_example,
    _get_image_full_model_config,
    _get_text_full_model_config,
    get_full_model_config,
)
from schemas.models.common import BaseModel, Source
from schemas.models.text_predefined import PredefinedTFTextModelName
from schemas.requests.reader import AllReaderConfigsU, TFReaderConfig


class ModelInfoRequest(BaseModel):
    date_added: Optional[DateRange] = Field(
        None,
        title="Date added range",
        description="Models that were added on a date within the range",
    )
    num_params: Optional[IntegerRange] = Field(
        None,
        title="Number of parameters range",
        description="Models that have a number of parameters within the range",
    )
    source: Optional[Source] = Field(
        None,
        title="Source",
        description="What is the origin of the model",
        example=Source.TENSORFLOW,
    )
    dimension: Optional[IntegerRange] = Field(
        None,
        title="Output Dimension",
        description="Models that have a number of output dimension within the range",
    )
    finetuned: Optional[bool] = Field(
        True,
        title="Finetuned",
        description="Include finedtuned values or not",
    )


class ReadersUsedWithAModelRequest(BaseModel):
    model: AllModelConfigsU = Field(
        ...,
        title="Model",
        description="Model that was already used with some readers",
        example=_alexnet_example,
    )

    @property
    def full_config(self) -> Union[ImageFullModelConfig, TextFullModelConfig]:
        return get_full_model_config(self.model)

    class Config(_DefaultConfig):
        title = "Model specification"


class ImageModelRegistrationRequest(BaseModel):
    model: ImageModelConfigsU = Field(
        ...,
        title="Image model",
        description="Image model that will be registered, "
        "so that it can be queried later",
        example=_alexnet_example,
    )
    info: ImageModelInfo = Field(
        ...,
        title="Image model info",
        description="Information about the image model",
        example=ImageModelInfo(
            batch_size=128,
            date_added=date(2021, 1, 1),
            num_params=10 ** 8,
            image_size=224,
        ),
    )
    finetuned: Optional[bool] = Field(
        False, title="Finetuned", description="Set to true if it is a finetuned model"
    )

    @property
    def full_config(self) -> ImageFullModelConfig:
        return _get_image_full_model_config(self.model)


class TextModelRegistrationRequest(BaseModel):
    model: TextModelConfigsU = Field(
        ...,
        title="Text model",
        description="Text model that will be registered, "
        "so that it can be queried later",
        example=TFTextModelConfig(tf_text_name=PredefinedTFTextModelName.ELMO),
    )
    info: TextModelInfo = Field(
        ...,
        title="Text model info",
        description="Information about the text model",
        example=TextModelInfo(
            batch_size=128,
            date_added=date(2021, 1, 1),
            num_params=10 ** 8,
            token_length=512,
        ),
    )

    @property
    def full_config(self) -> TextFullModelConfig:
        return _get_text_full_model_config(self.model)


class ImageModelInfoRequest(ModelInfoRequest):
    image_size: Optional[IntegerRange] = Field(
        None,
        title="Image size range",
        description="Models that resize images to a size within the range",
    )

    class Config(_DefaultConfig):
        title = "Image models that satisfy the criteria"


class TextModelInfoRequest(ModelInfoRequest):
    token_length: Optional[IntegerRange] = Field(
        None,
        title="Token length range",
        description="Models that use a number of tokens within the range",
    )

    class Config(_DefaultConfig):
        title = "Text models that satisfy the criteria"


class TaskAgnosticRequest(BaseModel):
    json_reader: AllReaderConfigsU = Field(
        ...,
        title="Reader",
        description="Reader used for embedding",
        example=TFReaderConfig(
            split="train",
            tf_dataset_name="cifar10:3.0.2",
            embed_feature_path=["image"],
            label_feature_path=["label"],
        ),
    )


class QueryModelByTagsRequest(BaseModel):
    tags: List[str] = Field(..., title="Tags", description="Tags used for searching")
    source: str = Field(..., title="ImageModel or TextModel")

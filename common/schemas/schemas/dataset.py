from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field
from schemas.requests.reader import AllReaderConfigsU

from ._base import _DefaultConfig


class DatasetInfo(BaseModel):
    trainsetLength: int = Field(
        ...,
        title="Trainset Length",
        description="The length (number of data points) in the training dataset",
        example=10000,
        ge=1,
    )
    testsetLength: int = Field(
        ...,
        title="Testset Length",
        description="The length (number of data points) in the test dataset",
        example=1000,
        ge=1,
    )

    class Config(_DefaultConfig):
        pass


class FullDatasetConfig(BaseModel, ABC):
    @property
    def invariant_json(self) -> str:
        return self.json(exclude={"info"})


class DatasetInfo(BaseModel):
    size: Optional[int] = Field(
        None, title="Size", description="Size of training set", ge=1
    )
    date_added: Optional[date] = Field(
        date.today(),
        title="Date added",
        description="Date when the dataset was registered in sh¡ft!",
    )
    name: Optional[str] = Field(None, title="Name", description="A name to the dataset")
    path: Optional[str] = Field(
        None, title="Path", description="Path to the folder of the dataset"
    )


class DatasetRegistrationRequest(BaseModel):
    # TODO: here the dict should be a union of the readers that we support
    # For simplicity, we now use a generic dict to support all.
    dataset: AllReaderConfigsU = Field(
        ...,
        title="Dataset",
        description="The dataset to be registered",
    )
    info: Optional[DatasetInfo] = Field(
        ...,
        title="Dataset Info",
        description="Information about the dataset",
        example=DatasetInfo(size=1, date_added=date(2021, 1, 1)),
    )

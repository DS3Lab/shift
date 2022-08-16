from enum import Enum

from pydantic.fields import Field
from pydantic.main import BaseModel
from schemas._base import _DefaultConfig


class GradientDescentSettings(BaseModel):
    learning_rate: float = 0.1
    num_epochs: int = 100

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    class Config(_DefaultConfig):
        title = "GradientDescentSettings"


class Classifier(str, Enum):
    EUCLIDEAN_NN = "Euclidean NN"
    COSINE_NN = "Cosine NN"
    LINEAR = "Linear"


class ClassifierWithParams(BaseModel):
    name: Classifier = Field(
        ...,
        title="Classifier",
        description="Classifiers to use on top of the embedded data",
    )
    parameters: GradientDescentSettings = Field(
        GradientDescentSettings(),
        title="Parameters",
        description="The parameters to use with the classifier",
    )

    @property
    def value(self):
        if self.name in [Classifier.EUCLIDEAN_NN, Classifier.COSINE_NN]:
            return self.name
        else:
            return "{}({})".format(self.name, self.parameters)

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    class Config(_DefaultConfig):
        title = "ClassifierWithParams"

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, root_validator
from schemas.models import (
    AllModelConfigsU,
    ImageFullModelConfigsU,
    TextFullModelConfigsU,
)
from schemas.requests.common import ClassifierRequest, InferenceRequest
from schemas.requests.reader import AllReaderConfigsU
from schemas.task.result import KnownResult
from dstool.database import add_event
from ._base import ID, Hash, Status, _DefaultConfig


# TODO: NN - also include distances
class NearestNeighborResult(BaseModel):
    """Contains data needed to calculate the nearest neighbor error and to easily recalculate the error if only labels are changed."""

    test_labels: List[int] = Field(
        ...,
        title="Test labels",
        description="Label for each test point",
        min_items=1,
    )
    test_indices_within_readers: List[int] = Field(
        ...,
        title="Test indices within readers",
        description="For each test point its index within the reader to which it belongs",
        min_items=1,
    )
    test_reader_indices: List[int] = Field(
        ...,
        title="Test reader indices",
        description="For each test point index of the test reader to which it belongs",
        min_items=1,
    )
    train_labels: List[int] = Field(
        ...,
        title="Train labels",
        description="Label of the closest training point for each test point",
        min_items=1,
    )
    train_indices_within_readers: List[int] = Field(
        ...,
        title="Train indices within readers",
        description="For each test point, index within the reader to which belongs its "
        "closest training point",
        min_items=1,
    )
    train_reader_indices: List[int] = Field(
        ...,
        title="Train reader indices",
        description="For each test point index of the train reader to which belongs "
        "its closest training point",
        min_items=1,
    )

    @property
    def error(self) -> float:
        """Returns 1-NN error corresponding to the labels.

        Returns:
            float: 1-NN error.
        """
        length = len(self.train_labels)

        return self.raw_error / float(length)

    @property
    def raw_error(self) -> int:
        """Returns the number of errors in the 1-NN result.

        Returns:
            int: Number of errors.
        """
        ts = add_event(payload={
            'classifier': 'NN',
        }, tags=['shift', 'calc_error', 'start'])
        cnt_error = 0
        length = len(self.train_labels)
        for i in range(length):
            if self.train_labels[i] != self.test_labels[i]:
                cnt_error += 1
        ts = add_event(payload={
            'classifier': 'NN',
        }, tags=['shift', 'calc_error', 'end'], previous_timestamp=ts)
        return cnt_error

    @root_validator
    def compatible_lengths(cls, values: dict) -> dict:
        """Checks that all indices fields have same length."""
        fields = [
            "test_labels",
            "test_indices_within_readers",
            "test_reader_indices",
            "train_labels",
            "train_indices_within_readers",
            "train_reader_indices",
        ]

        all_fields_present = True
        for f in fields:
            if f not in values:
                all_fields_present = False
                break

        if all_fields_present:
            list_lengths = [len(values[f]) for f in fields]
            if len(set(list_lengths)) != 1:
                raise ValueError("All lists must have same length")

        return values

    class Config(_DefaultConfig):
        title = "Nearest neighbor result"


class LinearResult(BaseModel):
    """Contains data needed to calculate the nearest neighbor error and to easily recalculate the error if only labels are changed."""

    test_labels: List[int] = Field(
        ...,
        title="Test labels",
        description="Label for each test point",
        min_items=1,
    )
    predicted_test_labels: List[int] = Field(
        ...,
        title="Predicted labels",
        description="Predicted label for each test point",
        min_items=1,
    )

    @property
    def error(self) -> float:
        """Returns error corresponding to the labels.

        Returns:
            float: 1-NN error.
        """
        length = len(self.test_labels)
        return self.raw_error / float(length)

    @property
    def raw_error(self) -> int:
        """Returns the number of errors in the linear result.

        Returns:
            int: Number of errors.
        """
        cnt_error = 0
        length = len(self.test_labels)
        for i in range(length):
            if self.test_labels[i] != self.predicted_test_labels[i]:
                cnt_error += 1

        return cnt_error

    @root_validator
    def compatible_lengths(cls, values: dict) -> dict:
        """Checks that all indices fields have same length."""
        fields = [
            "test_labels",
            "predicted_test_labels",
        ]

        all_fields_present = True
        for f in fields:
            if f not in values:
                all_fields_present = False
                break

        if all_fields_present:
            list_lengths = [len(values[f]) for f in fields]
            if len(set(list_lengths)) != 1:
                raise ValueError("All lists must have same length")

        return values

    class Config(_DefaultConfig):
        title = "Linear result"


class StatusResponse(BaseModel):
    """Reports status of some job."""

    status: Status = Field(
        ..., title="Status", description="Status of the job", example=Status.FINISHED
    )
    additional: Optional[Any] = Field(None, description="Additional information")

    @property
    def successful(self):
        return self.status == Status.FINISHED

    @property
    def failed(self):
        return self.status == Status.FAILED

    class Config(_DefaultConfig):
        title = "Status response"


class Error404(BaseModel):
    status: str = Field(
        "ID does not exist", description="Information why 404 is returned"
    )

    class Config(_DefaultConfig):
        title = "Error 404"


class JobSubmittedResponse(BaseModel):
    """Provides information about the submitted job."""

    description: str = Field(
        ..., title="Description", description="Description of the submitted job"
    )
    inference_request_paths: Sequence[str] = Field(
        ...,
        title="Inference request paths",
        description="Paths that can be queried to obtain statuses of inference "
        "jobs/requests",
    )
    classifier_request_paths: Sequence[str] = Field(
        ...,
        title="Classifier request paths",
        description="Paths that can be queried to obtain statuses of classifier "
        "jobs/requests",
    )
    inference_requests: Sequence[InferenceRequest] = Field(
        ...,
        title="Inference requests",
        description="Mapping from IDs to actual inference requests",
    )
    classifier_requests: Sequence[ClassifierRequest] = Field(
        ...,
        title="Classifier requests",
        description="Mapping from IDs to actual classifier requests",
    )

    class Config(_DefaultConfig):
        title = "Job submitted response"


class BatchSizeError409(BaseModel):
    status: str = Field(..., description="Information which batch size is missing")

    class Config(_DefaultConfig):
        title = "Batch size error"


class MatchingImageModelsResponse(BaseModel):
    """Contains image models matching the criteria."""

    models: Sequence[ImageFullModelConfigsU] = Field(
        ...,
        title="Matching image models",
        description="Image models matching the criteria",
    )
    info: Dict = Field(
        ..., title="Information", description="Information about the models"
    )

    @classmethod
    def from_model_json_sequence(
        cls, models: Sequence[Dict]
    ) -> "MatchingImageModelsResponse":
        return cls.parse_obj(
            {
                "models": [json.loads(m["str"]) for m in models],
                "info": {
                    "num_params": [m["num_params"] for m in models],
                    "up_acc": [m["up_acc"] for m in models],
                },
            }
        )


class MatchingTextModelsResponse(BaseModel):
    """Contains text models matching the criteria."""

    models: Sequence[TextFullModelConfigsU] = Field(
        ...,
        title="Matching text models",
        description="Text models matching the criteria",
    )

    @classmethod
    def from_model_json_sequence(
        cls, models_str: Sequence[str]
    ) -> "MatchingTextModelsResponse":
        return cls.parse_obj({"models": [json.loads(m) for m in models_str]})


class ReaderInfo(BaseModel):
    """Information about the reader"""

    job_hash: Hash = Field(
        ...,
        title="Job hash",
        description="Hash of the job that used the requested model and this reader",
    )
    reader: AllReaderConfigsU = Field(
        ...,
        title="Reader",
        description="Reader that was used together with the requested model",
    )

    class Config(_DefaultConfig):
        title = "Reader information"


class ReadersUsedWithAModelResponse(BaseModel):
    """Contains readers that were used with the requested model."""

    readers: Sequence[ReaderInfo] = Field(
        ...,
        title="Readers",
        description="Readers that were used together with the requested model",
    )

    @classmethod
    def from_tuples(cls, tuples: Sequence[Tuple[ID, str]]):
        sequence = [
            {
                "job_hash": tuple_[0],
                "reader": json.loads(tuple_[1]),
            }
            for tuple_ in tuples
        ]
        return cls.parse_obj({"readers": sequence})

    class Config(_DefaultConfig):
        title = "Readers used with a model"


class ModelInfo(BaseModel):
    """Information about the model"""

    job_hash: Hash = Field(
        ...,
        title="Job hash",
        description="Hash of the job that used the requested reader and this model",
    )
    model: AllModelConfigsU = Field(
        ...,
        title="Model",
        description="Model that was used together with the requested reader",
    )

    class Config(_DefaultConfig):
        title = "Model information"


class ModelsUsedWithAReaderResponse(BaseModel):
    """Contains models that were used with the requested reader."""

    models: Sequence[ModelInfo] = Field(
        ...,
        title="Models",
        description="Models that were used together with the requested reader",
    )

    @classmethod
    def from_tuples(cls, tuples: Sequence[Tuple[ID, str]]):
        sequence = [
            {
                "job_hash": tuple_[0],
                "model": json.loads(tuple_[1]),
            }
            for tuple_ in tuples
        ]
        return cls.parse_obj({"models": sequence})

    class Config(_DefaultConfig):
        title = "Models used with a reader"


class QueryResultResponse(BaseModel):
    """
    Contains models from known results and the number of tasks that are being executed and will return later on.
    """

    known_results: List[KnownResult] = Field(
        ...,
        title="Known Results",
        description="This field represents the known results",
    )
    num_remaining_tasks: int = Field(
        ...,
        title="Number of Remaining Tasks",
        description="The number of remaining tasks",
    )
    # TODO: add unknwon results


class Task2VecResponse(BaseModel):
    """
    Contains the task2vec model
    """

    distances: List[List[float]] = Field(
        ...,
        title="Distance Matrix",
        description="Distance matrix of the task2vec results",
    )
    num_remaining_tasks: int = Field(
        ...,
        title="Number of Remaining Tasks",
        description="The number of remaining task2vec tasks",
    )


class TaskAgnosticResponse(BaseModel):
    """
    Contains the task agnostic results
    """

    results: List = Field(
        ..., title="Task Agnostic", description="Task Agnostic Results"
    )

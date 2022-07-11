from typing import Optional, Tuple

from pydantic import BaseModel, Field


class KnownResult(BaseModel):
    json_model: dict = Field(
        ...,
        title="Model JSON",
        description="The representation of the model for this result",
    )
    classifier: str = Field(..., title="Classifier", description="The classifier used")
    err: float = Field(
        ...,
        title="Error",
        description="The error with the model and the classifier on the model",
    )
    classify_job_hash: str = Field(..., title="Classification Job Hash", description="")
    test_reader_name: Optional[str] = Field(
        None, title="Test Reader", description="The name of the testing reader"
    )
    start: Optional[int] = Field(
        None, title="Start", description="The start of the train_reader"
    )
    stop: Optional[int] = Field(
        None, title="Stop", description="The stop of the train_reader"
    )

from typing import List

from pydantic import BaseModel, Field
from schemas._base import generate_id
from schemas.models.image_predefined import TorchvisionFullModelConfig
from schemas.requests.common import Task2VecRequest
from schemas.requests.reader import AllReaderConfigsU, TFReaderConfig


class Task2VecMultipleReaderRequest(BaseModel):
    """This type of requests are used as endpoint, for calculating the embeddings with a given list of readers"""

    probe: TorchvisionFullModelConfig = Field(
        ...,
        title="Probe",
        description="The model used for the embedding calculation, must be ResNet18 or ResNet34",
        example=TorchvisionFullModelConfig(torchvision_name="ResNet18"),
    )
    readers: List[AllReaderConfigsU] = Field(
        ...,
        title="Readers",
        description="The readers used for the embedding calculation",
        example=[
            TFReaderConfig(
                split="train",
                tf_dataset_name="cifar10:3.0.2",
                embed_feature_path=["image"],
                label_feature_path=["label"],
            )
        ],
    )

    def generate_task2vec_requests(self) -> List[Task2VecRequest]:
        """Generates a list of Task2VecRequests, one for each reader"""
        return [
            Task2VecRequest(
                id=generate_id(),
                probe=self.probe,
                reader=reader,
            )
            for reader in self.readers
        ]

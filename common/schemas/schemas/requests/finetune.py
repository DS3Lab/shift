from typing import List, Optional, Union

from pydantic import BaseModel, Field
from schemas._base import ID, Hash, _id_regex, get_hash
from schemas.models import AllModelConfigsU, get_full_model_config
from schemas.models.common import ImageFullModelConfig
from schemas.models.text_model import TextFullModelConfig
from schemas.requests.reader import AllReaderConfigsU, ReaderConfig, get_reader_config


class FinetuneRequest(BaseModel):
    id: Optional[ID] = Field(
        None,
        title="Classifier request ID",
        description="Classifier request ID that is returned to the user",
        regex=_id_regex,
    )

    model: AllModelConfigsU = Field(
        ...,
        title="Model",
        description="Model used for finetunning",
    )
    readers: List[AllReaderConfigsU] = Field(..., title="Finetune Data")
    lr: float = Field(
        None,
        title="Learning rate",
        description="Learning rate for finetunning",
    )
    epochs: int = Field(
        None,
        title="Limit of number of models",
        description="Number of epochs for finetunning",
    )
    batch_size: Optional[int] = Field(
        None,
        title="Batch Size for Finetunning",
        description="Batch size for finetunning. If not given, the batch_size for the base model will be set.",
    )
    dry: bool = Field(
        False,
        title="Dry run",
        description="When set true, it will only return the remaining tasks",
    )

    @property
    def hash(self) -> Hash:
        return get_hash(
            str(
                (
                    [reader.invariant_json for reader in self.readers],
                    self.model.invariant_json,
                    self.lr,
                    self.epochs,
                )
            )
        )

    @property
    def readers_with_checked_type(self) -> List[ReaderConfig]:
        """Performs type conversion"""
        return [get_reader_config(reader) for reader in self.readers]

    @property
    def model_with_checked_type(
        self,
    ) -> Union[ImageFullModelConfig, TextFullModelConfig]:
        """Performs type conversion.

        Returns:
            Union[ImageFullModelConfig, TextFullModelConfig]: Full model config
            (possibly converted) with a known type.
        """
        return get_full_model_config(self.model)

from multiprocessing.sharedctypes import Value
from typing import List, Text, Union

from finetune.tunner.transformers import HFFinetuner
from finetune.tunner.vtab import VTabFinetuner
from loguru import logger
from pipeline.reader import ReaderFactory
from schemas.models.common import ImageFullModelConfig
from schemas.models.text_model import HFTextModelConfig, TextFullModelConfig
from schemas.requests.reader import AllReaderConfigsU, VTABReaderConfig


class FinetuneApp:
    def __init__(
        self,
    ) -> None:
        pass

    def finetune(
        self,
        model: Union[ImageFullModelConfig, TextFullModelConfig],
        readers: List[AllReaderConfigsU],
        hash: str,
        required_image_size,
        num_labels: int = 2,
        learning_rate: float = 0.01,
        epochs: int = 2500,
        batch_size: int = 8,
    ):
        logger.info("Finetunning with lr={}, epochs={}".format(learning_rate, epochs))
        if len(readers) > 1:
            raise ValueError("Multiple Readers not supported yet!")
        if isinstance(model, HFTextModelConfig):
            hf_finetuner = HFFinetuner()
            hf_finetuner.finetune(
                model=model,
                finetune_readers=readers,
                num_labels=num_labels,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
            )
        elif isinstance(readers[0], VTABReaderConfig):
            # here check if tftextmodel/tfimagemodel...
            vtab_finetuner = VTabFinetuner()
            vtab_finetuner.finetune(
                model=model,
                finetune_readers=readers,
                hash=hash,
                required_image_size=required_image_size,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
            )
        else:
            raise ValueError("Unsupported Model Type")

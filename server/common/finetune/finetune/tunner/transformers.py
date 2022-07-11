import os
from typing import List

import datasets
from schemas.models.text_model import HFModelConfig
from schemas.requests.reader import HFReaderConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class HFFinetuner:
    def __init__(self) -> None:
        self.storage = os.environ["TRANSFORMERS_CACHE"]

    def finetune(
        self,
        model: HFModelConfig,
        finetune_readers: List[HFReaderConfig],
        num_labels: int = 2,
        learning_rate: float = 0.01,
        epochs: int = 200,
        batch_size: int = 8,
    ):
        # Construct finetune dataset
        # TODO: Multiple Reader objects
        dataset = datasets.load_dataset(
            path=finetune_readers[0].hf_dataset_name,
            name=finetune_readers[0].configuration,
            split=finetune_readers[0].split,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model.hf_name)
        tokenized_data = dataset.map(self.preprocess, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenized_data)
        # Construct finetune model
        model = AutoModelForSequenceClassification.from_pretrained(
            model.hf_name, num_labels=num_labels
        )
        # Construct training arguments
        training_args = TrainingArguments(
            output_dir=self.storage,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

    def preprocess(self, dataset):
        return self.tokenizer(
            dataset["text"],
            truncation=True,
        )

from schemas.models.image_predefined import TorchvisionFullModelConfig
from schemas.requests.reader import TFReaderConfig

from worker_general.general.task2vec.src.interface import (
    convertReader,
    convertSHIFTModel,
)
from worker_general.general.task2vec.src.task2vec import Task2Vec

model = TorchvisionFullModelConfig(torchvision_name="ResNet18")


reader = TFReaderConfig(
    split="train",
    tf_dataset_name="cifar10:3.0.2",
    embed_feature_path=["image"],
    label_feature_path=["label"],
)


reader = convertReader(reader)

probe = convertSHIFTModel(model)

embedding = Task2Vec(probe, max_samples=1000, skip_layers=0).embed(reader)
embedding

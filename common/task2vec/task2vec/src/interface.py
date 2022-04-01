import torch
import torchvision
import torchvision.models.resnet as resnet
import torchvision.transforms as transforms
from PIL import Image
from schemas.models import ImageModelConfigsU
from schemas.requests.reader import AllReaderConfigsU

from .datasets import TfdsWrapper
from .task2vec import ProbeNetwork
from .task_similarity import pdist


class ResNet(resnet.ResNet, ProbeNetwork):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        # Saves the ordered list of layers. We need this to forward from an arbitrary intermediate layer.
        self.layers = [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            lambda z: torch.flatten(z, 1),
            self.fc,
        ]

    @property
    def classifier(self):
        return self.fc

    def forward(self, x, start_from=0):
        """Replaces the default forward so that we can forward features starting from any intermediate layer."""
        for layer in self.layers[start_from:]:
            x = layer(x)
        return x


def convertSHIFTModel(model: ImageModelConfigsU) -> ResNet:
    """
    Convert SHIFT model to a probe model.
    """
    num_classes = 1000
    if model.internal_config.id_name == "resnet18":
        layers = [2, 2, 2, 2]
    else:
        raise NotImplementedError("Only resnet18 and resnet34 are supported, yet.")
    probe_model: ProbeNetwork = ResNet(resnet.BasicBlock, layers, num_classes)
    state_dict = torch.hub.load(
        "pytorch/vision:v0.6.0", model.internal_config.id_name, pretrained=True
    ).state_dict()
    state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
    probe_model.load_state_dict(state_dict, strict=False)
    return probe_model


def convertReader(
    reader: AllReaderConfigsU, colorize=False, tfds_dir=None
) -> TfdsWrapper:
    if colorize:
        transform = torchvision.transforms.Compose(
            [
                lambda x: Image.fromarray(x.squeeze(), mode="L"),
                lambda x: x.convert("RGB"),
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
    else:
        transform = torchvision.transforms.Compose(
            [
                lambda x: Image.fromarray(x),
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
    target_reader = TfdsWrapper(
        name=reader.tf_dataset_name,
        split=reader.split,
        transform=transform,
        data_dir=tfds_dir,
    )
    return target_reader


def calculate_dist_matrix(embeddings, distance="cosine"):
    distance_matrix = pdist(embeddings, distance)
    return distance_matrix

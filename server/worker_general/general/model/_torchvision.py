import numpy as np
import torch as pt
from pipeline import DataType, Device
from pipeline.model import Model, PreprocessingSpecs
from schemas.models import TorchvisionFullModelConfig

from .preprocessing import ImageCropResize3Channels


class TorchvisionModel(Model):
    """Runs inference with a torchvision model.

    Args:
        config (TorchvisionFullModelConfig): Model configuration.
        device (Device): Device used for the inference.
    """

    def __init__(self, config: TorchvisionFullModelConfig, device: Device):
        internal_config = config.internal_config
        self._required_image_size = internal_config.required_image_size

        self._device = pt.device("cuda:0") if device == Device.GPU else pt.device("cpu")

        # No problem if loading the same model in parallel for the first time
        model = pt.hub.load(
            "pytorch/vision:v0.6.0", internal_config.id_name, pretrained=True
        )
        model.eval()
        model.to(self._device)
        self._model = model

        self._result = np.array([])
        layer_extractor_fn = internal_config.layer_extractor
        layer_extractor_fn(self._model).register_forward_hook(
            lambda _x, _y, result: self._store_result(result)
        )

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return ImageCropResize3Channels(self._required_image_size, normalize=True)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        # Swap dimensions from (batch x H x W x C) to (batch x C x H x W)
        features_pt = pt.as_tensor(
            features.transpose((0, 3, 1, 2)), dtype=pt.float32, device=self._device
        )
        with pt.no_grad():
            self._model(features_pt)
        return self._result

    def _store_result(self, result: pt.Tensor):
        self._result = result.cpu().numpy()

from pipeline import Device
from pipeline.model import Model, ModelFactory
from schemas.models import (  # PCAModelConfig,
    FinetunedTFFullImageModelConfig,
    FullModelConfig,
    HFTextModelConfig,
    ImageKerasLayerConfig,
    ImageNoOpModelConfig,
    ReshapeModelConfig,
    TextKerasLayerConfig,
    TextNoOpModelConfig,
    TFFullImageModelConfig,
    TFFullTextModelConfig,
    TorchvisionFullModelConfig,
)
from schemas.models.image_model import HFImageModelConfig

from ._huggingface import HFTextModel, HFImageModel
from ._keras_layer import ImageKerasLayer, TextKerasLayer

# from ._pca import PCAModel
from ._raw import ImageNoOpModel, ReshapeModel, TextNoOpModel
from ._tensorflow import TFImageModel, TFTextModel
from ._torchvision import TorchvisionModel


class AllModelFactory(ModelFactory):
    @staticmethod
    def get_model(config: FullModelConfig, device: Device) -> Model:
        # Proxy configs cannot be passed -> not handled
        if isinstance(config, HFTextModelConfig):
            return HFTextModel(config, device)
        
        if isinstance(config, HFImageModelConfig):
            return HFImageModel(config, device)

        if isinstance(config, ImageKerasLayerConfig):
            return ImageKerasLayer(config, device)

        if isinstance(config, ImageNoOpModelConfig):
            return ImageNoOpModel(config)

        # if isinstance(config, PCAModelConfig):
        #     return PCAModel(config)

        if isinstance(config, ReshapeModelConfig):
            return ReshapeModel(config)

        if isinstance(config, TextKerasLayerConfig):
            return TextKerasLayer(config, device)

        if isinstance(config, TextNoOpModelConfig):
            return TextNoOpModel()

        if isinstance(config, TFFullImageModelConfig):
            return TFImageModel(config, device)

        if isinstance(config, FinetunedTFFullImageModelConfig):
            return TFImageModel(config, device)

        if isinstance(config, TFFullTextModelConfig):
            return TFTextModel(config, device)

        if isinstance(config, TorchvisionFullModelConfig):
            return TorchvisionModel(config, device)

        raise RuntimeError("Unknown config")

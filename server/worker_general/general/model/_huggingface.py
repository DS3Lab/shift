from loguru import logger
import numpy as np
import torch as pt
from pipeline import DataType, Device
from pipeline.model import Model, PreprocessingSpecs
from schemas.models.text_model import HFTextModelConfig
from schemas.models.image_model import HFImageModelConfig
from transformers import AutoModel, AutoFeatureExtractor

from .preprocessing import HFPreprocessing, ImageCropResize3Channels


# Inspired by: https://discuss.pytorch.org/t/select-data-through-a-mask/45598/4
@pt.jit.script
def _get_mean_of_relevant_tokens(
    batch: pt.Tensor, attention_mask: pt.Tensor, device: pt.device
) -> pt.Tensor:
    """Computes mean of embeddings that were produced for each token individually.

    Args:
        batch (pt.Tensor): A tensor of shape
            ``batch size x #tokens x output dimension``.
        attention_mask (pt.Tensor): A tensor of shape ``batch size x #tokens`` that
            specifies which tokens are relevant.
        device (pt.device): Device on which the data is stored.

    Returns:
        pt.Tensor: The computed mean of embeddings for each sample in a batch. Shape:
        ``batch size x output dimension``.
    """
    # batch size x 1
    count_relevant_tokens = pt.sum(attention_mask, dim=1, keepdim=True)

    # Prevent division by 0 by replacing 0 values (counts) with 1
    # Why pt.tensor?
    # https://discuss.pytorch.org/t/torchscript-indexing-question-filling-nans/53100
    count_relevant_tokens[count_relevant_tokens == 0] = pt.tensor(
        1, dtype=count_relevant_tokens.dtype, device=device
    )

    # Input size: batch_size x #tokens x output_dimension
    # Output size: batch size x #tokens x output_dimension
    # 1. Change shape to output_dimension x batch size x #tokens
    # 2. Multiply with the attention mask -> this sets for each sample in a batch
    # and for all irrelevant tokens (attention mask = 0) the embedding vector to 0
    # 3. Change shape back to the original shape
    batch_w_irrelevant_set_to_zero = pt.mul(
        batch.permute(2, 0, 1), attention_mask
    ).permute(1, 2, 0)

    # Manually compute mean
    # 1. Sum tokens together in each batch to get a tensor of shape
    # batch_size x output_dimension
    # 2. Divide obtained embeddings by the count to get the mean
    return pt.div(pt.sum(batch_w_irrelevant_set_to_zero, dim=1), count_relevant_tokens)


class HFTextModel(Model):
    """Runs inference with a HuggingFace text model.

    Args:
        config (HFTextModelConfig): Model configuration.
        device (Device): Device used for the inference.
    """

    def __init__(self, config: HFTextModelConfig, device: Device):
        self._device = pt.device(
            "cuda:0") if device == Device.GPU else pt.device("cpu")

        # Prepare model
        # Not a problem if downloaded from two processes, one simply waits
        model = AutoModel.from_pretrained(config.hf_name)
        model.eval()
        model.to(self._device)
        self._model = model

        # Store config
        self._name = config.hf_name
        self._max_length = config.max_length
        self._pooled = config.pooled_output
        self._tokenizer_params = (
            config.tokenizer_params if config.tokenizer_params is not None else {}
        )

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return HFPreprocessing(
            name=self._name,
            max_length=self._max_length,
            tokenizer_params=self._tokenizer_params,
        )

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        features_pt = pt.as_tensor(
            features, dtype=pt.int64, device=self._device)
        input_ids = features_pt[:, 0, :]
        attention_mask = features_pt[:, 1, :]
        token_type_ids = features_pt[:, 2, :]

        if self._pooled:
            with pt.no_grad():
                result_pt = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )[1]

        else:
            with pt.no_grad():
                # Dimensions: batch size x #tokens x output dimension
                all_embeddings = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )[0]

            result_pt = _get_mean_of_relevant_tokens(
                all_embeddings, attention_mask, self._device
            )

        return result_pt.cpu().numpy()


class HFImageModel(Model):
    """
    Runs inference with a HuggingFace Image Model.
    Args:
        config (HFImageModelConfig): Model configuration.
        device (Device): Device used for the inference
    """

    def __init__(self, config: HFImageModelConfig, device: Device) -> None:
        self._device = pt.device(
            "cuda:0") if device == Device.GPU else pt.device("cpu")
        model = AutoModel.from_pretrained(config.hf_name)
        model.eval()
        model.to(self._device)
        self._model = model
        self._name = config.hf_name
        self._required_image_size = config.required_image_size

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE

    def get_preprocessing_specs(self) -> PreprocessingSpecs:
        return ImageCropResize3Channels(self._required_image_size, normalize=True)

    def apply_embedding(self, features: np.ndarray) -> np.ndarray:
        # Swap dimensions from (batch x H x W x C) to (batch x C x H x W)
        features_pt = pt.as_tensor(
            features.transpose((0, 3, 1, 2)), dtype=pt.float32, device=self._device
        )
        with pt.no_grad():
            result = self._model(
                features_pt,
                output_hidden_states=True
            )
        hidden_states = result.last_hidden_state[:, 0, :]
        hidden_states = hidden_states.squeeze()
        self._result = hidden_states.cpu().numpy()
        return self._result 

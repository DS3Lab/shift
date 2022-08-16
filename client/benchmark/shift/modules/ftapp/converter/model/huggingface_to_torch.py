from transformers import AutoModelForImageClassification
from torch.nn.modules import Module


def convert_huggingface_to_torch(model_identifier, num_classes):
    model = AutoModelForImageClassification.from_pretrained(
        model_identifier,
        num_labels=num_classes, ignore_mismatched_sizes=True
    )
    if isinstance(model, Module):
        return model
    else:
        raise ValueError(f"Given model is not a torch.nn.modules.Module, got {type(model)}")

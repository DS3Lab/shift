from datetime import date

import pytest
from pydantic.error_wrappers import ValidationError
from schemas.common import DateRange, IntegerRange
from schemas.models import (
    HFModelConfig,
    ImageModelInfo,
    TextModelInfo,
    TextNoOpModelConfig,
    get_all_torchvision_model_configs,
    get_predefined_image_model_configs_with_info,
    get_predefined_text_model_configs_with_info,
)
from schemas.models.common import TargetEnvironment


def test_integer_range_validation():
    range_ = IntegerRange(min=10, max=20)

    with pytest.raises(ValidationError):
        range_.min = 30

    with pytest.raises(ValidationError):
        range_.max = 5

    _ = IntegerRange(min=-5, max=-5)

    with pytest.raises(ValidationError):
        _ = IntegerRange(min=10, max=5)


def test_date_range_validation():
    range_ = DateRange(min=date(2020, 12, 5), max=date(2021, 1, 10))

    with pytest.raises(ValidationError):
        range_.min = date(2022, 1, 1)

    with pytest.raises(ValidationError):
        range_.max = (2019, 1, 1)

    _ = DateRange(min=date.today(), max=date.today())

    with pytest.raises(ValidationError):
        _ = DateRange(min=date(2021, 3, 10), max=date(2021, 2, 25))


def test_text_noop_model_config():
    # The field is there just to identify the model, the JSON should be invariant
    # wrt. the field value
    first = TextNoOpModelConfig(noop_text="noop")
    second = TextNoOpModelConfig(noop_text="")

    assert first.invariant_json == second.invariant_json


def test_hf_model_config_validation_pooled_output():
    # BERT supports pooled output
    bert = HFModelConfig(hf_name="bert-base-cased", max_length=10, pooled_output=True)

    # Cannot alter 'hf_name' into invalid state
    with pytest.raises(ValidationError):
        bert.hf_name = "xlnet-base-cased"

    # XLNet supports mean (=not using pooled output)
    xlnet = HFModelConfig(
        hf_name="xlnet-base-cased", max_length=10, pooled_output=False
    )

    # Cannot alter 'pooled_output' into invalid state
    with pytest.raises(ValidationError):
        xlnet.pooled_output = True

    # XLNet does not support pooled output
    with pytest.raises(ValidationError):
        _ = HFModelConfig(hf_name="xlnet-base-cased", max_length=10, pooled_output=True)


def test_torchvision_internal_configs():
    """Tests that all internal configs are defined."""
    torchvision_models = get_all_torchvision_model_configs()
    for model, _ in torchvision_models:
        _ = model.internal_config


def test_predefined_models():
    predefined_models = [
        *get_predefined_image_model_configs_with_info(),
        *get_predefined_text_model_configs_with_info(),
    ]

    # Check basic properties
    for predefined_model, info in predefined_models:
        assert isinstance(predefined_model.source_str, str)
        assert isinstance(predefined_model.target_environment, TargetEnvironment)
        assert isinstance(info, TextModelInfo) or isinstance(info, ImageModelInfo)

    # All predefined model JSONs must be unique
    jsons = [m.invariant_json for m, _ in predefined_models]
    assert (len(jsons)) == len(set(jsons))

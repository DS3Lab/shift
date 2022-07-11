from unittest import mock

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_datasets.core.features as tf_features
from pipeline.model import NullPreprocessingSpecs, PreprocessingSpecs

from .._tensorflow import TFReader


@pytest.fixture(scope="session")
def null_specs() -> PreprocessingSpecs:
    return NullPreprocessingSpecs()


@pytest.fixture(autouse=True)
def dataset(monkeypatch):
    """Patches _load_data to return predefined data instead of downloading the data
    from the internet."""
    np.random.seed(0)
    data = tf.data.Dataset.from_tensor_slices(
        np.random.randint(0, 255, size=(10, 10, 10, 1), dtype=np.uint8)
    ).map(lambda x: {"image": x, "label": 0, "objects": {"size": 10}})
    info = mock.NonCallableMagicMock()
    info.splits["train"].num_examples = 10
    info.features = tf_features.FeaturesDict(
        {
            "image": tf_features.Image(shape=(3, 3, 3)),
            "label": tf_features.ClassLabel(num_classes=10),
        }
    )

    monkeypatch.setattr(TFReader, "_load_data", lambda *_: (data, info))

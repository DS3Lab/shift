import os
from collections import Counter

import pytest
from schemas import READER_LABEL_FEATURE_NAME
from schemas.models.image_model import ImageSize
from schemas.requests.reader import ImageFolderReaderConfig

from ..._config import Settings
from ...model.preprocessing import ImageCropResizeFlatten
from .. import _image_folder
from .._image_folder import ImageFolderReader


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr(
        _image_folder,
        "settings",
        Settings(input_location=os.path.join(os.path.dirname(__file__), "data")),
    )


@pytest.fixture
def path_to_complete_dataset() -> str:
    return os.path.join("image_folder", "complete_dataset")


@pytest.fixture
def path_to_incomplete_dataset() -> str:
    return os.path.join("image_folder", "incomplete_dataset")


@pytest.fixture
def specs() -> ImageCropResizeFlatten:
    return ImageCropResizeFlatten(target_image_size=ImageSize(height=10, width=10))


def test_same_labels(path_to_complete_dataset, path_to_incomplete_dataset, specs):
    """Checks that labels stay the same if we remove images from subfolders (also if
    the subfolder is empty).

    complete_dataset
        first
            1.jpg
        second
            1.jpg
            2.jpg
        third
            1.jpg
            2.jpg
            3.jpg

    incomplete_dataset
        first
            1.jpg
        second
        third
            1.jpg
            3.jpg
    """
    config1 = ImageFolderReaderConfig(
        images_path=path_to_complete_dataset, use_images=True, use_labels=True
    )
    config2 = ImageFolderReaderConfig(
        images_path=path_to_incomplete_dataset, use_images=True, use_labels=True
    )

    reader1 = ImageFolderReader(config=config1, specs=specs, batch_size=None)
    reader2 = ImageFolderReader(config=config2, specs=specs, batch_size=None)

    reader1_labels = list(reader1)[0][READER_LABEL_FEATURE_NAME]
    reader2_labels = list(reader2)[0][READER_LABEL_FEATURE_NAME]

    reader2_label_count = Counter(reader2_labels)

    for label, count in Counter(reader1_labels).items():
        # Folder 'first'
        if count == 1:
            assert reader2_label_count[label] == 1

        # Folder 'second'
        elif count == 2:
            assert label not in reader2_label_count

        # Folder 'third'
        elif count == 3:
            assert reader2_label_count[label] == 2

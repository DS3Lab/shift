import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, root_validator, validator
from schemas._base import (
    READER_EMBED_FEATURE_NAME,
    READER_LABEL_FEATURE_NAME,
    Hash,
    _DefaultConfig,
    _hash_regex,
)

# Those should be the keys of the data to be embedded and labels if they are present in the reader

_embed_feature_value_error = ValueError(
    f"Name {READER_EMBED_FEATURE_NAME} cannot be used (reserved name)"
)
_label_feature_value_error = ValueError(
    f"Name {READER_LABEL_FEATURE_NAME} cannot be used (reserved name)"
)

_empty_reader_error = ValueError("Empty reader - no features present")


class ReaderConfig(BaseModel, ABC):
    """Specifies what should be known about each reader configuration."""

    def __init__(self, **data):
        """Overrides default initialization so that it can be checked whether a reader is empty on initialization."""
        super().__init__(**data)
        if not self._at_least_one_feature_present:
            raise _empty_reader_error

    def __setattr__(self, key, value):
        """Override default behaviour of updating parameters so that it can be checked whether a reader is empty after the parameter is updated."""
        super().__setattr__(key, value)
        if not self._at_least_one_feature_present:
            raise _empty_reader_error

    @property
    @abstractmethod
    def embed_feature_present(self) -> bool:
        """Returns True if reader will output a feature that will be embedded."""
        raise NotImplementedError

    @property
    @abstractmethod
    def label_feature_present(self) -> bool:
        """Returns True if reader will output labels."""
        raise NotImplementedError

    @property
    def _at_least_one_feature_present(self) -> bool:
        """Checks whether a reader is empty. The check is done externally, because checking with pydantic validators would be very tedious and repetitive.

        The method is meant to be overridden in cases when there can be additional features besides the embedded feature and the label.

        Returns:
            bool: True if there will be at least one feature present in the reader.
        """
        return self.embed_feature_present or self.label_feature_present

    @property
    def invariant_json(self) -> str:
        """JSON representation of the reader configuration that should be the same when exactly the same features (only those that will be embedded) and/or the same labels are returned by the reader."""
        dict = self.dict()
        clean_dict = {k: v for k, v in dict.items() if v}
        return json.dumps(clean_dict)


class Slice(BaseModel):
    """Determines which part of the dataset should be used."""

    start: int = Field(
        ...,
        title="Start index",
        description="Start index - inclusive, indexing starts with 0",
        example=100,
        ge=0,
    )
    stop: int = Field(
        ...,
        title="Stop index",
        description="Stop index - exclusive, indexing starts with 0",
        example=200,
        ge=0,
    )

    @root_validator
    def valid_slice(cls, values: dict) -> dict:
        if "start" in values and "stop" in values:
            start: int = values["start"]
            stop: int = values["stop"]
            if start > stop:
                raise ValueError("Invalid slice")
        return values

    class Config(_DefaultConfig):
        pass


_optional_slice_field = Field(
    None, title="Slice", description="Part of the dataset to be used"
)
_split_field = Field(..., title="Split", description="Dataset split", example="train")


class ShuffleParams(BaseModel):
    """Specifies how a dataset should be shuffled (for datasets that use a buffer for shuffling)."""

    buffer_size: int = Field(
        ...,
        title="Buffer size",
        description="Size of the buffer used for shuffling",
        example=100,
        ge=1,
    )
    seed: int = Field(
        ...,
        title="Shuffle seed",
        description="Seed to use for shuffling",
        example=123456,
        ge=0,
    )

    class Config(_DefaultConfig):
        pass


_optional_shuffle_params_field = Field(
    None, title="Shuffling parameters", description="What shuffling should be used"
)


class Feature(BaseModel):
    """Specifies how some feature from the dataset should be stored and where it can be
    found."""

    store_name: str = Field(
        ...,
        title="Name",
        description="How the feature will be named in the output file",
        example="image",
    )
    path: List[str] = Field(
        ...,
        title="Path to the feature",
        description="List of keys that lead to the desired feature",
        example=["image_small"],
        min_items=1,
    )

    class Config(_DefaultConfig):
        pass


class FeaturesMixin(ReaderConfig, ABC):
    """Implements features that are stored in a dictionary and corresponding check. Can be reused in configs with multiple inheritance."""

    embed_feature_path: Optional[List[str]] = Field(
        None,
        title="Path to the feature to embed",
        description="List of keys that lead to the feature to embed",
        example=["feature"],
        min_items=1,
    )
    label_feature_path: Optional[List[str]] = Field(
        None,
        title="Path to the label",
        description="List of keys that lead to the label",
        example=["info", "label"],
        min_items=1,
    )
    other_features: Optional[List[Feature]] = Field(
        None,
        title="Other features",
        description="Features that will not be embedded, just stored",
        example=[Feature(store_name="description", path=["image_description"])],
        min_items=1,
    )

    @property
    def embed_feature_present(self) -> bool:
        return self.embed_feature_path is not None

    @property
    def label_feature_present(self) -> bool:
        return self.label_feature_path is not None

    @property
    def _at_least_one_feature_present(self) -> bool:
        return (
            self.embed_feature_present
            or self.label_feature_present
            # If it is present, it has at least one item
            or self.other_features is not None
        )

    def get_features(self) -> Sequence[Feature]:
        """Get all requested data as Feature objects."""
        features = []
        if self.embed_feature_path is not None:
            features.append(
                Feature(
                    store_name=READER_EMBED_FEATURE_NAME, path=self.embed_feature_path
                )
            )
        if self.label_feature_path is not None:
            features.append(
                Feature(
                    store_name=READER_LABEL_FEATURE_NAME, path=self.label_feature_path
                )
            )
        if self.other_features is not None:
            features.extend(self.other_features)

        return features

    @validator("other_features")
    def unique_feature_names(
        cls,
        other_features: Optional[List[Feature]],
    ) -> Optional[List[Feature]]:
        if other_features is not None:
            all_names = [f.store_name for f in other_features]
            all_names_set = set(all_names)
            if len(all_names_set) != len(all_names):
                raise AssertionError("Feature names are not unique")
            if READER_EMBED_FEATURE_NAME in all_names_set:
                raise _embed_feature_value_error
            if READER_LABEL_FEATURE_NAME in all_names_set:
                raise _label_feature_value_error

        return other_features


class TFReaderConfig(FeaturesMixin, ReaderConfig):
    """Specifies how data from TFDS should be used."""

    slice: Optional[Slice] = _optional_slice_field
    split: str = _split_field
    tf_dataset_name: str = Field(
        ...,
        title="Pinned TensorFlow dataset",
        description="TensorFlow dataset with a pinned version (should end with "
        "':x.y.z', where x, y, z are numbers and not '*')",
        example="cifar100:3.0.2",
        regex=r"^.*:(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$",
    )
    shuffle: Optional[ShuffleParams] = _optional_shuffle_params_field

    def __len__(self):
        return len(self.slice)

    @property
    def invariant_json(self) -> str:
        return self.json(exclude={"other_features"}, exclude_none=True)

    class Config(_DefaultConfig):
        title = "TensorFlow Dataset"


class VTABNames(str, Enum):
    CALTECH_101 = "Caltech101"
    CIFAR_100 = "CIFAR-100"
    CLEVR_DISTANCE_PREDICTION = "CLEVR distance prediction"
    CLEVR_COUNTING = "CLEVR counting"
    DIABETIC_RETHINOPATHY = "Diabetic Rethinopathy"
    DMLAB = "Dmlab Frames"
    DSPRITES_ORIENTATION_PREDICTION = "dSprites orientation prediction"
    DSPRITES_LOCATION_PREDICTION = "dSprites location prediction"
    DTD = "Describable Textures Dataset (DTD)"
    EUROSAT = "EuroSAT"
    KITTI_DISTANCE_PREDICTION = "KITTI distance prediction"
    OXFORD_FLOWERS = "102 Category Flower Dataset"
    OXFORD_PET = "Oxford IIIT Pet dataset"
    PATCH_CAMELYON = "PatchCamelyon"
    RESISC45 = "Resisc45"
    SMALLNORB_AZIMUTH_PREDICTION = "Small NORB azimuth prediction"
    SMALLNORB_ELEVATION_PREDICTION = "Small NORB elevation prediction"
    # TODO: track https://github.com/tensorflow/datasets/issues/2889 OR remove line from
    #  site-packages/tensorflow_datasets/image_classification/sun397_tfds_tr.txt
    SUN397 = "SUN397"
    SVHN = "SVHN"


class VTABSplits(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TRAINVAL = "trainval"
    TEST = "test"
    TRAIN800 = "train800"
    VAL200 = "val200"
    TRAIN800VAL200 = "train800val200"


class VTABReaderConfig(ReaderConfig):
    """Specifies what how one of the VTAB datasets should be used."""

    vtab_name: VTABNames = Field(
        ...,
        title="VTAB dataset name",
        description="Name of the dataset that is a part of the VTAB",
        example=VTABNames.CIFAR_100,
    )
    split: VTABSplits = Field(
        ...,
        title="Split",
        description="One of the predefined VTAB splits",
        example=VTABSplits.TRAIN800VAL200,
    )
    use_feature: bool = True
    use_label: bool = True
    slice: Optional[Slice] = _optional_slice_field

    @property
    def embed_feature_present(self) -> bool:
        return self.use_feature

    @property
    def label_feature_present(self) -> bool:
        return self.use_label

    class Config(_DefaultConfig):
        title = "Visual Task Adaptation Benchmark (VTAB) Dataset"


class PTReaderConfig(ReaderConfig, ABC):
    """Specifies common parameters of PyTorch datasets, which are used by PTReader."""

    slice: Optional[Slice] = _optional_slice_field
    seed: Optional[int] = Field(
        None,
        title="Shuffle seed",
        description="Seed to use for shuffling",
        example=123456,
    )


class HFReaderConfig(FeaturesMixin, PTReaderConfig):
    hf_dataset_name: str = Field(
        ...,
        title="HuggingFace dataset",
        description="Name of the HuggingFace dataset to use",
        example="glue",
    )
    split: str = _split_field
    configuration: Optional[str] = Field(
        None,
        title="Dataset configuration",
        description="Some dataset require additional configuration specified: https://"
        "huggingface.co/docs/datasets/loading_datasets.html#selecting-a-configuration",
        example="mrpc",
    )

    @property
    def invariant_json(self) -> str:
        return self.json(exclude={"other_features"})

    class Config(_DefaultConfig):
        title = "HuggingFace Dataset"


class QMNISTSplit(str, Enum):
    TRAIN = "train"
    TEST = "test"
    TEST10K = "test10k"
    TEST50K = "test50k"
    NIST = "nist"


class QMNISTReaderConfig(PTReaderConfig):
    """Specifies how the QMNIST dataset should be used."""

    split: QMNISTSplit = Field(
        ...,
        title="QMNIST split",
        description="One of the predefined QMNIST splits",
        example=QMNISTSplit.TEST50K,
    )
    use_qmnist_images: bool = Field(
        ...,
        title="Use QMNIST images",
        description="Whether to use QMNIST images for embedding",
        example=True,
    )
    use_qmnist_labels: bool = Field(
        ...,
        title="Use QMNIST labels",
        description="Whether to include QMNIST labels in the output data",
        example=True,
    )

    @property
    def embed_feature_present(self) -> bool:
        return self.use_qmnist_images

    @property
    def label_feature_present(self) -> bool:
        return self.use_qmnist_labels

    class Config(_DefaultConfig):
        title = "QMNIST Dataset"


class USPSReaderConfig(PTReaderConfig):
    """Specifies how the USPS data should be used."""

    train_split: bool = Field(
        ...,
        title="USPS split",
        description="True for train split, False for test split",
        example=True,
    )
    use_usps_images: bool = Field(
        ...,
        title="Use USPS images",
        description="Whether to use USPS images for embedding",
        example=True,
    )
    use_usps_labels: bool = Field(
        ...,
        title="Use USPS labels",
        description="Whether to include QMNIST labels in the output data",
        example=True,
    )

    @property
    def embed_feature_present(self) -> bool:
        return self.use_usps_images

    @property
    def label_feature_present(self) -> bool:
        return self.use_usps_labels

    class Config(_DefaultConfig):
        title = "USPS Dataset"


class ImageFolderReaderConfig(PTReaderConfig):
    """Specifies how images stored in subfolders of a folder should be used."""

    slice: Optional[Slice] = _optional_slice_field
    images_path: str = Field(
        ...,
        title="Images path",
        description="Path to image folders relative to the specified mount location,"
        " the structure should be as described in "
        "https://pytorch.org/vision/stable/datasets.html#imagefolder "
        "IMPORTANT: All subfolders (labels) must always be present, even if they are "
        "empty, so that the same subfolder always has the same numerical label",
        example="data/images",
    )
    use_images: bool = Field(
        ...,
        title="Use images",
        description="Whether to use images for embedding",
        example=True,
    )
    use_labels: bool = Field(
        ...,
        title="Use labels",
        description="Whether to include labels in the output data",
        example=True,
    )

    @property
    def embed_feature_present(self) -> bool:
        return self.use_images

    @property
    def label_feature_present(self) -> bool:
        return self.use_labels

    class Config(_DefaultConfig):
        title = "Image Folder Dataset"


class NumPyLocation(str, Enum):
    RESULTS = "results"
    MOUNTED = "mounted"


class NumPyReaderConfig(ReaderConfig, ABC):
    """Base NumPy reader specification."""

    embed_feature: Optional[str] = Field(
        None,
        title="Embed feature",
        description="Key used to obtain embed features from the NumPy folder",
        example="image",
    )
    label_feature: Optional[str] = Field(
        None,
        title="Label feature",
        description="Key used to obtain labels from the NumPy folder",
        example="label",
    )

    @property
    def embed_feature_present(self) -> bool:
        return self.embed_feature is not None

    @property
    def label_feature_present(self) -> bool:
        return self.label_feature is not None

    @property
    def data_location(self) -> Tuple[NumPyLocation, str]:
        """Returns everything (base location + relative path) that is needed to locate
        a NumPy dataset.

        Returns:
            Tuple[NumPyLocation, str]: Specification of data location and the relative
            path to the actual data.
        """
        raise NotImplementedError


class ResultsNumPyReaderConfig(NumPyReaderConfig):
    """Specifies which data should be read from the results volume."""

    job_hash: Hash = Field(
        ...,
        title="Job hash (folder name)",
        description="Job hash (folder name) where the data can be found "
        "(relative to the specified 'numpy_location')",
        example="185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969",
        regex=_hash_regex,
    )

    @property
    def data_location(self) -> Tuple[NumPyLocation, str]:
        return NumPyLocation.RESULTS, self.job_hash

    class Config(_DefaultConfig):
        title = "NumPy Folder Dataset (From Results)"


class MountedNumPyReaderConfig(NumPyReaderConfig):
    """Specifies which data should be read from the mounted folder."""

    numpy_path: str = Field(
        ...,
        title="NumPy path",
        description="Path to the NumPy dataset relative to the specified mount "
        "location",
        example="data/numpy_dataset",
    )

    @property
    def data_location(self) -> Tuple[NumPyLocation, str]:
        return NumPyLocation.MOUNTED, self.numpy_path

    class Config(_DefaultConfig):
        title = "NumPy Folder Dataset (Mounted Folder)"


AllReaderConfigsU = Union[
    # CSVReaderConfig,
    HFReaderConfig,
    ImageFolderReaderConfig,
    MountedNumPyReaderConfig,
    QMNISTReaderConfig,
    ResultsNumPyReaderConfig,
    TFReaderConfig,
    USPSReaderConfig,
    VTABReaderConfig,
]


def get_reader_config(config: AllReaderConfigsU) -> ReaderConfig:
    """Performs type conversion from Union type (all models) to a regular type.

    Args:
        config (AllReaderConfigsU): Union type reader config.

    Returns:
        ReaderConfig: Regular type reader config.
    """

    if isinstance(config, ReaderConfig):
        return config

    raise ValueError(f"Unknown reader config {config!r}")


class QueryReaderByNameRequest(BaseModel):
    name: str = Field(..., title="name", description="The name of the reader")


class SimplifyReaderByJSONRequest(BaseModel):
    json_reader: AllReaderConfigsU = Field(
        ...,
        title="JSON of the Reader",
        description="The JSON Representations of the Reader",
    )


class GetReaderSizeByJSONRequest(BaseModel):
    json_reader: str = Field(
        ...,
        title="JSON of the Reader",
        description="The JSON Representations of the Reader",
    )


class ModelsUsedWithAReaderRequest(BaseModel):
    reader: AllReaderConfigsU = Field(
        ...,
        title="Reader",
        description="Reader that was already used with some models",
        example=VTABReaderConfig(
            vtab_name=VTABNames.CIFAR_100, split=VTABSplits.TRAIN800VAL200
        ),
    )

    @property
    def reader_config_with_checked_type(self) -> ReaderConfig:
        """Performs type conversion.

        Returns:
            ReaderConfig: Reader with a checked (converted) type.
        """
        return get_reader_config(self.reader)

    class Config(_DefaultConfig):
        title = "Reader specification"

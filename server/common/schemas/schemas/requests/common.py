import itertools
import math
import os
from collections import Counter
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

from db_tools.query import get_reader_size
from loguru import logger
from pydantic import BaseModel, Field, root_validator, validator
from schemas._base import (
    ID,
    Hash,
    _DefaultConfig,
    _hash_regex,
    _id_regex,
    generate_id,
    get_hash,
)
from schemas.classifier import ClassifierWithParams
from schemas.models import (
    AllModelConfigsU,
    ImageFullModelConfig,
    TextFullModelConfig,
    get_full_model_config,
)
from schemas.models.common import TargetEnvironment
from schemas.models.image_predefined import TorchvisionFullModelConfig
from schemas.requests.reader import (
    AllReaderConfigsU,
    ReaderConfig,
    Slice,
    TFReaderConfig,
    get_reader_config,
)


def _validate_indices(indices: List[int]) -> List[int]:
    """Checks that indices are non-negative and strictly monotonically increasing.

    Args:
        indices (List[int]): Indices to check.

    Returns:
        List[int]: Valid indices.
    """
    for i in range(1, len(indices)):
        if indices[i] < 0:
            raise ValueError(f"Invalid index {indices[0]}")
        if indices[i - 1] >= indices[i]:
            raise ValueError(
                f"Indices not strictly monotonically increasing "
                f"({indices[i - 1]} >= {indices[i]} holds)"
            )
    return indices


_indices_field = Field(
    ...,
    title="Indices",
    description="Indices of existing data to change (indexing starts with 0); "
    "indices must be unique and strictly monotonically increasing",
    example=[0, 2, 5, 7],
    min_items=0,
)

_inference_request_id_field = Field(
    ...,
    title="Inference request ID",
    description="Inference request ID that is returned to the user",
    regex=_id_regex,
)

_inference_request_hash_field = Field(
    ...,
    title="Inference request hash",
    description="Inference request hash that points to the correct data",
    regex=_hash_regex,
)


class Change(BaseModel):
    """A change that will be applied to the existing embedded data."""

    inference_request_id: ID = _inference_request_id_field
    inference_request_hash: Hash = _inference_request_hash_field

    base_indices: List[int] = _indices_field
    change_indices: List[int] = _indices_field

    embed_feature_present: bool = Field(
        ...,
        title="Embed feature present",
        description="Indicates whether the change alters the embedded feature",
    )
    label_feature_present: bool = Field(
        ...,
        title="Label feature present",
        description="Indicates whether the change alters the labels",
    )

    _valid_indices: classmethod = validator(
        "base_indices", "change_indices", allow_reuse=True
    )(_validate_indices)

    @property
    def label_only(self):
        """True if only label feature is present, False otherwise."""
        return self.label_feature_present and not self.embed_feature_present

    class Config(_DefaultConfig):
        pass


class ChangeReader(BaseModel):
    """A specification how existing embedded data should be changed."""

    reader: AllReaderConfigsU = Field(
        ..., title="Reader", description="Data that will replace the existing data"
    )
    base_indices: List[int] = _indices_field
    change_indices: List[int] = _indices_field

    _valid_indices: classmethod = validator(
        "base_indices", "change_indices", allow_reuse=True
    )(_validate_indices)

    def get_change(
        self, inference_request_id: ID, inference_request_hash: Hash
    ) -> Change:
        """Constructs a change which describes the current object after the inference has completed.

        Args:
            inference_request_id (ID): Inference request ID.
            inference_request_hash (Hash): Inference request hash.

        Returns:
            Change: Object corresponding to the current reader after the inference has completed.
        """
        reader_: ReaderConfig = get_reader_config(self.reader)

        return Change(
            inference_request_id=inference_request_id,
            inference_request_hash=inference_request_hash,
            base_indices=self.base_indices.copy(),
            change_indices=self.change_indices.copy(),
            embed_feature_present=reader_.embed_feature_present,
            label_feature_present=reader_.label_feature_present,
        )

    class Config(_DefaultConfig):
        title = "Change"


class MutableData(BaseModel):
    """Embedded data together with changes that should be applied on top of it.
    Describes where the embedded original data and the changes can be found."""

    inference_request_id: ID = _inference_request_id_field
    inference_request_hash: Hash = _inference_request_hash_field
    changes: List[Change] = Field(
        ...,
        title="Changes",
        description="A list of changes that will be applied to the original data",
    )

    @property
    def hash(self) -> Hash:
        """Returns hash, which uniquely identifies the embedded data together with changes."""
        return get_hash(
            str(
                (
                    self.inference_request_hash,
                    [
                        (c.inference_request_hash, c.base_indices, c.change_indices)
                        for c in self.changes
                    ],
                )
            )
        )

    def get_inference_request_ids(self) -> Sequence[ID]:
        """Returns all inference request IDs associated with the data and changes.

        Returns:
            Sequence[ID]: Inference request IDs.
        """
        return [self.inference_request_id] + [
            c.inference_request_id for c in self.changes
        ]

    def _get_index_first_closing_label_change(self) -> int:
        """Returns index of the first closing label change."""
        # Check if any change is not label only
        for i in reversed(range(len(self.changes))):
            if not self.changes[i].label_only:
                return i + 1

        # Default - all changes are label only
        return 0

    @property
    def without_closing_label_changes(self) -> "MutableData":
        """Returns same object, but without closing label changes."""
        copy = self.copy(deep=True)
        copy.changes = copy.changes[: self._get_index_first_closing_label_change()]
        return copy

    def get_closing_label_changes(self) -> Sequence[Change]:
        changes_copy = self.changes.copy()
        return changes_copy[self._get_index_first_closing_label_change() :]

    class Config(_DefaultConfig):
        pass


class MutableReader(BaseModel):
    """A specification how some data should be overridden with changes."""

    reader: AllReaderConfigsU = Field(
        ...,
        title="Reader",
        description="Data to use",
    )
    changes: List[ChangeReader] = Field(
        [],
        title="Changes",
        description="A list of changes that will be applied to the original data",
    )

    @validator("reader")
    def specified_embed_and_label_feature(
        cls, reader: AllReaderConfigsU
    ) -> AllReaderConfigsU:
        """Ensures that the base data contains both embed feature and label. For changes that does not need to hold - only one of them can be specified. Check that at least one is specified is already performed in the reader schema."""
        reader_: ReaderConfig = get_reader_config(reader)
        if not reader_.embed_feature_present or not reader_.label_feature_present:
            raise ValueError("Both embed and label feature must be specified")
        return reader

    class Config(_DefaultConfig):
        title = "Mutable Reader (reader with applied changes)"


class InferenceRequest(BaseModel):
    """A specification of how some data should be embedded."""

    id: ID = _inference_request_id_field
    # Need to be verbose, because JSON is parsed again in workers
    # Using regular types would not work!
    reader: AllReaderConfigsU = Field(
        ..., title="Reader", description="Reader used for inference"
    )
    model: AllModelConfigsU = Field(
        ..., title="Model", description="Model used for inference"
    )
    batch_size: int = Field(
        ..., title="Batch size", description="Batch size used for inference", ge=1
    )
    inference_type: Optional[str] = Field(
        title="Inference Type", description="training/validation", default=""
    )

    @property
    def hash(self) -> Hash:
        """Returns hash, which uniquely identifies the content of the request."""
        return get_hash(
            str(
                (
                    self.reader_config_with_checked_type.invariant_json,
                    self.model_config_with_checked_type.invariant_json,
                )
            )
        )

    @property
    def reader_config_with_checked_type(self) -> ReaderConfig:
        """Performs type conversion.

        Returns:
            ReaderConfig: Reader with a checked (converted) type.
        """
        return get_reader_config(self.reader)

    # Returns the same model, but ensures the correct type
    @property
    def model_config_with_checked_type(
        self,
    ) -> Union[ImageFullModelConfig, TextFullModelConfig]:
        """Performs type conversion.

        Returns:
            Union[ImageFullModelConfig, TextFullModelConfig]: Full model config
            (possibly converted) with a known type.
        """
        return get_full_model_config(self.model)

    @property
    def target_environment(self) -> TargetEnvironment:
        """Specifies in which environment the inference should run.

        Returns:
            TargetEnvironment: The target environment.
        """
        return self.model_config_with_checked_type.target_environment

    class Config(_DefaultConfig):
        title = "Inference request"


class Task2VecRequest(BaseModel):
    """A specification of how a reader should be embedded into a vector"""

    id: Optional[ID] = Field(
        title="Task2Vec request ID",
        description="ID of the request",
        regex=_id_regex,
        example=None,
    )
    probe: TorchvisionFullModelConfig = Field(
        ...,
        title="Probe",
        description="The Probe Model used for embedding, must be resnet18 or resnet34",
        example=TorchvisionFullModelConfig(torchvision_name="ResNet18"),
    )
    reader: AllReaderConfigsU = Field(
        ...,
        title="Reader",
        description="Reader used for embedding",
        example=TFReaderConfig(
            split="train",
            tf_dataset_name="cifar10:3.0.2",
            embed_feature_path=["image"],
            label_feature_path=["label"],
        ),
    )

    @property
    def hash(self) -> Hash:
        return get_hash(
            str(
                (
                    self.reader.invariant_json,
                    self.probe.invariant_json,
                )
            )
        )

    @property
    def reader_config_with_checked_type(self) -> ReaderConfig:
        """Performs type conversion.

        Returns:
            ReaderConfig: Reader with a checked (converted) type.
        """
        return get_reader_config(self.reader)

    # Returns the same model, but ensures the correct type
    @property
    def probe_config_with_checked_type(
        self,
    ) -> ImageFullModelConfig:
        """Performs type conversion."""
        return get_full_model_config(self.probe)


class ClassifierRequest(BaseModel):
    """A specification of how a classifier should be trained on embedded data."""

    id: ID = Field(
        ...,
        title="Classifier request ID",
        description="Classifier request ID that is returned to the user",
        regex=_id_regex,
    )
    classifier: ClassifierWithParams = Field(
        ..., title="Classifier", description="Classifier to use"
    )
    train: List[MutableData] = Field(
        ..., title="Train data", description="Train data used with a classifier"
    )
    test: List[MutableData] = Field(
        ..., title="Test data", description="Test data used with a classifier"
    )

    @property
    def hash(self) -> Hash:
        """Returns hash, which uniquely identifies the classifier request."""
        return get_hash(
            str(
                (
                    self.classifier.value,
                    [t.hash for t in self.train],
                    [t.hash for t in self.test],
                )
            )
        )

    def get_request_without_closing_label_changes(self) -> "ClassifierRequest":
        """Returns same object, but without closing label changes for train and test data."""
        copy = self.copy(deep=True)
        copy.train = [md.without_closing_label_changes for md in copy.train]
        copy.test = [md.without_closing_label_changes for md in copy.test]
        return copy

    @property
    def hash_without_closing_label_changes(self) -> Hash:
        return self.get_request_without_closing_label_changes().hash

    def get_closing_train_label_changes(self) -> Dict[int, Sequence[Change]]:
        return {i: md.get_closing_label_changes() for i, md in enumerate(self.train)}

    def get_closing_test_label_changes(self) -> Dict[int, Sequence[Change]]:
        return {i: md.get_closing_label_changes() for i, md in enumerate(self.test)}

    def get_inference_request_ids(self) -> Sequence[ID]:
        """Returns all inference request IDs associated with the classifier request.

        Returns:
            Sequence[ID]: Inference request IDs.
        """
        result: List[ID] = []
        for md in self.train:
            result.extend(md.get_inference_request_ids())
        for md in self.test:
            result.extend(md.get_inference_request_ids())

        return result

    class Config(_DefaultConfig):
        title = "Classifier request"


class HyperbandRequest(BaseModel):
    """A specification of the hyperband optimized requests."""

    id: ID = _inference_request_id_field
    # Need to be verbose, because JSON is parsed again in workers
    # Using regular types would not work!
    train: List[MutableReader] = Field(
        ..., title="Train", description="Reader used for inference"
    )
    test: List[MutableReader] = Field(
        ..., title="Test", description="Reader used for inference"
    )
    models: List[AllModelConfigsU] = Field(
        ..., title="Models", description="Model used for inference"
    )
    chunk_size: int = Field(
        ..., title="Chunk size", description="Chunk size used for inference", ge=1
    )
    limit: int = Field(
        ..., title="Limit", description="Maximum number of models", get=1
    )
    classifier: ClassifierWithParams = Field(
        None,
        title="Classifier",
        description="Classifier to use on top of the embedded data",
    )
    budget: Optional[int] = Field(
        None, title="budget", description="Budget for running hyperband requests"
    )

    @property
    def hash(self) -> Hash:
        """Returns hash, which uniquely identifies the content of the request."""
        return get_hash(
            str(
                (
                    [t.json for t in self.train],
                    [t.json for t in self.test],
                    [m.invariant_json for m in self.models],
                    self.classifier.value,
                    self.chunk_size,
                    self.budget,
                    self.limit,
                )
            )
        )

    @property
    def reader_config_with_checked_type(self) -> ReaderConfig:
        """Performs type conversion.

        Returns:
            ReaderConfig: Reader with a checked (converted) type.
        """
        return get_reader_config(self.reader)

    # Returns the same model, but ensures the correct type
    @property
    def model_config_with_checked_type(
        self,
    ) -> Union[ImageFullModelConfig, TextFullModelConfig]:
        """Performs type conversion.

        Returns:
            Union[ImageFullModelConfig, TextFullModelConfig]: Full model config
            (possibly converted) with a known type.
        """
        return get_full_model_config(self.model)

    @property
    def target_environment(self) -> TargetEnvironment:
        """Specifies in which environment the inference should run.

        Returns:
            TargetEnvironment: The target environment.
        """
        return self.model_config_with_checked_type.target_environment

    class Config(_DefaultConfig):
        title = "Hyperband request"

    def generate_requests(
        self,
        model_idx: int,
        classifier: ClassifierWithParams,
        current_index: int,
        increment: int,
        sizes: List[int],
        needed_num_pulls: List[int],
    ):
        """generates normal requests from hyperband requests"""
        # we only slice the train readers
        # the test readers are not sliced, as the evaluation is performed on the entire test set
        # this case is different from the splitting for multi-GPU.
        sliced_readers = slice_readers(
            sizes,
            self.train,
            self.chunk_size,
            current_index,
            needed_num_pulls,
        )
        logger.info(sliced_readers)
        return Request(
            train=sliced_readers,
            test=self.test,
            models=[self.models[model_idx]],
            classifiers=[classifier],
        )


class Request(BaseModel):
    """Base request meant to be used by the user."""

    train: List[MutableReader] = Field(
        ...,
        title="Train data",
        description="Data used for training with classifiers, or simply data that will be embedded",
        min_items=1,
    )
    test: Optional[List[MutableReader]] = Field(
        None,
        title="Test data",
        description="Test data used with classifiers (for simple embedding 'train' should be used)",
        min_items=1,
    )
    models: List[AllModelConfigsU] = Field(
        ..., title="Models", description="Models used to embed the data", min_items=1
    )
    classifiers: Optional[List[ClassifierWithParams]] = Field(
        None,
        title="Classifiers",
        description="Classifiers to use on top of the embedded data",
        min_items=1,
    )
    benchmark: bool = Field(
        False,
        title="Benchmark",
        description="Specify if the request is a benchmark request - if so the server will not merge test dataset",
    )
    chunk_size: Optional[Union[int, Literal['AUTO']]] = Field(
        None,
        title="Chunk Size for Hyperband Optimization",
        description="Chunk size for Hyperband optimization",
    )
    budget: Optional[Union[int, Literal['AUTO']]] = Field(
        None,
        title="Budget for Hyperband Optimization",
        description="Budget for Hyperband optimization",
    )
    limit: Optional[int] = Field(
        None,
        title="Limit of number of models",
        description="The maximum number of models",
    )
    dry: bool = Field(
        False,
        title="Dry run",
        description="When set true, it will only return the remaining tasks",
    )

    @property
    def full_model_configs(
        self,
    ) -> Sequence[Union[ImageFullModelConfig, TextFullModelConfig]]:
        return [get_full_model_config(m) for m in self.models]

    @validator("classifiers")
    def unique_classifiers(
        cls, classifiers: Optional[List[ClassifierWithParams]]
    ) -> Optional[List[ClassifierWithParams]]:
        if classifiers is not None:
            counts = Counter(classifiers)
            for classifier in counts:
                if counts[classifier] > 1:
                    raise ValueError(
                        f"Duplicated classifier {classifier!r} ({counts[classifier]} "
                        f"occurrences)"
                    )
        return classifiers

    @root_validator()
    def classifiers_depend_on_test_data(cls, values: dict) -> dict:
        # Only validate when validation on individual fields succeeded
        if "test" in values and "classifiers" in values:
            test = values["test"]
            classifiers = values["classifiers"]
            if test is None and classifiers is not None:
                raise ValueError(
                    "'classifiers' should not be specified when 'test' is not specified"
                )
            if test is not None and classifiers is None:
                raise ValueError(
                    "'test' should not be set when only inference is run, use 'train' instead"
                )
        return values

    def _get_inference_request(
        self,
        reader: AllReaderConfigsU,
        model_index: int,
        batch_size: int,
        is_change: False,
    ) -> List[InferenceRequest]:
        """Generates an appropriate inference request."""

        if not is_change:
            """
            Since change is usually small, we do not split if it is a change request.
            """
            num_gpus = len(os.environ["SHIFT_DEVICES"].split(","))
            reader_size = get_reader_size(reader.invariant_json)
            if reader_size is not None and num_gpus > 1:
                reader_size = int(reader_size)

                """
                Here we split the inference request into multiple smaller pieces
                We only split the request when N>8000 and batch_size < 256
                
                Assume the given batch size could occupy all mem
                The goal is to split the tasks to occupy all GPU at the same time
                * The larger the reader -> more splits -> smaller interval
                * The larger the batch_size -> smaller models --> less splits --> larger interval
                * more gpu -> more splits -> smaller interval
                * maybe num_gpus is good enough?
                """
                if reader_size > int(
                    os.environ["SHIFT_SPLIT_SIZE_THRESHOLD"]
                ) and batch_size < int(os.environ["SHIFT_SPLIT_BATCH_THRESHOLD"]):

                    inference_requests = []
                    if os.environ["SHIFT_SPLIT_CHUNK_SIZE"]:
                        interval = int(os.environ["SHIFT_SPLIT_CHUNK_SIZE"]) - 1
                        # here the splits is floor(reader_size / interval)
                        # e.g., we have 10 samples in reader, interval = 1
                        # [0,9] is splitted into [0,1],[1,2]...,[8,9]
                        splits = math.floor(reader_size / interval)
                    else:
                        splits = num_gpus
                        interval = math.ceil(reader_size / splits)
                    for i in range(splits):
                        reader = reader.copy(deep=True)
                        start = i * (interval + 1)
                        stop = (
                            start + interval
                            if start + interval < reader_size
                            else reader_size
                        )
                        reader.slice = Slice(
                            start=start,
                            stop=stop,
                        )
                        ir = InferenceRequest(
                            reader=reader,
                            model=self.models[model_index].copy(deep=True),
                            id=generate_id(),
                            batch_size=batch_size,
                        )
                        inference_requests.append(ir)
                    return inference_requests
        return [
            InferenceRequest(
                reader=reader.copy(deep=True),
                model=self.models[model_index].copy(deep=True),
                id=generate_id(),
                batch_size=batch_size,
            )
        ]

    def _process_mutable_readers(
        self,
        mutable_readers: List[MutableReader],
        model_index: int,
        batch_size: int,
    ) -> Tuple[List[MutableData], List[InferenceRequest]]:
        """Prepares requests for inference and objects that describe where the data can be found after the generated requests have finished.

        Args:
            mutable_readers (List[MutableReader]): Mutable readers (readers + changes) that should be embedded.
            model_index (int): Index of the model within a current request that should be used for embedding.
            batch_size (int): Batch size used for inference with the current model.

        Returns:
            Tuple[List[MutableData], List[InferenceRequest]]:
            1. A list of objects that describe where the original data and the changes can be found after the inference has completed.
            2. A list of requests that describe which readers and models should be used for inference.
        """
        mds: List[MutableData] = []
        irs: List[InferenceRequest] = []

        for mr in mutable_readers:
            # Reader
            reader_irs = self._get_inference_request(
                reader=mr.reader,
                model_index=model_index,
                batch_size=batch_size,
                is_change=False,
            )
            irs.extend(reader_irs)
            # Changes
            changes: List[Change] = []
            if mr.changes is not None:
                for change_reader in mr.changes:
                    change_ir = self._get_inference_request(
                        reader=change_reader.reader,
                        model_index=model_index,
                        batch_size=batch_size,
                        is_change=True,
                    )[0]
                    changes.append(
                        change_reader.get_change(
                            inference_request_id=change_ir.id,
                            inference_request_hash=change_ir.hash,
                        )
                    )
                    irs.append(change_ir)

            for reader_ir in reader_irs:
                mds.append(
                    MutableData(
                        inference_request_id=reader_ir.id,
                        inference_request_hash=reader_ir.hash,
                        changes=changes,
                    )
                )
        return mds, irs

    def generate_hyperband_requests(self) -> HyperbandRequest:
        """Generate hyperband requests, if multiple models are provided"""
        assert (
            len(self.models) > 1
        ), "There should be more than one candidate models to perform hyperband optimization"

        assert self.chunk_size > 0, "Chunk size should be positive"
        return HyperbandRequest(
            id=generate_id(),
            train=self.train,
            test=self.test,
            models=self.models,
            chunk_size=self.chunk_size,
            budget=self.budget,
            limit=self.limit,
        )

    def generate_requests(
        self, batch_sizes: Sequence[int]
    ) -> Tuple[List[InferenceRequest], List[ClassifierRequest]]:
        """Generate all requests (inference and classifier) needed to fulfill the general (this) request.

        Args:
            batch_sizes (Sequence[int]): Batch size used for inference for each model.

        Returns:
            Tuple[List[InferenceRequest], List[ClassifierRequest]]: Lists of inference requests and classifier requests that correspond to this request.
        """

        assert len(batch_sizes) == len(
            self.models
        ), "There should be the same number of batch sizes as there are models"

        inference_requests_per_model: List[List[InferenceRequest]] = []
        classifier_requests_per_model: List[List[ClassifierRequest]] = []
        for model_index in range(len(self.models)):
            (
                train_mutable_data,
                train_inference_requests,
            ) = self._process_mutable_readers(
                mutable_readers=self.train,
                model_index=model_index,
                batch_size=batch_sizes[model_index],
            )
            # Case when test data is specified and classifiers none is not permitted
            if self.test is not None and self.classifiers is not None:
                (
                    test_mutable_data,
                    test_inference_requests,
                ) = self._process_mutable_readers(
                    mutable_readers=self.test,
                    model_index=model_index,
                    batch_size=batch_sizes[model_index],
                )
                inference_requests_per_model.append(
                    test_inference_requests + train_inference_requests
                )
                if self.benchmark:
                    classifier_requests_per_model.append(
                        [
                            ClassifierRequest(
                                classifier=classifier,
                                train=train_mutable_data,
                                test=[benchmark_test_mutable_data],
                                id=generate_id(),
                            )
                            for classifier in self.classifiers
                            for benchmark_test_mutable_data in test_mutable_data
                        ]
                    )
                else:
                    classifier_requests_per_model.append(
                        [
                            ClassifierRequest(
                                classifier=classifier,
                                train=train_mutable_data,
                                test=test_mutable_data,
                                id=generate_id(),
                            )
                            for classifier in self.classifiers
                        ]
                    )

            else:
                inference_requests_per_model.append(train_inference_requests)
        # Reorder inference requests (R = reader, M = model, x = #reader, y = #model)
        # from: (R1,M1),...,(Rx,M1),...,(R1,My),...,(Rx,My)
        # to  : (R1,M1),...,(R1,My),...,(R2,M1),...,(Rx,My)
        # so that different models get executed simultaneously rather than multiple readers of one model
        """
        Now we cannot only look at the the first object, as there might be different number of requests
        """
        inference_requests: List[InferenceRequest] = []
        max_num_irs = max([len(ir) for ir in inference_requests_per_model])
        for j in range(max_num_irs):
            for i in range(len(inference_requests_per_model)):
                # append only when there is an ir
                if j < len(inference_requests_per_model[i]):
                    inference_requests.append(inference_requests_per_model[i][j])
        if self.classifiers is not None:
            # Reorder classifier requests
            # (C = classifier, M = model, x = #classifier, y = #model)
            # from: (C1,M1),...,(Cx,M1),...,(C1,My),...,(Cx,My)
            # to  : (C1,M1),...,(C1,My),...,(C2,M1),...,(Cx,My)
            # so that different models get executed simultaneously rather than multiple classifiers of one model
            classifier_requests: List[ClassifierRequest] = [
                classifier_requests_per_model[i][j]
                # Outer loop (iterates through classifiers)
                for j in range(len(classifier_requests_per_model[0]))
                # Inner loop (iterates through models)
                for i in range(len(classifier_requests_per_model))
            ]
        else:
            classifier_requests = []
        return inference_requests, classifier_requests

    class Config(_DefaultConfig):
        title = "Request"


def slice_readers(
    sizes: List[int],
    readers: List[MutableReader],
    chunk_size: int,
    current_index: int,
    needed_num_pulls: List[int],
    changes: List[Change] = [],
) -> List[MutableReader]:
    """
    Now we don't split the readers into smaller, equal-size chunks, but we determine the start and end of the reader.
    Several notes:
    1. start -> current_index
    2. end -> current_index + chunk_size * needed_num_pulls
    3. the reader shall be splitted as
    [0, chunk_size-1]... [chunk_size (current_index),  chunk_size * i]...[]

    # below is deprecated
    This is used for hyperband requests.
    How we divide the train readers.
        Assume there are R train readers, each has the size Rs, and there is only one model.
        We want to split the train readers into structures like
        b_1 = (R1[0: R1_slice_1], R2[0:R2_slice_1], ....)
        b_2 = (R1[0: R1_slice_1], R2[0:R2_slice_1], ...., R1[R1_slice_1+1: R1_slice_2], ...)
    How we determine the R1_slice_1, ...
        1. the param increment is only to the first reader.
        2. Assume there are N_x samples in the first reader, we find other readers as
                increment_1/N_1 = increment_2/N2
            --> increment_2 = N_2 * increment_1 / N_1
    The goal is to utilise existing results to the maximal extent.
    > In future, maybe we can set the increment as an exponent of 2, and persist it in the database?
    > Maybe we can let user define the increments?
    """
    increments = [chunk_size]
    base_size = sizes[0]
    for size in sizes:
        increments.append(math.ceil((size * chunk_size) / base_size))
    sliced_readers = []
    needed_num_pulls = list(itertools.accumulate(needed_num_pulls))
    for id_reader, reader in enumerate(readers):
        slices = []
        start = 0
        stop = start
        logger.info(
            "stop: {}, current_index: {}, increments: {}, needed_num_pulls:{}".format(
                stop, current_index, increments[id_reader], needed_num_pulls
            )
        )
        for pull_id in needed_num_pulls[1:]:
            stop = start+increments[id_reader] * pull_id
            if stop >= sizes[id_reader]:
                stop = sizes[id_reader]
                slice = Slice(start=start, stop=stop)
                slices.append(slice)
                break
            else:
                slice = Slice(start=start, stop=stop)
                slices.append(slice)
                start = stop
        
        for slice in slices:
            sliced_reader = reader.reader
            sliced_reader.slice = slice
            # here we should find the changes in this slice
            # get the entire changes of this reader and perform a filter operation
            sliced_change_readers = []
            for change in reader.changes:
                needed_indices = []
                new_base_indices = []

                for i, x in enumerate(change.base_indices):
                    if x > slice.start and x < slice.stop:
                        needed_indices.append(i)
                        # here we need to deduce the start...
                        # i.e., x -> x-start in this slice
                        new_base_indices.append(x-slice.start)
                new_change_indices = [
                    x
                    for i, x in enumerate(change.change_indices)
                    if i in needed_indices
                ]
                new_change_reader = ChangeReader(
                    reader=change.reader,
                    base_indices=new_base_indices,
                    change_indices=new_change_indices,
                )
                sliced_change_readers.append(new_change_reader)
            if len(sliced_change_readers) > 0:
                sliced_mutable_reader = MutableReader(
                    reader=sliced_reader, changes=sliced_change_readers
                )
            else:
                sliced_mutable_reader = MutableReader(reader=sliced_reader)
            sliced_readers.append(sliced_mutable_reader)

    return sliced_readers

import pickle

from celery import Task
from celery.utils.log import get_task_logger
from finetune.interface import FinetuneApp
from pydantic.env_settings import BaseSettings
from schemas import READER_EMBED_FEATURE_NAME, Status
from schemas.requests.common import InferenceRequest, Task2VecRequest
from schemas.requests.finetune import FinetuneRequest
from task2vec.src.interface import convertReader, convertSHIFTModel
from task2vec.src.task2vec import Task2Vec
from common.telemetry.telemetry import add_event
import timeit

from ._base import DataType, Device
from ._config import settings
from .io import NumPyWriter
from .model import ModelFactory, NullModel, PreprocessingSpecs
from .reader import ReaderFactory

__all__ = ["InferenceRunner", "FinetuneRunner"]

_logger = get_task_logger(__name__)


class _StatusUpdater:
    def __init__(self, task: Task):
        self._task = task

    def update(self, message: str):
        _logger.info("Status was updated with message: %r", message)
        self._task.update_state(state=Status.RUNNING, meta={
                                "additional": message})


class Task2VecRunner:
    """Runs task2vec based on the specified task2vec request on the specified device."""

    def __init__(
        self,
        reader_factory: ReaderFactory,
        model_factory: ModelFactory,
        settings: BaseSettings,
    ):
        self._reader_factory = reader_factory
        self._model_factory = model_factory
        self._settings = settings

    def __call__(
        self,
        task2vec_request: Task2VecRequest,
        device: Device,
        celery_task: Task,
    ):
        """Executes the task2vec job"""
        status_updater = _StatusUpdater(celery_task)
        reader_config = task2vec_request.reader_config_with_checked_type
        probe_config = task2vec_request.probe_config_with_checked_type
        # TODO: here we check if the probe is allowed
        status_updater.update("Starting Task2vec")
        task2vec_reader = convertReader(
            reader_config, tfds_dir=self._settings.tfds_location
        )
        task2vec_probe = convertSHIFTModel(probe_config)
        embedding = Task2Vec(task2vec_probe, max_samples=1000, skip_layers=0).embed(
            task2vec_reader
        )
        result_path = settings.get_results_path_str(task2vec_request.hash)
        with open(result_path, "wb") as output:
            pickle.dump(embedding, output, pickle.HIGHEST_PROTOCOL)
        status_updater.update(
            "Finished Task2Vec, result saved to {}".format(result_path)
        )


class InferenceRunner:
    """Runs inference based on the specified inference request on the specified device.

    Args:
        reader_factory (ReaderFactory): Factory used to obtain the reader from its specification.
        model_factory (ModelFactory): Factory used to obtain the model from its specification.
    """

    def __init__(self, reader_factory: ReaderFactory, model_factory: ModelFactory):
        self._reader_factory = reader_factory
        self._model_factory = model_factory

    def __call__(
        self,
        inference_request: InferenceRequest,
        device: Device,
        celery_task: Task,
    ):
        """Executes the inference job.

        Args:
            inference_request (InferenceRequest): Inference job specification.
            device (Device): Device to use for inference.
            celery_task (Task): Celery task used for providing updates regarding the job progress.
        """
        # Used for updating status of current inference
        status_updater = _StatusUpdater(celery_task)

        # 1. Get configs with checked type
        reader_config = inference_request.reader_config_with_checked_type
        model_config = inference_request.model_config_with_checked_type

        # 2. Load model if that is needed
        start = timeit.default_timer()
        status_updater.update("Loading model (reader not loaded yet)")
        if reader_config.embed_feature_present:
            model = self._model_factory.get_model(model_config, device)
        else:
            model = NullModel()

        preprocessing_specs: PreprocessingSpecs = model.get_preprocessing_specs()
        stop = timeit.default_timer()
        # add_event('load_model', {
        #     'model': model_config.invariant_json,
        # }, round(1000 * (stop-start)))
        # 3. Load reader
        start = timeit.default_timer()
        status_updater.update("Loading reader (model loaded)")
        reader = self._reader_factory.get_reader(
            reader_config, inference_request.batch_size, preprocessing_specs
        )
        stop = timeit.default_timer()
        # add_event('load_reader', {
        #     'reader': reader_config.invariant_json,
        # }, round(1000 * (stop-start)))
        # 4. Check compatibility of reader and model
        if (
            reader.data_type != model.data_type
            and reader.data_type != DataType.UNKNOWN
            and model.data_type != DataType.UNKNOWN
        ):
            raise ValueError(
                f"Incompatible type {reader.data_type!r} with {model.data_type!r}"
            )

        # 5. Run inference
        start = timeit.default_timer()
        status_updater.update("Starting with inference")
        writer = NumPyWriter(
            settings.get_results_path_str(inference_request.hash))

        for data_index, current_dictionary in enumerate(reader):
            if (data_index + 1) % 10 == 0:
                status_updater.update(f"Processing batch {data_index + 1}")

            # Embed data if there is data to embed
            if reader_config.embed_feature_present:
                embedded_feature = model.apply_embedding(
                    current_dictionary[READER_EMBED_FEATURE_NAME]
                )
                current_dictionary[READER_EMBED_FEATURE_NAME] = embedded_feature

            # Store embedded data
            writer.add(current_dictionary)

        writer.finalize()
        stop = timeit.default_timer()
        # add_event('inference', {
        #     'model': model_config.invariant_json,
        #     'reader':reader_config.invariant_json
        # }, round(1000 * (stop-start)))

class FinetuneRunner:
    """Runs finetune job based on the specified finetune request on the specified device"""

    def __init__(
        self,
        reader_factory: ReaderFactory,
        model_factory: ModelFactory,
        settings: BaseSettings,
    ) -> None:
        self._reader_factory = reader_factory
        self._model_factory = model_factory
        self._settings = settings
        self.finetuneapp = FinetuneApp()

    def __call__(
        self, finetune_request: FinetuneRequest, device: Device, celery_task: Task
    ):
        """Executes the Finetune Job"""
        status_updater = _StatusUpdater(celery_task)
        status_updater.update("Loading model (reader not loaded yet)")
        readers_config = finetune_request.readers_with_checked_type
        model_config = finetune_request.model_with_checked_type
        # we take one sample from readers_config to check if there is a feature present
        # all readers_config should have the same embed_feature_present.
        # TODO: Add a check
        if readers_config[0].embed_feature_present:
            model = self._model_factory.get_model(model_config, device)
        else:
            model = NullModel()
        status_updater.update("Starting Finetunning...")
        self.finetuneapp.finetune(
            model=model_config,
            readers=readers_config,
            hash=finetune_request.hash,
            learning_rate=finetune_request.lr,
            epochs=finetune_request.epochs,
            required_image_size=model._required_image_size,
        )

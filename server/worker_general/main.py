from typing import Optional

from billiard.process import current_process
from celery import Task
from celery.utils.log import get_task_logger
from general.celery import celery_app
from pipeline.request import FinetuneRunner
from schemas.classifier import Classifier
from schemas.requests.common import Task2VecRequest
from schemas.requests.finetune import FinetuneRequest
from schemas.response import LinearResult
import timeit
from common.telemetry.telemetry import add_event

_logger = get_task_logger(__name__)

@celery_app.task(bind=True)
def run_inference(task: Task, inference_request_json: str, device_id: Optional[str]):
    current_process().daemon = False
    try:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = device_id if device_id is not None else ""
        # https://github.com/tensorflow/tensorflow/issues/1258#issuecomment-261365022
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        from general.model import AllModelFactory
        from general.reader import AllReaderFactory
        from pipeline import Device
        from pipeline.request import InferenceRunner
        from schemas.requests.common import InferenceRequest

        inference_runner = InferenceRunner(
            reader_factory=AllReaderFactory(), model_factory=AllModelFactory()
        )
        inference_runner(
            inference_request=InferenceRequest.parse_raw(inference_request_json),
            device=Device.GPU if device_id is not None else Device.CPU,
            celery_task=task,
        )

    except Exception as e:
        _logger.exception(str(e))
        raise RuntimeError


@celery_app.task
def run_classifier(
    classifier_request_json: str,
    partial_result_json: Optional[str],
    device_id: Optional[str],
):
    current_process().daemon = True
    try:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = device_id if device_id is not None else ""
        from general.classifier import linear_classifier, nearest_neighbors
        from pipeline import Device
        from schemas.requests.common import ClassifierRequest
        from schemas.response import NearestNeighborResult
        start = timeit.default_timer()
        request = ClassifierRequest.parse_raw(classifier_request_json)
        if request.classifier.name in [Classifier.COSINE_NN, Classifier.EUCLIDEAN_NN]:
            result = nearest_neighbors(
                request=request,
                nn_result=NearestNeighborResult.parse_raw(partial_result_json)
                if partial_result_json is not None
                else None,
                device=Device.GPU if device_id is not None else Device.CPU,
            )
        elif request.classifier.name == Classifier.LINEAR:
            result = linear_classifier(
                request=request,
                linear_result=LinearResult.parse_raw(partial_result_json)
                if partial_result_json is not None
                else None,
                device=Device.GPU if device_id is not None else Device.CPU,
                settings=request.classifier.parameters,
            )
        stop = timeit.default_timer()
        # add_event(
        #     'classifier',
        #     {
        #         'classifier': request.classifier.name,
        #         'device': "GPU" if device_id is not None else "CPU",
        #         'hash': request.hash
        #     },
        #     round(1000 * (stop - start))
        # )
        return {key: result[key].json() for key in result}

    except Exception as e:
        _logger.exception(str(e))
        raise RuntimeError


@celery_app.task(bind=True)
def run_task2vec(
    task: Task,
    task2vec_request_json: str,
    device_id: Optional[str],
):
    # according to docs:
    # howtos.html#why-is-daemon-set-to-false-at-the-beginning-of-celery-tasks
    current_process().daemon = False
    try:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = device_id if device_id is not None else ""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        from general._config import settings
        from general.model import AllModelFactory
        from general.reader import AllReaderFactory
        from pipeline.request import Task2VecRunner

        task2vec_runner = Task2VecRunner(
            reader_factory=AllReaderFactory(),
            model_factory=AllModelFactory(),
            settings=settings,
        )
        task2vec_runner(
            Task2VecRequest.parse_raw(task2vec_request_json), device_id, task
        )
    except Exception as e:
        _logger.exception(str(e))
        raise RuntimeError


@celery_app.task(bind=True)
def run_finetune(task: Task, finetune_request_json: str, device_id: Optional[str]):
    current_process().daemon = False
    try:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = device_id if device_id is not None else ""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        from general._config import settings
        from general.model import AllModelFactory
        from general.reader import AllReaderFactory

        finetuner = FinetuneRunner(
            reader_factory=AllReaderFactory(),
            model_factory=AllModelFactory(),
            settings=settings,
        )
        finetuner(FinetuneRequest.parse_raw(finetune_request_json), device_id, task)
    except Exception as e:
        _logger.exception(str(e))
        raise RuntimeError

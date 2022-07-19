import logging
import threading
from abc import ABC, abstractmethod
from time import sleep
from typing import Callable, Dict, List, NamedTuple, Tuple

import celery.states as states
from celery.result import AsyncResult
from db_tools.postgres import JobsDBInterface
from db_tools.queues.finetune import FinetuneQueue
from db_tools.queues.hyperband import HyperbandQueue
from db_tools.redis import ClassifierDeps, InferenceQueue, StatusMap, Task2vecQueue
from loguru import logger
from optimizations.interface import ShiftArm
from optimizations.successive_halving import SuccessiveHalving
from schemas import ID, Hash, Status
from schemas.classifier import Classifier
from schemas.models.common import TargetEnvironment
from schemas.requests.common import (
    ClassifierRequest,
    HyperbandRequest,
    InferenceRequest,
    Task2VecRequest,
)
from schemas.requests.finetune import FinetuneRequest
from schemas.response import LinearResult, NearestNeighborResult, StatusResponse

from ._devices import DeviceID, DeviceManager

_logger = logging.getLogger(__name__)

# Following two classes store information about running inference and classifier jobs, respectively. At runtime there is one object per unique request (those that have the same hash). This way, the same request is not executed multiple times. IDs of equivalent requests are stored in 'ids'.


class _InferenceJobInfo(NamedTuple):
    request: InferenceRequest
    device_id: DeviceID
    ids: List[ID]
    celery_id: str


class _ClassifierJobInfo(NamedTuple):
    request: ClassifierRequest
    device_id: DeviceID
    ids: List[ID]
    celery_id: str


class _Task2VecJobInfo(NamedTuple):
    device_id: DeviceID
    celery_id: str
    request: Task2VecRequest
    ids: List[ID]


class _HyperbandJobInfo(NamedTuple):
    request: HyperbandRequest
    device_id: DeviceID
    ids: List[ID]
    celery_id: str


class _FinetuneJobInfo(NamedTuple):
    request: FinetuneRequest
    device_id: DeviceID
    celery_id: str
    ids: List[ID]


# Parameters used by Celery to schedule the job correctly


class RemoteJobParams(NamedTuple):
    general_inference_job_name: str
    general_classifier_job_name: str
    general_task2vec_job_name: str
    general_finetune_job_name: str
    general_queue_name: str
    tf_1_inference_job_name: str
    tf_1_queue_name: str


class CeleryJobManager(ABC):
    """Manages communication with Celery."""

    @abstractmethod
    def start_job(self, job_name: str, args: tuple, queue: str) -> str:
        """Starts a job using Celery.

        Args:
            job_name (str): Name of the remote function.
            args (tuple): Arguments passed to the remote function.
            queue (str): Name of the queue into which the job will be put.

        Returns:
            str: Celery ID which can be used with get_result to obtain the job status.
        """
        raise NotImplementedError

    @abstractmethod
    def get_status(self, celery_id: str) -> AsyncResult:
        """Returns the current status of the job and the result if relevant.

        Args:
            celery_id (str): Job ID assigned by Celery.

        Returns:
            AsyncResult: Job status and result if relevant.
        """
        raise NotImplementedError


class Runner:
    """Checks for new inference and classifier jobs and scheduler them when there is a free device. Progress of running jobs is tracked and dependencies between different request are taken into account. Furthermore, it is ensured that same request is not executed twice if the same request has successfully finished in the past, is currently waiting for execution or is being executed at the moment.

    However, if the same request has failed in the past, then it is executed again.

    Internally, hashes and IDs are used to identify requests. Hash denotes the meaning of the request and is same for two equivalent requests. On the other hand, ID is different for two requests, even if they have the same meaning. Each request that is a duplicate of some other request is simply marked as successful if its equivalent has completed successfully in the past. If its equivalent is running at the moment, then the new request adopts its status.

    Dependencies are resolved based on IDs rather than hashes. This way, if some job failed in the past, a new job that depends on that job will not automatically fail, but will wait for the execution of the new equivalent job.

    Args:
        check_interrupted (Callable[[], bool]): A function which when called without any parameters returns True if the runner should stop processing new requests and monitoring running jobs and False if runner should continue with its operation.
        celery (CeleryJobManager): Interface for communication with Celery. device_manager (DeviceManager): Device manager.
        remote_job_params (RemoteJobParams): Parameters relevant for scheduling Celery jobs.
        jobs_db (JobsDBInterface): Interface for communication with PostgreSQL.
        redis_data (Tuple[InferenceQueue, StatusMap, ClassifierDeps]): Interfaces for communication with Redis.
        no_sleep (bool): True is there should be no delay between iterations of scheduling new jobs and checking existing jobs, False otherwise.
    """

    def __init__(
        self,
        check_interrupted: Callable[[], bool],
        check_reload: Callable[[], bool],
        reset_reload: Callable[[], bool],
        celery: CeleryJobManager,
        device_manager: DeviceManager,
        remote_job_params: RemoteJobParams,
        jobs_db: JobsDBInterface,
        redis_data: Tuple[
            Task2vecQueue,
            InferenceQueue,
            StatusMap,
            ClassifierDeps,
            HyperbandQueue,
            FinetuneQueue,
        ],
        no_sleep: bool = False,
    ):
        self.sh_threads: List[threading.Thread] = []
        self._check_interrupted = check_interrupted
        self._check_reload = check_reload
        self._reset_reload = reset_reload
        self._celery = celery
        self._device_manager = device_manager
        self._remote_job_params = remote_job_params
        self._jobs_db = jobs_db
        (
            self._task2vec_queue,
            self._inference_queue,
            self._status_map,
            self._classifier_deps,
            self._hyperband_queue,
            self._finetune_queue,
        ) = redis_data
        self._no_sleep = no_sleep

        # All inference and classifier requests that have successfully finished in the past. The set and dictionary are extended dynamically at runtime without synchronization with PostgreSQL
        self._successful_ir_hashes = (
            self._jobs_db.get_successful_inference_request_hashes()
        )
        self._successful_tr_hashes = (
            self._jobs_db.get_successful_task2vec_request_hashes()
        )
        self._successful_fr_hashes = (
            self._jobs_db.get_successful_finetune_request_hashes()
        )

        _logger.info(
            "Retrieved successful inference request hashes: %r",
            self._successful_ir_hashes,
        )

        self._successful_cr_hashes_to_errors = (
            self._jobs_db.get_successful_classifier_request_hashes_and_errors()
        )
        _logger.info(
            "Retrieved successful classifier request hashes and errors: %r",
            self._successful_cr_hashes_to_errors,
        )

        # All currently running inference and classifier requests together with info
        # Dictionary keys are used to determine whether a job is currently running
        self._hash_to_inference_job_info: Dict[Hash, _InferenceJobInfo] = {}
        self._hash_to_classifier_job_info: Dict[Hash, _ClassifierJobInfo] = {}
        self._hash_to_task2vec_job_info: Dict[Hash, _Task2VecJobInfo] = {}
        self._hash_to_hyperband_job_info: Dict[Hash, _HyperbandJobInfo] = {}
        self._hash_to_finetune_job_info: Dict[Hash, _FinetuneJobInfo] = {}

    def _reload(self):
        _logger.info("Reloading...")
        self.sh_threads: List[threading.Thread] = []
        self._hash_to_inference_job_info: Dict[Hash, _InferenceJobInfo] = {}
        self._hash_to_classifier_job_info: Dict[Hash, _ClassifierJobInfo] = {}
        self._hash_to_task2vec_job_info: Dict[Hash, _Task2VecJobInfo] = {}
        self._hash_to_hyperband_job_info: Dict[Hash, _HyperbandJobInfo] = {}
        self._hash_to_finetune_job_info: Dict[Hash, _FinetuneJobInfo] = {}
        self._successful_ir_hashes = (
            self._jobs_db.get_successful_inference_request_hashes()
        )
        self._successful_tr_hashes = (
            self._jobs_db.get_successful_task2vec_request_hashes()
        )
        self._successful_fr_hashes = (
            self._jobs_db.get_successful_finetune_request_hashes()
        )
        self._successful_cr_hashes_to_errors = (
            self._jobs_db.get_successful_classifier_request_hashes_and_errors()
        )

        _logger.info(
            "Retrieved successful inference request hashes: %r",
            self._successful_ir_hashes,
        )
        self._reset_reload()

    def _start_successive_halving_job(self):
        _logger.debug("Starting Successive Halving")
        any_request_ready = not self._hyperband_queue.empty()
        while self._device_manager.any_cpu_free() and any_request_ready:
            hr = self._hyperband_queue.get()
            logger.info([test.reader.invariant_json for test in hr.test])
            for test in hr.test:
                size = self._jobs_db.get_reader_size(test.reader.invariant_json)
                logger.info(size)
            total_test_size = sum(
                [
                    self._jobs_db.get_reader_size(test.reader.invariant_json)
                    for test in hr.test
                ]
            )
            train_sizes = [
                self._jobs_db.get_reader_size(train.reader.invariant_json)
                for train in hr.train
            ]
            total_train_size = sum(train_sizes)
            arms = {
                str(model): ShiftArm(
                    hr,
                    classifier=hr.classifier,
                    model_idx=idx,
                    initial_error=total_test_size,
                    jobs_db=self._jobs_db,
                    iq=self._inference_queue,
                    sm=self._status_map,
                    cd=self._classifier_deps,
                    sizes=train_sizes,
                )
                for idx, model in enumerate(hr.models)
            }
            sh_algorithm = SuccessiveHalving(arms)
            thread = threading.Thread(
                target=sh_algorithm.run,
                kwargs={
                    "eta": 2,
                    "budget": hr.budget,
                    "jobs_db": self._jobs_db,
                    "job_hash": hr.hash,
                },
            )
            self.sh_threads.append(thread)
            thread.start()
            # sh_algorithm.run(eta=2, budget=hr.budget, jobs_db=self._jobs_db, job_hash=hr.hash)
            any_request_ready = not self._hyperband_queue.empty()

    def _start_inference_job(
        self, inference_request: InferenceRequest, device_id: DeviceID
    ) -> _InferenceJobInfo:
        """Starts an inference job on the specified device."""
        is_general_environment = (
            inference_request.target_environment == TargetEnvironment.GENERAL
        )
        _logger.info("inference request model...")
        _logger.info(inference_request.model.invariant_json)
        celery_id = self._celery.start_job(
            job_name=self._remote_job_params.general_inference_job_name
            if is_general_environment
            else self._remote_job_params.tf_1_inference_job_name,
            args=(
                inference_request.json(),
                device_id,
            ),
            queue=self._remote_job_params.general_queue_name
            if is_general_environment
            else self._remote_job_params.tf_1_queue_name,
        )
        logger.info("inference-{}-started".format(inference_request.hash))
        _logger.info(
            "Inference request %s - started on device %s",
            inference_request.id,
            device_id,
        )
        _logger.info("%s", str(self._device_manager))

        return _InferenceJobInfo(
            request=inference_request,
            device_id=device_id,
            ids=[inference_request.id],
            celery_id=celery_id,
        )

    def _start_classifier_job(
        self, classifier_request: ClassifierRequest, device_id: DeviceID
    ) -> _ClassifierJobInfo:
        """Start a classifier job on the specified device."""
        # TODO: NN - change logic for retrieving partial results as there might be more than one suitable partial result
        # Retrieve partial result if exists

        if classifier_request.classifier.name in [
            Classifier.COSINE_NN,
            Classifier.EUCLIDEAN_NN,
        ]:
            partial_result = self._jobs_db.get_nn_result(
                classifier_request.hash_without_closing_label_changes
            )
        else:
            # TODO: Fetch partial results for linear classification?
            partial_result = None
        _logger.info(
            "Classifier job %s - partial result retrieved (exists): %s",
            classifier_request.id,
            partial_result is not None,
        )
        celery_id = self._celery.start_job(
            job_name=self._remote_job_params.general_classifier_job_name,
            args=(
                classifier_request.json(),
                partial_result.json() if partial_result is not None else None,
                device_id,
            ),
            queue=self._remote_job_params.general_queue_name,
        )
        logger.info("classifier-{}-started".format(classifier_request.hash))
        _logger.info(
            "Classifier request %s - started on device %s",
            classifier_request.id,
            device_id,
        )
        _logger.info("%s", str(self._device_manager))

        return _ClassifierJobInfo(
            device_id=device_id,
            ids=[classifier_request.id],
            celery_id=celery_id,
            request=classifier_request,
        )

    def _start_task2vec_job(
        self, task2vec_request: Task2VecRequest, device_id: DeviceID
    ):
        """Start a task2vec job on the specified device."""

        celery_id = self._celery.start_job(
            job_name=self._remote_job_params.general_task2vec_job_name,
            args=(
                task2vec_request.json(),
                device_id,
            ),
            queue=self._remote_job_params.general_queue_name,
        )
        _logger.info(
            "Task2vec request %s - started on device %s",
            task2vec_request.id,
            device_id,
        )
        return _Task2VecJobInfo(
            request=task2vec_request,
            device_id=device_id,
            ids=[task2vec_request.id],
            celery_id=celery_id,
        )

    def _start_finetune_job(
        self, finetune_request: FinetuneRequest, device_id: DeviceID
    ):
        """Start a finetune job on the specified device."""
        celery_id = self._celery.start_job(
            job_name=self._remote_job_params.general_finetune_job_name,
            args=(
                finetune_request.json(),
                device_id,
            ),
            queue=self._remote_job_params.general_queue_name,
        )
        _logger.info(
            "Finetune request %s - started on device %s",
            finetune_request.id,
            device_id,
        )
        return _FinetuneJobInfo(
            request=finetune_request,
            device_id=device_id,
            ids=[finetune_request.id],
            celery_id=celery_id,
        )

    def _schedule_inference_jobs(self):
        """Schedules as many inference jobs as possible, based on the jobs in queue and available GPU devices."""

        _logger.debug("Scheduling inference jobs")
        any_request_ready = not self._inference_queue.empty()
        while self._device_manager.any_gpu_free() and any_request_ready:
            # This is the only process that pops elements from the queue, there is no way that a queue would be empty
            ir = self._inference_queue.get()

            # 1. Run inference if the same job has not been run in the past and is not running at the moment
            if (
                ir.hash not in self._successful_ir_hashes
                and ir.hash not in self._hash_to_inference_job_info
            ):
                # Get free device
                device_id = self._device_manager.get_free_gpu()
                _logger.debug(
                    "Inference job %s - will be run on device %s", ir.id, device_id
                )

                # Start job and store currently running job
                inference_job_info = self._start_inference_job(ir, device_id)
                self._hash_to_inference_job_info[ir.hash] = inference_job_info
                self._status_map[ir.id] = StatusResponse(status=Status.STARTED)

            # 2. Same job already running - do not run the same job again, but only update the info, so that the updates of the equivalent job (same hash, different IDs) also apply for the current job
            elif ir.hash in self._hash_to_inference_job_info:
                self._hash_to_inference_job_info[ir.hash].ids.append(ir.id)
                self._status_map[ir.id] = StatusResponse(status=Status.STARTED)
                _logger.info(
                    "Inference job %s  - same job already running, updating info", ir.id
                )

            # 3. Same job finished in the past - mark it as finished
            elif ir.hash in self._successful_ir_hashes:
                self._status_map[ir.id] = StatusResponse(
                    status=Status.FINISHED, additional=ir.hash
                )
                _logger.info(
                    "Inference job %s - same job finished in the past, setting status to finished",
                    ir.id,
                )
            any_request_ready = not self._inference_queue.empty()

    def _schedule_classifier_jobs(self):
        """Schedules as many classifier jobs as possible, based on the job dependencies and available devices."""

        # Check whether any classifier request has all dependencies fulfilled and return all jobs that cannot be executed, because one of their dependencies failed
        _logger.debug("Scheduling classifier jobs")
        failed = self._classifier_deps.update_dependencies(self._status_map)

        for main_id, dep_id in failed:
            self._status_map[main_id] = StatusResponse(
                status=Status.FAILED,
                additional=f"Caused by failure of {dep_id}",
            )
            _logger.info(
                "Classifier job %s - failed because of failed dependency %s",
                main_id,
                dep_id,
            )

        # Initially set to True to check whether any device is actually free
        any_device_free = True
        any_request_ready = self._classifier_deps.any_request_ready()

        while any_device_free and any_request_ready:
            gpu_free = self._device_manager.any_gpu_free()
            cpu_free = self._device_manager.any_cpu_free()

            if gpu_free:

                # This is the only process that removes requests, there is no way that no request would be ready
                cr = self._classifier_deps.get_ready_request()

                # 1. Run classifier if the same job has not successfully finished in the past and is not running at the moment
                if (
                    cr.hash not in self._successful_cr_hashes_to_errors
                    and cr.hash not in self._hash_to_classifier_job_info
                ):
                    # Get free device
                    if gpu_free:
                        device_id = self._device_manager.get_free_gpu()
                    # else:
                    #    device_id = self._device_manager.get_free_cpu()
                    _logger.debug(
                        "Classifier job %s - will be run on device %s", cr.id, device_id
                    )

                    # Start job and store currently running job
                    classifier_job_info = self._start_classifier_job(cr, device_id)
                    self._hash_to_classifier_job_info[cr.hash] = classifier_job_info
                    self._status_map[cr.id] = StatusResponse(status=Status.FINISHED)

                # 2. Same job already running - do not run the same job again, but only
                # update the info, so that the updates of the equivalent job (same hash,
                # different IDs) also apply for the current job
                elif cr.hash in self._hash_to_classifier_job_info:
                    self._hash_to_classifier_job_info[cr.hash].ids.append(cr.id)
                    self._status_map[cr.id] = StatusResponse(status=Status.STARTED)
                    _logger.info(
                        "Classifier job %s - same job already running, updating info",
                        cr.id,
                    )

                # 3. Same job finished in the past - mark it as finished
                elif cr.hash in self._successful_cr_hashes_to_errors:
                    self._status_map[cr.id] = StatusResponse(
                        status=Status.FINISHED,
                        additional=self._successful_cr_hashes_to_errors[cr.hash],
                    )
                    _logger.info(
                        "Classifier job %s - same job finished in the past, setting "
                        "status to finished",
                        cr.id,
                    )
            else:
                any_device_free = False
            any_request_ready = self._classifier_deps.any_request_ready()

    def _schedule_task2vec_jobs(self):
        """Schedules as many task2vec jobs as possible, based on the jobs in queue and available devices."""
        _logger.debug("Scheduling task2vec jobs")
        any_request_ready = not self._task2vec_queue.empty()
        while self._device_manager.any_gpu_free() and any_request_ready:
            tr = self._task2vec_queue.get()
            if (
                tr.hash not in self._successful_tr_hashes
                and tr.hash not in self._hash_to_task2vec_job_info
            ):

                device_id = self._device_manager.get_free_gpu()
                _logger.debug(
                    "Task2Vec job %s - will be run on device %s", tr.id, device_id
                )
                # Start job and store currently running job
                task2vec_job_info = self._start_task2vec_job(tr, device_id)
                self._hash_to_task2vec_job_info[tr.hash] = task2vec_job_info
                self._status_map[tr.id] = StatusResponse(status=Status.STARTED)
            # same job already running
            elif tr.hash in self._hash_to_task2vec_job_info:
                self._hash_to_task2vec_job_info[tr.hash].ids.append(tr.id)
                self._status_map[tr.id] = StatusResponse(status=Status.STARTED)
                _logger.info(
                    "Task2Vec job %s - same job already running, updating info", tr.id
                )
            elif tr.hash in self._successful_tr_hashes:
                self._status_map[tr.id] = StatusResponse(
                    status=Status.FINISHED, additional=tr.hash
                )
                _logger.info(
                    "Task2Vec job %s - same job finished in the past, setting status to finished",
                    tr.id,
                )
            any_request_ready = not self._task2vec_queue.empty()

    def _schedule_finetune_jobs(self):
        _logger.debug("Scheduling finetunning jobs")
        any_request_ready = not self._finetune_queue.empty()
        while self._device_manager.any_gpu_free() and any_request_ready:
            fr = self._finetune_queue.get()
            if (
                fr.hash not in self._successful_fr_hashes
                and fr.hash not in self._hash_to_finetune_job_info
            ):
                device_id = self._device_manager.get_free_gpu()
                _logger.debug(
                    "Finetune job %s - will be run on device %s", fr.id, device_id
                )
                finetune_job_info = self._start_finetune_job(fr, device_id)
                self._hash_to_finetune_job_info[fr.hash] = finetune_job_info
                self._status_map[fr.id] = StatusResponse(status=Status.STARTED)

            elif fr.hash in self._hash_to_finetune_job_info:
                self._hash_to_finetune_job_info[fr.hash].ids.append(fr.id)
                self._status_map[fr.id] = StatusResponse(status=Status.STARTED)
                _logger.info(
                    "Finetune job %s - same job already running, updating info", fr.id
                )

            elif fr.hash in self._successful_fr_hashes:
                self._status_map[fr.id] = StatusResponse(
                    status=Status.FINISHED, additional=fr.hash
                )
                _logger.info(
                    "Finetune job %s - same job finished in the past, setting status to finished",
                    fr.id,
                )
            any_request_ready = not self._finetune_queue.empty()

    def _check_inference_jobs(self):
        """Checks the status of inference job and takes care of successful and failed jobs."""
        to_delete = []

        _logger.debug("Checking inference jobs")

        # Iterate through unique inference jobs
        for hash_ in self._hash_to_inference_job_info.keys():
            info = self._hash_to_inference_job_info[hash_]
            status = self._celery.get_status(info.celery_id)

            # Default values
            response_additional_info = None
            response_status = Status.RUNNING

            # Job was successful or failed
            if (
                status.state == states.SUCCESS
                or status.state in states.EXCEPTION_STATES
            ):
                if status.state in states.EXCEPTION_STATES:
                    response_status = Status.FAILED
                    _logger.info("Inference job (hash=%s) - failed", hash_)

                elif status.state == states.SUCCESS:
                    self._successful_ir_hashes.add(hash_)
                    self._jobs_db.store_inference_job(info.request)
                    response_additional_info = hash_
                    response_status = Status.FINISHED
                    logger.info("inference-{}-finished".format(hash_))
                    _logger.info("Inference job (hash=%s) - finished", hash_)

                self._device_manager.release_device(info.device_id)
                _logger.info("%s", str(self._device_manager))
                to_delete.append(hash_)
                status.forget()

            # Job is still running - get new status
            else:
                result = status.result
                if (
                    result is not None
                    and isinstance(result, dict)
                    and "additional" in result
                ):
                    response_additional_info = result["additional"]
                _logger.debug("Inference job (hash=%s) - still running", hash_)

            # Update status for all equivalent jobs
            for job_id in info.ids:
                self._status_map[job_id] = StatusResponse(
                    status=response_status,
                    additional=response_additional_info,
                )
                _logger.debug("Inference job %s - updated status", job_id)

        for key in to_delete:
            del self._hash_to_inference_job_info[key]

    def _check_classifier_jobs(self):
        """Checks the status of classifier jobs and takes care of successful and failed jobs."""

        to_delete = []

        # Iterate through unique classifier jobs
        for hash_ in self._hash_to_classifier_job_info.keys():

            info = self._hash_to_classifier_job_info[hash_]
            status = self._celery.get_status(info.celery_id)
            # Default values
            response_additional_info = None
            response_status = Status.RUNNING

            # Job was successful or failed
            if (
                status.state == states.SUCCESS
                or status.state in states.EXCEPTION_STATES
            ):
                if status.state in states.EXCEPTION_STATES:
                    response_status = Status.FAILED
                    _logger.info("Classifier job (hash=%s) - failed", hash_)

                elif status.state == states.SUCCESS:
                    logger.info("classifier-{}-finished".format(hash_))
                    _logger.info("Classifier job (hash=%s) - finished", hash_)

                    # TODO: NN - change logic for storing partial results
                    def save_nn_result(h: Hash, nnr: NearestNeighborResult):
                        self._successful_cr_hashes_to_errors[h] = nnr.error
                        self._jobs_db.store_nearest_neighbor_job(h, nnr)
                        self._jobs_db.store_known_result(
                            h, nnr, self._hash_to_classifier_job_info[h].request
                        )

                    def save_linear_result(h: Hash, lr: LinearResult):
                        self._successful_cr_hashes_to_errors[h] = lr.error
                        self._jobs_db.store_linear_job(h, lr)
                        self._jobs_db.store_known_result(
                            h, lr, self._hash_to_classifier_job_info[h].request
                        )

                    # check if it is linear or nn classifier
                    if info.request.classifier.name == Classifier.LINEAR:
                        linear_result = LinearResult.parse_raw(status.result[hash_])
                        save_linear_result(hash_, linear_result)
                        response_additional_info = linear_result.error

                    elif info.request.classifier.name in [
                        Classifier.COSINE_NN,
                        Classifier.EUCLIDEAN_NN,
                    ]:
                        # Store the main result
                        nn_result = NearestNeighborResult.parse_raw(
                            status.result[hash_]
                        )
                        save_nn_result(hash_, nn_result)
                        response_additional_info = nn_result.error
                    response_status = Status.FINISHED

                    # Store the sub-result if present
                    for result_hash in status.result:
                        if result_hash != hash_:
                            _logger.debug(
                                "Classifier partial result (hash= %s ) - storing",
                                result_hash,
                            )
                            save_nn_result(
                                result_hash,
                                NearestNeighborResult.parse_raw(
                                    status.result[result_hash]
                                ),
                            )

                self._device_manager.release_device(info.device_id)
                _logger.info("%s", str(self._device_manager))
                to_delete.append(hash_)
                status.forget()

            # Update status for all equivalent jobs
            for user_id in info.ids:
                self._status_map[user_id] = StatusResponse(
                    status=response_status,
                    additional=response_additional_info,
                )
                _logger.debug("Classifier job %s - updated status", user_id)

        for key in to_delete:
            del self._hash_to_classifier_job_info[key]

    def _check_task2vec_jobs(self):
        """Checks the status of task2vec job and takes care of successful and failed jobs"""
        to_delete = []
        _logger.debug("Checking Task2Vec jobs")

        for hash_ in self._hash_to_task2vec_job_info.keys():
            info = self._hash_to_task2vec_job_info[hash_]
            status = self._celery.get_status(info.celery_id)

            # Default Value
            response_additional_info = None
            response_status = Status.RUNNING
            if (
                status.state == states.SUCCESS
                or status.state in states.EXCEPTION_STATES
            ):
                if status.state in states.EXCEPTION_STATES:
                    response_status = Status.FAILED
                    _logger.info("Task2Vec job (hash=%s) - failed", hash_)
                elif status.state == states.SUCCESS:
                    self._successful_ir_hashes.add(hash_)
                    self._jobs_db.store_task2vec_job(info.request)
                    response_additional_info = hash_
                    response_status = Status.FINISHED
                    _logger.info("Task2Vec job (hash=%s) - finished", hash_)
                self._device_manager.release_device(info.device_id)
                _logger.info("%s", str(self._device_manager))
                to_delete.append(hash_)
                status.forget()
            else:
                result = status.result
                if (
                    result is not None
                    and isinstance(result, dict)
                    and "additional" in result
                ):
                    response_additional_info = result["additional"]
                _logger.debug("Task2Vec job (hash=%s) - still running", hash_)
            for job_id in info.ids:
                self._status_map[job_id] = StatusResponse(
                    status=response_status,
                    additional=response_additional_info,
                )
                _logger.debug("Task2Vec job %s - updated status", job_id)

        for key in to_delete:
            del self._hash_to_task2vec_job_info[key]

    def _check_finetune_jobs(self):
        to_delete = []
        _logger.debug("Checking finetune jobs")
        for hash_ in self._hash_to_finetune_job_info.keys():
            info = self._hash_to_finetune_job_info[hash_]
            status = self._celery.get_status(info.celery_id)

            response_additional_info = None
            response_status = Status.RUNNING
            if (
                status.state == states.SUCCESS
                or status.state in states.EXCEPTION_STATES
            ):
                if status.state in states.EXCEPTION_STATES:
                    response_status = Status.FAILED
                    _logger.info("Finetune job (hash=%s) - failed", hash_)
                elif status.state == states.SUCCESS:
                    self._successful_fr_hashes.add(hash_)
                    self._jobs_db.store_finetune_job(
                        self._hash_to_finetune_job_info[hash_].request
                    )
                    _logger.info("Finetune job (hash=%s) - finished", hash_)
                self._device_manager.release_device(info.device_id)
                _logger.info("%s", str(self._device_manager))
                to_delete.append(hash_)
                status.forget()
            else:
                result = status.result
                if (
                    result is not None
                    and isinstance(result, dict)
                    and "additional" in result
                ):
                    response_additional_info = result["additional"]
                _logger.debug("Finetune job (hash=%s) - still running", hash_)
            for job_id in info.ids:
                self._status_map[job_id] = StatusResponse(
                    status=response_status,
                    additional=response_additional_info,
                )
                _logger.debug("Finetune job %s - updated status", job_id)
        for key in to_delete:
            del self._hash_to_finetune_job_info[key]

    def run(self):
        """Main method that calls all other private methods."""

        # Periodically check if the runner was externally interrupted
        while not self._check_interrupted():
            if self._check_reload():
                self._reload()
            if not self._no_sleep:
                sleep(2)
            # We put the hyperband job in another thread for some reasons:
            # 1. Workers do not have access to the redis queue, which is required for hyperband job.
            # 2. We need to continue to process other jobs, otherwise, even though hyperband requests has created new jobs, it won't be processed.
            # 3. It is essentially only about checking status, so it is fine to not put it into computationally-heavily workers.
            self._start_successive_halving_job()
            # self._proceed_successive_halving()
            # self._proceed_successive_halving()
            # Give priority to classifier requests - if there is free GPU,
            # classifiers will be executed first
            # order is important!
            self._schedule_classifier_jobs()
            self._schedule_inference_jobs()
            self._schedule_task2vec_jobs()
            self._schedule_finetune_jobs()
            # Check status of all running jobs and store results
            self._check_inference_jobs()
            self._check_classifier_jobs()
            self._check_task2vec_jobs()
            self._check_finetune_jobs()

        _logger.info("Runner has been interrupted")

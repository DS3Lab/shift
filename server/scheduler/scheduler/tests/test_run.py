from typing import Any, Dict, NamedTuple, Optional, Sequence, Set
from unittest import mock
from uuid import uuid4

from celery import states
from celery.result import AsyncResult
from db_tools.postgres import JobsDBInterface
from db_tools.redis import ClassifierDeps, InferenceQueue, StatusMap
from schemas import ID, Hash, Status, generate_id, get_hash
from schemas.request import InferenceRequest
from schemas.response import NearestNeighborResult, StatusResponse

from .._devices import DeviceManager
from .._run import CeleryJobManager, RemoteJobParams, Runner

remote_job_params = RemoteJobParams("gi", "gc", "gq", "tf1i", "tf1q")


# Stores everything needed to mock the result of job or its failure
class _MockedJob(NamedTuple):
    request_id: ID
    num_iters_running: int
    fail: bool = False
    result: Any = None


class MockedCeleryJobManager(CeleryJobManager):
    def __init__(self, jobs: Sequence[_MockedJob]):
        self._request_id_to_job = {j.request_id: j for j in jobs}
        self._started_job_request_ids: Set[ID] = set()
        self._id_to_status: Dict[str, Any] = dict()
        self._id_to_num_iters_remaining: Dict[str, int] = dict()
        self._id_to_job: Dict[str, _MockedJob] = dict()

    def start_job(self, job_name: str, args: tuple, queue: str) -> str:
        assert job_name in {
            remote_job_params.general_inference_job_name,
            remote_job_params.general_classifier_job_name,
            remote_job_params.tf_1_inference_job_name,
        }
        assert queue in {
            remote_job_params.general_queue_name,
            remote_job_params.tf_1_queue_name,
        }

        for request_id in self._request_id_to_job:
            if request_id == args[0]:
                job: _MockedJob = self._request_id_to_job[request_id]
                break
        else:
            raise AssertionError("Corresponding job not found")

        celery_id = uuid4().hex
        self._id_to_status[celery_id] = mock.NonCallableMock()
        self._id_to_num_iters_remaining[celery_id] = job.num_iters_running
        self._id_to_job[celery_id] = job
        self._started_job_request_ids.add(job.request_id)
        return celery_id

    def get_status(self, celery_id: str) -> AsyncResult:
        status = self._id_to_status[celery_id]
        job: _MockedJob = self._id_to_job[celery_id]

        if self._id_to_num_iters_remaining[celery_id] == 0:
            status.state = states.FAILURE if job.fail else states.SUCCESS
            status.result = None if job.fail else job.result
        else:
            status.state = states.STARTED
            self._id_to_num_iters_remaining[celery_id] -= 1

        return status

    def get_started_job_ids(self) -> Set[ID]:
        """Returns all job IDs (not Celery IDs) that were requested to start."""
        return self._started_job_request_ids

    def check_forget_called(self):
        """Checks that .forgot() was called on all statuses (AsyncResult). See warning:
        https://docs.celeryproject.org/en/stable/reference/celery.result.html"""
        for id_ in self._id_to_status:
            self._id_to_status[id_].forget.assert_called()


class MockedJobsDB(JobsDBInterface):
    def __init__(
        self, existing_ir_hashes: Set[Hash], existing_cr_results: Dict[Hash, float]
    ):
        # Stored at runtime
        self._stored_inference_job_hashes: Set[Hash] = set()
        self._stored_classifier_job_hashes: Set[Hash] = set()

        # Present before
        self._existing_ir_hashes = existing_ir_hashes
        self._existing_cr_results = existing_cr_results

    def store_inference_job(self, req: InferenceRequest):
        self._stored_inference_job_hashes.add(req.hash)

    def store_nearest_neighbor_job(
        self, classifier_job_hash: Hash, nn_result: NearestNeighborResult
    ):
        self._stored_classifier_job_hashes.add(classifier_job_hash)

    def get_nn_result(self, nn_job_hash: Hash) -> Optional[NearestNeighborResult]:
        pass

    def get_successful_inference_request_hashes(self) -> Set[Hash]:
        return self._existing_ir_hashes

    def get_successful_classifier_request_hashes_and_errors(self) -> Dict[Hash, float]:
        return self._existing_cr_results

    def get_stored_inference_job_hashes(self) -> Set[Hash]:
        return self._stored_inference_job_hashes

    def get_stored_classifier_job_hashes(self) -> Set[Hash]:
        return self._stored_classifier_job_hashes


class PredefinedNumIters:
    """Interrupts runtime after the predefined number of iterations.

    Args:
        num_iters (int): Predefined number of iterations.
    """

    def __init__(self, num_iters: int):
        self._num_iters = num_iters

    def __call__(self) -> bool:
        if self._num_iters == 0:
            return True
        self._num_iters -= 1
        return False


def generate_inference_request_mock(hash_: str = None):
    id_ = generate_id()
    ir = mock.NonCallableMock()
    ir.id = id_
    ir.hash = get_hash(id_) if hash_ is None else hash_
    ir.json = lambda: id_
    return ir


def generate_classifier_request_mock(
    deps: Sequence[ID], hash_: str = None, parent_hash: str = None
):
    cr = generate_inference_request_mock(hash_=hash_)
    cr.get_inference_request_ids = lambda: deps
    cr.hash_without_closing_label_changes = (
        get_hash(cr.hash) if parent_hash is None else parent_hash
    )
    return cr


def test_inference_jobs():
    # Allow two concurrent jobs
    device_manager = DeviceManager(gpu_ids_string="0,1", max_cpu_jobs=0)

    ir_1 = generate_inference_request_mock()
    ir_2 = generate_inference_request_mock()
    ir_3 = generate_inference_request_mock(hash_=ir_1.hash)
    ir_4 = generate_inference_request_mock(hash_=ir_2.hash)
    ir_5 = generate_inference_request_mock()

    # Timeline:
    # # Job 1 runs normally - should be scheduled
    # # Job 2 fails - should be scheduled (fails afterwards)
    # # Job 3 is equivalent to job 1 and could be scheduled while job 1 is still
    # # running - should not be scheduled again
    # # Job 4 could be scheduled after job 2 failed - should be scheduled
    # # Job 5 was executed in the past (see below) - should not be scheduled
    job_manager = MockedCeleryJobManager(
        jobs=[
            _MockedJob(request_id=ir_1.id, num_iters_running=4),
            _MockedJob(request_id=ir_2.id, num_iters_running=1, fail=True),
            _MockedJob(request_id=ir_3.id, num_iters_running=3),
            _MockedJob(request_id=ir_4.id, num_iters_running=3),
            _MockedJob(request_id=ir_5.id, num_iters_running=2),
        ]
    )

    inference_queue = InferenceQueue()
    for i in [ir_1, ir_2, ir_3, ir_4, ir_5]:
        inference_queue.put(i)

    status_map = StatusMap()
    # Job 5 was executed in the past
    jobs_db = MockedJobsDB({ir_5.hash}, dict())
    runner = Runner(
        check_interrupted=PredefinedNumIters(20),
        celery=job_manager,
        device_manager=device_manager,
        remote_job_params=remote_job_params,
        jobs_db=jobs_db,
        redis_data=(inference_queue, status_map, ClassifierDeps()),
        no_sleep=True,
    )
    runner.run()

    # GPU should have been released
    assert device_manager.any_gpu_free()

    # All statuses should have been forgotten
    job_manager.check_forget_called()

    # Three distinct jobs
    # # 1 and 3 are identical (1 job)
    # # 2 and 4 are identical but 2 fails before 4 starts execution (2 jobs)
    # # 5 was already executed in the past (0 jobs)
    assert job_manager.get_started_job_ids() == {ir_1.id, ir_2.id, ir_4.id}

    # Check DB
    assert jobs_db.get_stored_inference_job_hashes() == {
        ir_1.hash,
        ir_2.hash,
        ir_4.hash,
    }

    # Check status
    assert status_map[ir_1.id].status == Status.FINISHED
    assert status_map[ir_2.id].status == Status.FAILED
    assert status_map[ir_3.id].status == Status.FINISHED
    assert status_map[ir_4.id].status == Status.FINISHED
    assert status_map[ir_5.id].status == Status.FINISHED


def test_classifier_jobs():
    device_manager = DeviceManager(gpu_ids_string="0,1", max_cpu_jobs=1)

    # Construct dependencies
    ir_1 = generate_inference_request_mock()
    ir_2 = generate_inference_request_mock()
    ir_3 = generate_inference_request_mock()
    ir_4 = generate_inference_request_mock()

    # Construct jobs
    cr_1 = generate_classifier_request_mock(deps=[ir_1.id, ir_3.id])
    cr_2 = generate_classifier_request_mock(
        deps=[ir_1.id, ir_3.id],
        hash_=cr_1.hash,
        parent_hash=cr_1.hash_without_closing_label_changes,
    )
    cr_3 = generate_classifier_request_mock(deps=[ir_2.id, ir_3.id])
    cr_4 = generate_classifier_request_mock(deps=[ir_3.id, ir_4.id])
    cr_5 = generate_classifier_request_mock(deps=[ir_1.id, ir_4.id])

    # Example result returned for all classifier jobs, is not checked
    example_result = NearestNeighborResult(
        test_labels=[0, 0],
        test_indices_within_readers=[0, 1],
        test_reader_indices=[0, 0],
        train_labels=[1, 0],
        train_indices_within_readers=[0, 1],
        train_reader_indices=[0, 0],
    ).json()

    # Timeline for classifier jobs:
    # # Job 1 runs normally - should be scheduled
    # # Job 2 is identical to job 1 - should not be scheduled
    # # Job 3 dependency failure (inference job 2) - should not be scheduled
    # # Job 4 was executed in the past (see below) - should not be scheduled
    # # Job 5 fails - should be scheduled (fails afterwards)
    job_manager = MockedCeleryJobManager(
        jobs=[
            _MockedJob(request_id=ir_1.id, num_iters_running=1),
            _MockedJob(request_id=ir_2.id, num_iters_running=1, fail=True),
            _MockedJob(request_id=ir_3.id, num_iters_running=1),
            _MockedJob(request_id=ir_4.id, num_iters_running=1),
            _MockedJob(
                request_id=cr_1.id,
                num_iters_running=1,
                result={
                    cr_1.hash_without_closing_label_changes: example_result,
                    cr_1.hash: example_result,
                },
            ),
            _MockedJob(
                request_id=cr_2.id,
                num_iters_running=1,
                result={
                    cr_2.hash_without_closing_label_changes: example_result,
                    cr_2.hash: example_result,
                },
            ),
            _MockedJob(
                request_id=cr_3.id,
                num_iters_running=1,
                result={
                    cr_3.hash_without_closing_label_changes: example_result,
                    cr_3.hash: example_result,
                },
            ),
            _MockedJob(
                request_id=cr_4.id,
                num_iters_running=1,
                result={
                    cr_4.hash_without_closing_label_changes: example_result,
                    cr_4.hash: example_result,
                },
            ),
            _MockedJob(
                request_id=cr_5.id,
                num_iters_running=2,
                fail=True,
                result={
                    cr_5.hash_without_closing_label_changes: example_result,
                    cr_5.hash: example_result,
                },
            ),
        ]
    )

    # Set status as it would be set by FastAPI
    status_map = StatusMap()

    inference_queue = InferenceQueue()
    for i in [ir_1, ir_2, ir_3, ir_4]:
        status_map[i.id] = StatusResponse(status=Status.WAITING)
        inference_queue.put(i)

    classifier_deps = ClassifierDeps()
    for i in [cr_1, cr_2, cr_3, cr_4, cr_5]:
        status_map[i.id] = StatusResponse(status=Status.WAITING)
        classifier_deps.add_request(i)

    # Job 4 was executed in the past
    jobs_db = MockedJobsDB(set(), {cr_4.hash: 0.55})
    runner = Runner(
        check_interrupted=PredefinedNumIters(20),
        celery=job_manager,
        device_manager=device_manager,
        remote_job_params=remote_job_params,
        jobs_db=jobs_db,
        redis_data=(inference_queue, status_map, classifier_deps),
        no_sleep=True,
    )
    runner.run()

    # All devices should have been released
    assert device_manager.any_gpu_free() and device_manager.any_cpu_free()

    # All statuses should have been forgotten
    job_manager.check_forget_called()

    # Check Celery
    assert job_manager.get_started_job_ids() == {
        ir_1.id,
        ir_2.id,
        ir_3.id,
        ir_4.id,
        cr_1.id,
        cr_5.id,
    }

    # Check DB
    assert jobs_db.get_stored_classifier_job_hashes() == {
        cr_1.hash,
        cr_1.hash_without_closing_label_changes,
    }

    # Check status
    assert status_map[cr_1.id].status == Status.FINISHED
    assert status_map[cr_2.id].status == Status.FINISHED
    assert status_map[cr_3.id].status == Status.FAILED
    assert status_map[cr_4.id].status == Status.FINISHED
    assert status_map[cr_5.id].status == Status.FAILED

"""
The interface of optimizations will do the following:

1. Intercept the requests
2. Split the train set into chunks

"""
import math
import os
import time
from typing import List

from db_tools.postgres import JobsDB
from db_tools.query import get_nn_result, get_reader_size
from db_tools.redis import ClassifierDeps, InferenceQueue, StatusMap
from loguru import logger
from schemas import Status
from schemas.classifier import ClassifierWithParams
from schemas.requests.common import HyperbandRequest, slice_readers
from schemas.requests.reader import Slice, TFReaderConfig, VTABReaderConfig
from schemas.response import StatusResponse
from schemas.task.result import KnownResult

from ._base import Arm, ProgressResult


def composeResult(
    jobs_db: JobsDB,
    request: HyperbandRequest,
    classifier: ClassifierWithParams,
    batch_sizes: List[int],
) -> List[KnownResult]:

    num_gpus = len(os.environ["SHIFT_DEVICES"].split(","))
    chunk_size = request.chunk_size
    models = request.models
    train_readers = request.train
    test_readers = request.test
    classifier = request.classifier
    
    test_sizes = [
        jobs_db.get_reader_size(reader.reader.invariant_json) for reader in test_readers
    ]
    final_results = []
    sizes = [
        jobs_db.get_reader_size(train_reader.reader.invariant_json)
        for train_reader in train_readers
    ]
    num_gpus = len(os.environ["SHIFT_DEVICES"].split(","))
    
    # here we calculates how models are reduced
    len_models = len(models)
    remaining_models = [len_models]
    pulls = []
    for i in range ( math.ceil(math.log2(len_models)) ):
        log_2_n = math.ceil(math.log2(len_models))
        pulls.append(math.floor(request.budget / (remaining_models[-1] * (log_2_n))))
        remaining_models.append(math.ceil(remaining_models[-1] / 2))
        
    pulls = [0] + pulls

    for id_model, each_model in enumerate(models):
        """
        First compose the test reader
        """
        sliced_test_readers = []
        for id_tr, test_reader in enumerate(test_readers):
            logger.info(f"test_sizes: {test_sizes}")
            test_size = int(test_sizes[id_tr])
            if (
                test_size > int(os.environ["SHIFT_SPLIT_SIZE_THRESHOLD"])
                and batch_sizes[id_model]
                < int(os.environ["SHIFT_SPLIT_BATCH_THRESHOLD"])
                and num_gpus > 1
            ):
                if os.environ["SHIFT_SPLIT_CHUNK_SIZE"]:
                    interval = int(os.environ["SHIFT_SPLIT_CHUNK_SIZE"]) - 1
                    splits = math.floor(test_size / interval)
                else:
                    splits = num_gpus
                    interval = math.ceil(test_size / splits)
                for i in range(splits):
                    t_test_reader = test_reader.reader.copy(deep=True)
                    start = i * (interval + 1)
                    stop = (
                        start + interval if start + interval < test_size else test_size
                    )
                    t_test_reader.slice = Slice(start=start, stop=stop)
                    sliced_test_readers.append(t_test_reader)
            else:
                sliced_test_readers.append(test_reader.reader)
        
        current_index = 0
        largest_training_result = None
        final_readers = None
        for pull in range(len(pulls)):
            logger.info("Sizes: {}".format(sizes))
            readers = slice_readers(sizes, train_readers, chunk_size, current_index, needed_num_pulls=pulls[:pull+2])
            logger.info("readers: {}".format(readers))
            result = jobs_db.get_known_result_by_params(
                classifier_type=classifier.value,
                model_json=each_model.invariant_json,
                train_reader_json=[reader.reader.invariant_json for reader in readers],
                test_reader_json=[
                    reader.invariant_json for reader in sliced_test_readers
                ],
            )
            if result is not None:
                largest_training_result = result
                final_readers = readers
                current_index += chunk_size
            else:
                logger.info(
                    "Result is none for Current Index: {}, sizes: {}, chunk_size: {}".format(
                        current_index, sizes, chunk_size
                    )
                )
                logger.info("model: {}".format(each_model.invariant_json))
                logger.info(
                    "train: {}".format(
                        [reader.reader.invariant_json for reader in readers]
                    )
                )
                logger.info(
                    "test: {}".format(
                        [reader.invariant_json for reader in sliced_test_readers]
                    )
                )
                break
        if final_readers is None:
            logger.info(
                "Current Index: {}, sizes: {}, chunk_size: {}".format(
                    current_index, sizes, chunk_size
                )
            )
        largest_training_result.start = final_readers[0].reader.slice.start
        largest_training_result.stop = final_readers[-1].reader.slice.stop
        final_results.append(largest_training_result)
    return final_results


class ShiftArm(Arm):
    def __init__(
        self,
        hyperband_request: HyperbandRequest,
        classifier: ClassifierWithParams,
        model_idx: int,
        initial_error: int,
        jobs_db: JobsDB,
        iq: InferenceQueue,
        sm: StatusMap,
        cd: ClassifierDeps,
        sizes: List[int],
    ) -> None:

        super().__init__()
        self._initial_error = initial_error
        self.request = hyperband_request
        # there should be only one model in each arm
        model = self.request.models[model_idx]
        self.name = model.invariant_json
        self.jobs_db = jobs_db
        self.classifier = classifier
        self.sm = sm
        self.cd = cd
        self.iq = iq
        self.model_idx = model_idx
        if isinstance(hyperband_request.train[0].reader, TFReaderConfig) or isinstance(
            hyperband_request.train[0].reader, VTABReaderConfig
        ):
            self._train = hyperband_request.train
        else:
            raise NotImplementedError("Only TFReaderConfig/VTAB is supported")
        # get the batch size of the model
        batch_size = self.jobs_db.get_batch_size(model)
        if batch_size is None:
            raise ValueError("Cannot find the batch size of the model!")
        self.batch_sizes = [batch_size]
        # get the size
        self.total_size = sum(sizes)
        self.sizes = sizes
        self.sh_finished = False
        self._increment = hyperband_request.chunk_size
        self._index = 0

    @property
    def initial_error(self) -> ProgressResult:
        return ProgressResult(0, self._initial_error)

    def can_progress(self) -> bool:
        return self._index <= self.total_size

    def progress(self, needed_pulls) -> ProgressResult:
        # generate inference requests
        # determine slice
        num_processed = (
            self._index + self._increment
            if self._index + self._increment <= self.total_size
            else self.total_size
        )
        logger.info("Generating requests, needed_pulls: {}".format(needed_pulls))
        request = self.request.generate_requests(
            self.model_idx,
            self.classifier,
            self._index,
            self._increment,
            self.sizes,
            needed_pulls,
        )
        logger.info(request)
        # now we try to generate the "smaller" inference and classifier requests
        inference_requests, classifier_requests = request.generate_requests(
            self.batch_sizes
        )
        logger.info(classifier_requests)
        for ir in inference_requests:
            # ORDER IMPORTANT!
            # Update status of that job
            self.sm[ir.id] = StatusResponse(status=Status.WAITING)
            # Add job to queue
            self.iq.put(ir)
        cr = classifier_requests[0]
        # ORDER IMPORTANT!
        # Update status of the job
        self.sm[cr.id] = StatusResponse(status=Status.WAITING)
        # Add job
        self.cd.add_request(cr)
        # now try to get the results...
        finished = False
        while not finished:
            sleep_time = 10
            logger.debug("Waiting {:.2f}s for the result...".format(sleep_time))
            time.sleep(sleep_time)
            result = None
            result = get_nn_result(cr.hash, self.classifier.name)
            if result is not None:
                err = int(result)
                finished = True
        if self._increment * sum(needed_pulls) <= self.total_size:
            self._index = self._increment * sum(needed_pulls)
        else:
            self._index = self.total_size
        return ProgressResult(num_processed, err)

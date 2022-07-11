import logging
from math import ceil, log
from typing import List

from db_tools.postgres import JobsDB
from db_tools.redis import ClassifierDeps, HyperbandQueue, InferenceQueue, StatusMap
from fastapi import status
from fastapi.responses import JSONResponse
from loguru import logger
from optimizations.interface import composeResult
from schemas import Status
from schemas.classifier import ClassifierWithParams
from schemas.requests.common import HyperbandRequest, Request
from schemas.response import BatchSizeError409, StatusResponse
from schemas.task.result import KnownResult

from .simplify_handler import find_reader_from_db


def process_hyperband(
    request: HyperbandRequest,
    classifiers: ClassifierWithParams,
    jobs_db: JobsDB,
    hq: HyperbandQueue,
    dry=False,
    batch_sizes: List[int] = [],
):
    classifier = classifiers[0]
    total_size = 0
    total_size = sum(
        [
            jobs_db.get_reader_size(train.reader.invariant_json)
            for train in request.train
        ]
    )
    num_jobs_per_model = ceil(total_size / request.chunk_size)

    if request.budget is None:
        # by default we set the budget to be N_jobs/log_2(N_models)
        request.budget = ceil(num_jobs_per_model / log(len(request.models), 2))
    eager_results = []
    request.classifier = classifier
    result = jobs_db.check_hyperband_job(request.hash)
    if not dry and result is None:
        hq.put(request)
    remaining_jobs = 9999
    if result is not None:
        remaining_jobs = 0
        eager_results = composeResult(
            jobs_db=jobs_db,
            request=request,
            classifier=classifier,
            batch_sizes=batch_sizes,
        )
    return eager_results, remaining_jobs


def process_ordinary_request(
    request: Request,
    batch_sizes,
    jobs_db: JobsDB,
    iq: InferenceQueue,
    sm: StatusMap,
    cd: ClassifierDeps,
):
    dry_run = request.dry
    inference_requests, classifier_requests = request.generate_requests(batch_sizes)
    eager_results: List[KnownResult] = []
    remaining_classification_jobs = 0
    remaining_inference_jobs = 0
    known_inference_hashes = jobs_db.get_successful_inference_request_hashes()
    # We first compose the known results of the query, based on the classifier hash. The classifier hash has already uniquely identified the classifier (incl. train and test)
    for each_classifier_req in classifier_requests[:]:
        results = jobs_db.get_known_result(each_classifier_req.hash)
        if len(results) > 1:
            raise ValueError(
                "More than one results fetched from the database, but we only expect one."
            )
        if len(results) > 0:
            result = results[0]
            if request.benchmark:
                for each_test in each_classifier_req.test:
                    for each_inference in inference_requests:
                        if each_test.inference_request_hash == each_inference.hash:
                            if each_inference.model.dict() == result.json_model:
                                result.test_reader_name = find_reader_from_db(
                                    jobs_db, each_inference.reader.invariant_json
                                )
            eager_results.append(result)
        if len(results) > 0:
            classifier_requests.remove(each_classifier_req)

    # Then we calculate how many tasks are left to be done.
    # we start from classification tasks
    remaining_classification_jobs = len(classifier_requests)

    # We then compute the number of remaining inference tasks.
    # N.B. we don't remove it from the list, as the classification task may need the id as in dependencies.
    # The scheduler will ensure the inference tasks are not executed redundantly.
    remaining_inference_jobs = len(inference_requests)
    for each_inference in inference_requests:
        if each_inference.hash in known_inference_hashes:
            remaining_inference_jobs -= 1
    if not dry_run:
        for ir in inference_requests:
            logger.info("Processing ir {}".format(ir))
            # ORDER IMPORTANT!
            # Update status of that job
            sm[ir.id] = StatusResponse(status=Status.WAITING)
            # Add job to queue
            iq.put(ir)
        for cr in classifier_requests:
            # ORDER IMPORTANT!
            # Update status of the job
            sm[cr.id] = StatusResponse(status=Status.WAITING)
            # Add job
            cd.add_request(cr)
    return eager_results, remaining_inference_jobs + remaining_classification_jobs


def query_result(
    request: Request,
    jobs_db: JobsDB,
    iq: InferenceQueue,
    sm: StatusMap,
    cd: ClassifierDeps,
    hq: HyperbandQueue,
):
    batch_sizes = []
    for model in request.full_model_configs:
        batch_size = jobs_db.get_batch_size(model)
        if batch_size is None:
            logger.info("batch_size for {} is not defined.".format(model))
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content=BatchSizeError409(
                    status=f"Batch size for model {model} has not been specified before, please specify it first by registering the model"
                ).dict(),
            )
        batch_sizes.append(batch_size)

    if request.chunk_size:
        hyperband_request = request.generate_hyperband_requests()
        return process_hyperband(
            hyperband_request,
            request.classifiers,
            jobs_db,
            hq,
            request.dry,
            batch_sizes,
        )
    else:
        return process_ordinary_request(request, batch_sizes, jobs_db, iq, sm, cd)

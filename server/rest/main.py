import logging
import os
import signal

from db_tools.postgres import JobsDB
from db_tools.queues.finetune import FinetuneQueue
from db_tools.queues.hyperband import HyperbandQueue
from db_tools.redis import ClassifierDeps, InferenceQueue, RedisData, StatusMap
from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseSettings
from schemas import Status
from schemas.dataset import DatasetRegistrationRequest
from schemas.requests.common import Request
from schemas.requests.finetune import FinetuneRequest
from schemas.requests.model_request import (
    ImageModelInfoRequest,
    ImageModelRegistrationRequest,
    QueryModelByTagsRequest,
    ReadersUsedWithAModelRequest,
    TextModelInfoRequest,
    TextModelRegistrationRequest,
)
from schemas.requests.reader import (
    GetReaderSizeByJSONRequest,
    ModelsUsedWithAReaderRequest,
    QueryReaderByNameRequest,
    SimplifyReaderByJSONRequest,
)
from schemas.requests.task2vec import Task2VecMultipleReaderRequest
from schemas.response import (
    BatchSizeError409,
    Error404,
    JobSubmittedResponse,
    MatchingImageModelsResponse,
    MatchingTextModelsResponse,
    ModelsUsedWithAReaderResponse,
    QueryResultResponse,
    ReadersUsedWithAModelResponse,
    StatusResponse,
    Task2VecResponse,
)

from rest.handlers.finetune_handler import finetune_handler
from rest.handlers.result_handler import query_result
from rest.handlers.simplify_handler import simplify_readers_json_to_name
from rest.handlers.task2vec_handler import task2vec

logging.basicConfig(level=logging.INFO)


class Settings(BaseSettings):
    redis_host: str
    redis_port: int

    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_database: str

    class Config:
        env_file = "py-dev.env"
        env_file_encoding = "utf-8"
        # secrets_dir = "/run/secrets"


class Secrets(BaseSettings):
    postgres_password: str

    class Config:
        env_file = "py-dev.env"
        env_file_encoding = "utf-8"
        # secrets_dir = "/run/secrets"


settings_ = Settings()
secrets_ = Secrets()

# Each request gets a separate Redis instance for thread safety (perhaps that is not needed), all requests share the same pool
redis_data = RedisData(host=settings_.redis_host, port=settings_.redis_port)

jobs_db = JobsDB(
    dbname=settings_.postgres_database,
    user=settings_.postgres_user,
    password=secrets_.postgres_password,
    host=settings_.postgres_host,
    port=settings_.postgres_port,
)

app = FastAPI(title="SHiFT", description="Search Engine for Transfer Learning")

@app.get(
    "/progress/{job_id}",
    response_model=StatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": Error404},
        status.HTTP_200_OK: {"model": StatusResponse},
    },
)
def progress(job_id: str, status_map: StatusMap = Depends(redis_data.get_status_map)):
    if job_id not in status_map:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=Error404().dict(),
        )

    result = status_map[job_id]
    if result.successful:
        return JSONResponse(status_code=status.HTTP_200_OK, content=result.dict())
    if result.failed:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=result.dict()
        )
    return result


@app.post(
    "/query",
    response_model=QueryResultResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def query(
    request: Request,
    iq: InferenceQueue = Depends(redis_data.get_inference_queue),
    sm: StatusMap = Depends(redis_data.get_status_map),
    cd: ClassifierDeps = Depends(redis_data.get_classifier_deps),
    hq: HyperbandQueue = Depends(redis_data.get_hyperband_queue),
):
    known_results, num_remaining = query_result(request, jobs_db, iq, sm, cd, hq)
    return QueryResultResponse(
        known_results=known_results,
        num_remaining_tasks=num_remaining,
    )


@app.post(
    "/",
    response_model=JobSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={status.HTTP_409_CONFLICT: {"model": BatchSizeError409}},
)
def root(
    request: Request,
    iq: InferenceQueue = Depends(redis_data.get_inference_queue),
    sm: StatusMap = Depends(redis_data.get_status_map),
    cd: ClassifierDeps = Depends(redis_data.get_classifier_deps),
):
    batch_sizes = []
    for model in request.full_model_configs:
        batch_size = jobs_db.get_batch_size(model)
        if batch_size is None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content=BatchSizeError409(
                    status=f"Batch size for model {model} was not specified beforehand, specify it first by registering the model"
                ).dict(),
            )
        batch_sizes.append(batch_size)

    inference_requests, classifier_requests = request.generate_requests(batch_sizes)
    for ir in inference_requests:
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

    return JobSubmittedResponse(
        description="Please visit the following URL(s) to track progress:",
        inference_request_paths=[f"progress/{i.id}" for i in inference_requests],
        classifier_request_paths=[f"progress/{c.id}" for c in classifier_requests],
        inference_requests=inference_requests,
        classifier_requests=classifier_requests,
    )


@app.post("/register_image_model/")
def register_image_model(request: ImageModelRegistrationRequest):
    return jobs_db.register_image_model(request)


@app.get("/readers")
def get_all_readers():
    return jobs_db.get_all_registered_readers()


@app.post("/query_reader/")
def query_reader(request: QueryReaderByNameRequest):
    return jobs_db.get_registered_readers_with_name(request.name)


@app.post("/register_reader/")
def register_reader(request: DatasetRegistrationRequest):
    return jobs_db.register_reader(request)


@app.post("/register_text_model/")
def register_text_model(request: TextModelRegistrationRequest):
    return jobs_db.register_text_model(request)


@app.post("/text_models/", response_model=MatchingTextModelsResponse)
def text_models(request: TextModelInfoRequest) -> MatchingTextModelsResponse:
    return jobs_db.get_text_models(request)


@app.post("/image_models/", response_model=MatchingImageModelsResponse)
def image_models(request: ImageModelInfoRequest) -> MatchingImageModelsResponse:
    return jobs_db.get_image_models(request)


@app.post("/query_model/", response_model=ReadersUsedWithAModelResponse)
def query_model(request: ReadersUsedWithAModelRequest) -> ReadersUsedWithAModelResponse:
    return jobs_db.get_readers_used_with_a_model(request)


@app.post("/query_model_by_tags")
def query_model_by_tags(request: QueryModelByTagsRequest):
    return jobs_db.get_model_by_tags(request)


@app.post("/query_reader/", response_model=ModelsUsedWithAReaderResponse)
def query_reader(
    request: ModelsUsedWithAReaderRequest,
) -> ModelsUsedWithAReaderResponse:
    return jobs_db.get_models_used_with_a_reader(request)


@app.get("/nn_result")
def query_nn_result(job_hash: str):
    result = jobs_db.get_nn_result(job_hash)
    if result is not None:
        return result.raw_error
    else:
        return None


@app.get("/lc_result")
def query_lc_result(job_hash: str):
    result = jobs_db.get_linear_result(job_hash)
    if result is not None:
        return result.raw_error
    else:
        return None


@app.post("/query_reader_by_name")
def query_reader_by_name(request: QueryReaderByNameRequest):
    return jobs_db.get_dataset(request.name)


@app.post("/query_reader_size_by_json")
def query_reader_size_by_json(request: GetReaderSizeByJSONRequest):
    return jobs_db.get_reader_size(request.json_reader)


@app.post("/simplify_reader/")
def simplify_reader(request: SimplifyReaderByJSONRequest):
    return simplify_readers_json_to_name(request, jobs_db)


@app.post("/task2vec")
def task2vec_request(
    request: Task2VecMultipleReaderRequest,
    tq: InferenceQueue = Depends(redis_data.get_task2vec_queue),
    sm: StatusMap = Depends(redis_data.get_status_map),
):
    results, remaining = task2vec(request, jobs_db, tq, sm)
    return Task2VecResponse(distances=results, num_remaining_tasks=remaining)


@app.post("/finetune")
def finetune_request(
    request: FinetuneRequest,
    fq: FinetuneQueue = Depends(redis_data.get_finetune_queue),
):
    if request.batch_size is None:
        request.batch_size = jobs_db.get_batch_size(request.model)
    finetune_handler(request, jobs_db=jobs_db, fq=fq)


@app.get("/purge")
def purge_request():
    jobs_db.purge()
    # redis_data.flush()
    PID = 886580
    if PID != 0:
        os.kill(PID, signal.SIGHUP)

if __name__ == "__main__":
    logging.info("Populating...")
    jobs_db.populate_model_databases()

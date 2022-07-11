import logging
import os
import signal

from celery import Celery
from celery.result import AsyncResult
from db_tools.postgres import JobsDB
from db_tools.redis import RedisData
from pydantic import BaseSettings

from scheduler import CeleryJobManager, DeviceManager, RemoteJobParams, Runner


class ConcreteCeleryJobManager(CeleryJobManager):
    def __init__(self, redis_broker: str, redis_backend: str):
        self._celery_app = Celery("", broker=redis_broker, backend=redis_backend)

    def start_job(self, job_name: str, args: tuple, queue: str) -> str:
        async_result = self._celery_app.send_task(name=job_name, args=args, queue=queue)
        return async_result.id

    def get_status(self, celery_id: str) -> AsyncResult:
        return AsyncResult(celery_id)


class Settings(BaseSettings):
    redis_broker: str
    redis_backend: str

    redis_host: str
    redis_port: int

    worker_general_inference_job_name: str
    worker_general_task2vec_job_name: str
    worker_general_classifier_job_name: str
    worker_general_finetune_job_name: str
    worker_tf1_inference_job_name: str

    worker_general_queue_name: str
    worker_tf1_queue_name: str

    max_cpu_jobs: int
    shift_devices: str = ""

    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_database: str

    class Config:
        env_file = "py-dev.env"
        env_file_encoding = "utf-8"


class Secrets(BaseSettings):
    postgres_password: str

    class Config:
        env_file = "py-dev.env"
        env_file_encoding = "utf-8"
        # secrets_dir = "/run/secrets"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger(__name__)
    _logger.info("My PID is {}".format(os.getpid()))
    runner_interrupted = False
    runner_reload = False

    def check_interrupted():
        global runner_interrupted
        return runner_interrupted

    def handler(*_):
        global runner_interrupted
        runner_interrupted = True
        _logger.debug("Scheduler was interrupted with SIGTERM signal")

    def check_reload():
        global runner_reload
        return runner_reload

    def reload_handler(*_):
        global runner_reload
        runner_reload = True
        _logger.debug("Scheduler was reloaded with SIGTERM signal")
    
    def reset_reload(*_):
        global runner_reload
        runner_reload = False
        _logger.debug("Reload Status Reset")

    settings_ = Settings()
    secrets_ = Secrets()
    redis_data = RedisData(host=settings_.redis_host, port=settings_.redis_port)
    runner = Runner(
        check_interrupted=check_interrupted,
        check_reload=check_reload,
        reset_reload=reset_reload,
        celery=ConcreteCeleryJobManager(
            redis_broker=settings_.redis_broker, redis_backend=settings_.redis_backend
        ),
        device_manager=DeviceManager(
            gpu_ids_string=settings_.shift_devices, max_cpu_jobs=settings_.max_cpu_jobs
        ),
        remote_job_params=RemoteJobParams(
            general_inference_job_name=settings_.worker_general_inference_job_name,
            general_classifier_job_name=settings_.worker_general_classifier_job_name,
            general_finetune_job_name=settings_.worker_general_finetune_job_name,
            general_task2vec_job_name=settings_.worker_general_task2vec_job_name,
            general_queue_name=settings_.worker_general_queue_name,
            tf_1_inference_job_name=settings_.worker_tf1_inference_job_name,
            tf_1_queue_name=settings_.worker_tf1_queue_name,
        ),
        jobs_db=JobsDB(
            dbname=settings_.postgres_database,
            user=settings_.postgres_user,
            password=secrets_.postgres_password,
            host=settings_.postgres_host,
            port=settings_.postgres_port,
        ),
        redis_data=redis_data.get_all(),
    )
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGHUP, reload_handler)
    runner.run()

import os
from loguru import logger
from pydantic import BaseSettings
from db_tools.postgres import JobsDB

class Settings(BaseSettings):
    redis_host: str
    redis_port: int

    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_database: str

class Secrets(BaseSettings):
    postgres_password: str



class ShiftExtension():
    def __init__(self, name):
        self.name = name
        settings_ = Settings()
        secrets_ = Secrets()
        base_path = os.environ.get("USERSPACE_LOCATION", None)
        if base_path is None:
            raise Exception(
                "USERSPACE_LOCATION environment variable is not set")
        self.jobs_db = JobsDB(
            dbname=settings_.postgres_database,
            user=settings_.postgres_user,
            password=secrets_.postgres_password,
            host=settings_.postgres_host,
            port=settings_.postgres_port,
        )
        self.base_src_path = os.path.join(
            base_path, "extensions", "src", self.name)
        self.base_data_path = os.path.join(self.base_src_path, "data")
        logger.info(
            f"Extension {name} initialized: base_data_path={self.base_data_path}, base_src_path={self.base_src_path}")
        os.makedirs(self.base_data_path, exist_ok=True)

    def __call__(self, models, task):
        raise NotImplementedError

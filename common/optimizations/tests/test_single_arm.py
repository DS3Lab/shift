import unittest
from collections import OrderedDict

from db_tools.postgres import JobsDB
from db_tools.redis import RedisData
from loguru import logger
from optimizations._base import Observer, ProgressResult
from optimizations.interface import ShiftArm
from optimizations.successive_halving import SuccessiveHalving
from pydantic import BaseSettings
from schemas import generate_id
from schemas.classifier import ClassifierWithParams
from schemas.models.text_model import TFFullTextModelConfig
from schemas.requests.common import HyperbandRequest, MutableReader
from schemas.requests.reader import TFReaderConfig


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

redis_data = RedisData(host=settings_.redis_host, port=settings_.redis_port)

jobs_db = JobsDB(
    dbname=settings_.postgres_database,
    user=settings_.postgres_user,
    password=secrets_.postgres_password,
    host=settings_.postgres_host,
    port=settings_.postgres_port,
)

classifier = ClassifierWithParams(
    name="Euclidean NN",
)

request = HyperbandRequest(
    id=generate_id(),
    train=MutableReader(
        reader=TFReaderConfig(
            tf_dataset_name="glue/sst2:1.0.0",
            split="train",
            embed_feature_path=["sentence"],
            label_feature_path=["label"],
        )
    ),
    test=MutableReader(
        reader=TFReaderConfig(
            tf_dataset_name="glue/sst2:1.0.0",
            split="validation",
            embed_feature_path=["sentence"],
            label_feature_path=["label"],
        )
    ),
    models=[
        TFFullTextModelConfig(
            tf_text_model_url="https://tfhub.dev/google/universal-sentence-encoder/4",
        )
    ],
    chunk_size=4096,
)


class TestSuccessiveHalving(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arms = OrderedDict()
        status_map = redis_data.get_status_map()
        inference_queue = redis_data.get_inference_queue()
        classifier_deps = redis_data.get_classifier_deps()
        for idx, model in enumerate(request.models):
            cls.arms[str(model)] = ShiftArm(
                request,
                classifier,
                idx,
                initial_error=99999,
                jobs_db=jobs_db,
                iq=inference_queue,
                sm=status_map,
                cd=classifier_deps,
            )

    def test_successive_halving_with_doubling(self):
        class MyObserver(Observer):
            def on_update(self, name: str, progress_result: ProgressResult):
                logger.debug(f"{name} {progress_result}")

        algorithm = SuccessiveHalving(self.arms, MyObserver())
        algorithm.successive_halving_with_doubling(eta=2, snoopy=False)


if __name__ == "__main__":
    unittest.main()

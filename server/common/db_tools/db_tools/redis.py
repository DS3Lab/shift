from typing import Tuple

import redis
from db_tools.queues.finetune import FinetuneQueue
from redis import BlockingConnectionPool

from .queues.base import __loader__
from .queues.classifier_deps import ClassifierDeps
from .queues.hyperband import HyperbandQueue
from .queues.inference import InferenceQueue
from .queues.status_map import StatusMap
from .queues.task2vec import Task2vecQueue

__all__ = [
    "StatusMap",
    "InferenceQueue",
    "ClassifierDeps",
    "RedisData",
    "Task2vecQueue",
    "HyperbandQueue",
]


class RedisData:
    """Provides a simple way of instantiating Redis data structures.

    Args:
        host (str): Redis host.
        port (str): Redis port.
    """

    def __init__(self, host: str, port: int):
        self._pool = BlockingConnectionPool(host=host, port=port)

    def get_status_map(self) -> StatusMap:
        return StatusMap(self._pool)

    def get_task2vec_queue(self) -> Task2vecQueue:
        return Task2vecQueue(self._pool)

    def get_inference_queue(self) -> InferenceQueue:
        return InferenceQueue(self._pool)

    def get_classifier_deps(self) -> ClassifierDeps:
        return ClassifierDeps(self._pool)

    def get_hyperband_queue(self) -> HyperbandQueue:
        return HyperbandQueue(self._pool)

    def get_finetune_queue(self) -> FinetuneQueue:
        return FinetuneQueue(self._pool)

    def flush(self):
        client = redis.Redis(connection_pool=self._pool)
        client.flushdb(asynchronous=False)

    def get_all(
        self,
    ) -> Tuple[Task2vecQueue, InferenceQueue, StatusMap, ClassifierDeps]:
        """Returns all Redis data structures.

        Returns:
            Tuple[InferenceQueue, StatusMap, ClassifierDeps]: All data structures.
        """
        return (
            self.get_task2vec_queue(),
            self.get_inference_queue(),
            self.get_status_map(),
            self.get_classifier_deps(),
            self.get_hyperband_queue(),
            self.get_finetune_queue(),
        )

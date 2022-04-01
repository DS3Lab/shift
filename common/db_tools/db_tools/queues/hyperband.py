from collections import deque
from typing import Any, Optional

from pottery import RedisDeque, Redlock
from redis import Redis
from schemas.requests.common import HyperbandRequest

from .base import _EmptyContextManager, _logger


class HyperbandQueue:
    def __init__(self, pool: Optional[BlockingIOError]) -> None:
        self._deque: Any
        self._lock: Any
        if pool is not None:
            key = "shift:hyperband_queue"
            redis_instance = Redis(connection_pool=pool)
            self._deque = RedisDeque(redis=redis_instance, key=key)
            self._lock = Redlock(key=key + "-lock", masters={redis_instance})
            self._serialize = lambda x: x.json()
            self._deserialize = lambda x: HyperbandRequest.parse_raw(str(x))
        else:
            self._deque, self._lock = deque(), _EmptyContextManager()
            self._serialize, self._deserialize = lambda x: x, lambda x: x

    def empty(self) -> bool:
        """Checks whether the queue is empty.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        with self._lock:
            return len(self._deque) == 0

    def put(self, value: HyperbandRequest):
        """Puts the request into queue.

        Args:
            value (Hyperband Request): Request to put into queue.
        """
        with self._lock:
            self._deque.append(self._serialize(value))
        _logger.info(
            "Hyperband request %s - inserted into hyperband queue",
            value.id,
        )

    def get(self) -> HyperbandRequest:
        """Returns the first element from the queue. Before this call, it should be checked with another method that the queue is not empty. Note that it is not safe to call this method from multiple threads/processes as the element could have already been returned to another thread/process.

        Returns:
            HyperbandStatus: First element from the queue.
        """
        with self._lock:
            return_value = self._deserialize(self._deque.popleft())
        _logger.info(
            "Hyperband request %s - popped from the hyperband queue",
            return_value.id,
        )
        return return_value

    def __len__(self) -> int:
        return len(self._deque)

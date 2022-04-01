from typing import Any, Optional

from pottery import RedisDict, Redlock
from redis import BlockingConnectionPool, Redis
from schemas import ID
from schemas.response import StatusResponse

from .base import _EmptyContextManager


class StatusMap:
    """Stores status of jobs. If Redis is used, each thread/process should use its own
    instance. If Redis is not used, a single instance should be used by one
    thread/process.

    Args:
        pool (BlockingConnectionPool, optional): Connection pool needed to use Redis.
            The default value None means that Redis is not used. Instead, regular Python
            data structures are used.
    """

    def __init__(self, pool: Optional[BlockingConnectionPool] = None):
        self._dict: Any
        self._lock: Any
        if pool is not None:
            key = "shift:status_map_key"
            redis_instance = Redis(connection_pool=pool)
            self._dict = RedisDict(redis=redis_instance, key=key)
            self._lock = Redlock(key=key + "-lock", masters={redis_instance})
        else:
            self._dict, self._lock = {}, _EmptyContextManager()

    def __getitem__(self, job_id: ID) -> StatusResponse:
        """Returns the status for the job/request with the specified ID.

        Args:
            job_id (ID): Job/request ID.

        Returns:
            StatusResponse: Status of the requested job.
        """
        with self._lock:
            return StatusResponse.parse_raw(self._dict[job_id])

    def __setitem__(self, job_id: ID, status: StatusResponse):
        """Sets the status for the job/request with the specified ID.

        Args:
            job_id (ID): Job/request ID.
            status (StatusResponse): Status of the job.
        """
        with self._lock:
            self._dict[job_id] = status.json()

    def __contains__(self, job_id: ID) -> bool:
        """Checks whether the map contains status for a job/request.

        Args:
            job_id (ID): Job/request ID.

        Returns:
            bool: True if the status of the job is stored in the map, False otherwise.
        """
        with self._lock:
            return job_id in self._dict

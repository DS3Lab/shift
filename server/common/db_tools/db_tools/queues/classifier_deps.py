from collections import deque
from typing import Any, Optional, Sequence, Tuple

from pottery import RedisDeque, RedisDict, Redlock
from redis import BlockingConnectionPool, Redis
from schemas import ID
from schemas.requests.common import ClassifierRequest

from .base import _EmptyContextManager, _logger
from .status_map import StatusMap


class ClassifierDeps:
    """Stores classifier requests waiting for execution and tracks their dependencies.
    If Redis is used, each thread/process should use its own instance. If Redis is not used, a single instance should be used by one thread/process.

    Args:
        pool (BlockingConnectionPool, optional): Connection pool needed to use Redis.
        The default value None means that Redis is not used. Instead, regular Python data structures are used.
    """

    def __init__(self, pool: Optional[BlockingConnectionPool] = None):
        self._lock: Any
        self._waiting_ids: Any
        self._ready_ids: Any
        self._id_to_deps: Any
        self._id_to_json: Any
        if pool is not None:
            key = "shift:classifier_dependencies"
            redis_instance = Redis(connection_pool=pool)
            self._id_to_deps = RedisDict(redis=redis_instance, key=key)
            self._id_to_json = RedisDict(redis=redis_instance, key=key + "-json")
            self._waiting_ids = RedisDeque(redis=redis_instance, key=key + "-wait")
            self._ready_ids = RedisDeque(redis=redis_instance, key=key + "-ready")
            self._lock = Redlock(key=key + "-lock", masters={redis_instance}, auto_release_time=30)
            self._serialize = lambda x: x.json()
            self._deserialize = lambda x: ClassifierRequest.parse_raw(x)
        else:
            self._id_to_deps, self._id_to_json = {}, {}
            self._waiting_ids, self._ready_ids = deque(), deque()
            self._lock = _EmptyContextManager()
            self._serialize, self._deserialize = lambda x: x, lambda x: x

    def add_request(self, request: ClassifierRequest):
        """Stores a classifier request and starts to track its dependencies.

        Args:
            request (ClassifierRequest): Classifier request.
        """
        with self._lock:
            self._id_to_deps[request.id] = request.get_inference_request_ids()
            self._id_to_json[request.id] = self._serialize(request)
            self._waiting_ids.append(request.id)
        _logger.info("Classifier request %s - added to the classifier deps", request.id)

    def update_dependencies(self, status_map: StatusMap) -> Sequence[Tuple[ID, ID]]:
        """Updates dependencies of all classifier jobs to determine which jobs are ready for execution and which jobs cannot be executed, because one of their dependencies failed.

        Args:
            status_map (StatusMap): Statuses of all jobs used to check statuses of classifier jobs dependencies.

        Returns:
            Sequence[Tuple[ID, ID]]: Pairs (classifier job ID, dependency ID). Each pair means that a dependency failed and because of that the classifier job cannot be executed, so it should be marked as failed.
        """

        failed = []

        with self._lock:
            # Requests for which the dependencies failed
            failed_ids = []
            successful_ids = []

            for id_ in self._waiting_ids:
                all_deps_fulfilled = True
                any_dep_failed = False
                for dependency_key in self._id_to_deps[id_]:
                    status = status_map[dependency_key]

                    # FAIL - can break, enough that one instance of failed dependency is reported
                    if status.failed:
                        failed.append((id_, dependency_key))
                        any_dep_failed = True
                        all_deps_fulfilled = False
                        break

                    # NOT SUCCESSFUL - should not break, because some other dependency could have failed and it must be reached
                    if not status.successful:
                        all_deps_fulfilled = False

                # Cannot happen that all dependencies are fulfilled and some dependency failed, so id_ cannot be present twice (both in successful_ids and failed_ids)
                if all_deps_fulfilled:
                    successful_ids.append(id_)
                    _logger.info("Classifier job %s - dependencies are fulfilled", id_)

                elif any_dep_failed:
                    failed_ids.append(id_)
                    _logger.info("Classifier job %s - one of dependencies failed", id_)

                else:
                    _logger.debug(
                        "Classifier job %s - dependencies not ready yet, none failed",
                        id_,
                    )

            for id_ in successful_ids:
                self._ready_ids.append(id_)
                self._waiting_ids.remove(id_)

            for id_ in failed_ids:
                self._waiting_ids.remove(id_)
                self._id_to_deps.pop(id_)
                self._id_to_json.pop(id_)
        _logger.info("Lock status: %s", self._lock.locked())
        return failed

    def any_request_ready(self) -> bool:
        """Checks whether any classifier job/request is ready for execution, because
        all of its dependencies are ready.

        Returns:
            bool: True if any request is ready for execution, False otherwise.
        """
        with self._lock:
            return len(self._ready_ids) > 0

    def get_ready_request(self) -> ClassifierRequest:
        """Returns a request that has all of its dependencies ready. Before this call, it should be checked with another method that such request actually exists.
        Note that it is not safe to call this method from multiple threads/processes as the request could have already been returned to another thread/process.

        Returns:
            ClassifierRequest: Request ready for execution.
        """
        with self._lock:
            id_ = self._ready_ids.popleft()
            self._id_to_deps.pop(id_)
            return self._deserialize(self._id_to_json.pop(id_))

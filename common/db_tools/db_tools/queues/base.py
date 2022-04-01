import logging

_logger = logging.getLogger(__name__)


class _EmptyContextManager:
    """No-op context manager that replaces Redis locking when data is stored in Python
    data structures rather than in Redis."""

    def __enter__(self, *_):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

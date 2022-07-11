from celery import Celery
from celery.app.log import Logging

from ._config import settings

celery_app = Celery(
    "main", broker=settings.redis_broker, backend=settings.redis_backend
)
celery_app.conf.update(
    worker_task_log_format="[%(asctime)s] %(levelname)s:%(task_id)s:%(name)s:"
    "%(message)s",
    worker_redirect_stdouts_level="DEBUG",
)

logging = Logging(celery_app)
logging.setup(loglevel="INFO")

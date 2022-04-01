References
==========

This page contains links that were used during the development of the sh¡ft! system.
Links that are directly referenced in the code are here omitted.

General
-------
* `Project structure and use of BaseSettings <https://github.com/tiangolo/full-stack-fastapi-postgresql>`__.
* `API design <https://docs.microsoft.com/en-us/azure/architecture/patterns/async-request-reply>`__.
* `TF1 models can (for now) be run using general worker <https://www.tensorflow.org/hub/model_compatibility>`__.
* `Default Redis configuration does not work with Docker <https://github.com/docker-library/redis/issues/181>`__.
* `Which signals should containers respond to <https://docs.docker.com/compose/faq/#why-do-my-services-take-10-seconds-to-recreate-or-stop>`__. As can be seen in ``docker-compose.yml``, this needs to be changed for Celery.
* `Why not use async def with FastAPI <https://fastapi.tiangolo.com/async/>`__.
* `Setting environment variables from file <https://docs.docker.com/compose/env-file/>`__.

Celery
------
* `How to call Celery task by name <https://docs.celeryproject.org/en/stable/userguide/calling.html#example>`__.
* `How to get task result <https://docs.celeryproject.org/en/latest/faq.html#how-do-i-get-the-result-of-a-task-if-i-have-the-id-that-points-there>`__.


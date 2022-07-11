celery -A main worker -c 8 -Q shift:queue_1 --loglevel=INFO --max-tasks-per-child=1

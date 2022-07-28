# --- REMOVE .local.env IF IN DOCKER ---

export $(grep -v '^#' .env | xargs -d '\n') && export $(grep -v '^#' .local.env | xargs -d '\n') && cd worker_general && celery -A main worker -c 8 -Q shift:queue_1 --loglevel=INFO --max-tasks-per-child=1
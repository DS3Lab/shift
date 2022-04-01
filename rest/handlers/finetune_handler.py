from db_tools.postgres import JobsDB
from db_tools.redis import FinetuneQueue
from schemas import generate_id
from schemas.requests.finetune import FinetuneRequest


def finetune_handler(request: FinetuneRequest, jobs_db: JobsDB, fq: FinetuneQueue):
    # TODO: Type validations...
    # Fetch existing models
    if request.id is None:
        request.id = generate_id()
    fq.put(request)

import json

from db_tools.postgres import JobsDB
from schemas.requests.model_request import TaskAgnosticRequest


def clean_dict(dict_to_clean: dict):
    return {k: v for k, v in dict_to_clean.items() if v is not None}


from schemas.response import TaskAgnosticResponse


def get_task_agnostic_result(request: TaskAgnosticRequest, db: JobsDB):
    json_reader = json.loads(request.json_reader.invariant_json)
    json_reader = clean_dict(json_reader)
    results = db.get_task_agnostic_results(json.dumps(json_reader))
    return TaskAgnosticResponse(results=results)

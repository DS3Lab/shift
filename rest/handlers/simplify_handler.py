import json

from db_tools.postgres import JobsDB
from schemas.requests.reader import SimplifyReaderByJSONRequest


def clean_dict(dict_to_clean: dict):
    return {k: v for k, v in dict_to_clean.items() if v is not None}


def find_reader_from_db(db: JobsDB, json_reader: str):
    json_reader = json.loads(json_reader)
    json_reader = clean_dict(json_reader)
    reader_name = db.get_reader_by_json(json.dumps(json_reader))
    if reader_name is None:
        raise ValueError("Reader not found in database")
    return reader_name


def simplify_readers_json_to_name(request: SimplifyReaderByJSONRequest, db: JobsDB):
    return {
        "name": find_reader_from_db(db, request.json_reader.invariant_json),
    }

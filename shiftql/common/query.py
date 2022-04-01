from enum import Enum


class QueryTypes(str, Enum):
    RESTRICT_MODELS = "restrict_models"
    QUERY_MODELS = "query_models"
    SELECT_READERS = "select_readers"
    TASK2VEC_MATRIX = "task2vec_matrix"
    DECLARE = "declare"
    USE = "use"
    PRINT = "print"
    EXPLAIN = "explain"
    TASK_AGNOSTIC = "task_agnostic"


class ReturnedType(str, Enum):
    models = "MODELS"
    readers = "READERS"


class ReservedKeywords(str, Enum):
    TEXT_MODELS = "text_models"
    IMAGE_MODELS = "image_models"
    READERS = "readers"
    MODELS = "models"
    QUERY_TYPE = "query_type"
    RESPONSE = "response"
    REMAINING_TASKS = "remaining_tasks"

    @classmethod
    def contains(cls, key):
        return key in cls._value2member_map_

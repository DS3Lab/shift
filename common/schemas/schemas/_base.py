import hashlib
import secrets
from enum import Enum

READER_EMBED_FEATURE_NAME = "embed"
READER_LABEL_FEATURE_NAME = "label"


class _DefaultConfig:
    """Some extra configuration to ensure that errors are caught earlier. See:
    https://pydantic-docs.helpmanual.io/usage/model_config/"""

    extra = "forbid"
    validate_assignment = True
    validate_all = True


class Status(str, Enum):
    WAITING = "Waiting"
    STARTED = "Started"
    RUNNING = "Running"
    FAILED = "Failed"
    FINISHED = "Finished"


# Regex for checking whether the generated request id is valid
_hash_regex = r"^([a-z]|\d){64}$"
_id_regex = r"^([a-z]|\d){32}$"

Hash = str


def get_hash(string: str) -> Hash:
    return hashlib.sha256(string.encode()).hexdigest()


ID = str


# TODO: perhaps replace with UUID4
def generate_id() -> ID:
    """Generates id for some job that needs to be performed."""
    return secrets.token_hex(16)

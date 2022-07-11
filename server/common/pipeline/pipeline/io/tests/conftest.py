from pathlib import Path

import pytest

from ..._config import Settings
from .. import _numpy_io


@pytest.fixture(autouse=True)
def data_path(tmp_path, monkeypatch) -> Path:
    path_str = str(tmp_path.resolve())

    monkeypatch.setattr(
        _numpy_io,
        "settings",
        Settings(
            result_max_rows=10,
            result_max_values=100,
            results_location=path_str,
            input_location=path_str,
        ),
    )

    return tmp_path

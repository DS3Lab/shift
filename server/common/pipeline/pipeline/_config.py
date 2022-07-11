import os

from pydantic import BaseSettings

__all__ = ["Settings", "settings"]


class Settings(BaseSettings):
    result_max_rows: int
    result_max_values: int
    results_location: str
    input_location: str

    def get_results_path_str(self, job_id: str) -> str:
        return os.path.join(self.results_location, job_id)

    def get_input_path_str(self, relative_path: str) -> str:
        return os.path.join(self.input_location, relative_path)


settings = Settings()

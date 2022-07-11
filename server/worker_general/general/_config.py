import os

from pydantic import BaseSettings

__all__ = ["Settings", "settings"]


class Settings(BaseSettings):
    redis_broker: str
    redis_backend: str
    tf_prefetch_size: int

    # Disable multiprocessing for unit tests
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    pt_prefetch_factor: int
    pt_num_workers: int

    tfds_location: str
    torch_dataset_location: str
    input_location: str

    def get_input_path_str(self, relative_path: str) -> str:
        return os.path.join(self.input_location, relative_path)


settings = Settings()

import os
from loguru import logger

class ShiftExtension():
    def __init__(self, name):
        self.name = name
        base_path = os.environ.get("USERSPACE_LOCATION", None)
        if base_path is None:
            raise Exception("USERSPACE_LOCATION environment variable is not set")
        self.base_src_path = os.path.join(base_path, "extensions", "src", self.name)
        self.base_data_path = os.path.join(self.base_src_path, "data")
        logger.info(f"Extension {name} initialized: base_data_path={self.base_data_path}, base_src_path={self.base_src_path}")        
        os.makedirs(self.base_data_path, exist_ok=True)
        
    def __call__(self, models, task):
        raise NotImplementedError
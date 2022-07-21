import os

class ShiftExtension():
    def __init__(self, name):
        self.name = name
        base_path = os.environ.get("USERSPACE_LOCATION", ".")
        self.base_data_path = os.path.join(base_path,"extensions","src", self.name, "data")
        os.makedirs(self.base_data_path, exist_ok=True)
        
    def __call__(self, models, task):
        raise NotImplementedError
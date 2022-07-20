import os

class ShiftExtension():
    def __init__(self, name):
        self.name = name
        self.base_data_path = os.path.join("userspace","extensions","src",self.name, "data")
        os.makedirs(self.base_data_path, exist_ok=True)
        
    def __call__(self, models, task):
        raise NotImplementedError
import os
from fastapi import FastAPI

ext_app = FastAPI(title="shift_ext", description="Extensions API for SHiFT")

@ext_app.get("/exts")
def get_exts():
    return {"exts": ["shift_ext"]}

@ext_app.post("/exts/{ext_name}")
def request_ext():
    return {"ext_name": "shift_ext"}

if __name__=="__main__":
    """
    Here we register all extensions available in the system.
    """
    # available_srcs = os.listdir()
    base_path = os.environ.get("USERSPACE_LOCATION", ".")
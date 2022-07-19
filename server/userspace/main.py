from fastapi import FastAPI

ext_app = FastAPI(title="shift_ext", description="Search Engine for Transfer Learning")

@ext_app.get("/exts")
def get_exts():
    return {"exts": ["shift_ext"]}

@ext_app.post("/exts/{ext_name}")
def request_ext():
    return {"ext_name": "shift_ext"}
import os
import inspect
import importlib
from typing import List, Dict
from fastapi import FastAPI

from pydantic import BaseModel

ext_app = FastAPI(title="shift_ext", description="Extensions API for SHiFT")
registry = {}

class ExtensionRequest(BaseModel):
    models: List
    readers: Dict


@ext_app.get("/exts")
async def get_exts():
    return registry

@ext_app.post("/ext/{ext_name}")
async def request_ext(ext_name, req: ExtensionRequest):
    ext_cls = registry[ext_name]()
    response = ext_cls(req.models, req.readers)
    return response

@ext_app.on_event("startup")
async def startup_event():
    base_path = os.environ.get("USERSPACE_LOCATION", ".")
    extensions_dir = os.path.join(base_path, "extensions", "src")
    extensions = os.listdir(extensions_dir)
    for extension in extensions:
        entry_file = os.path.join(extensions_dir, extension, "__init__.py")
        spec = importlib.util.spec_from_file_location(extension, entry_file)
        ext_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext_module)
        clsmembers = inspect.getmembers(ext_module, inspect.isclass)
        clsmembers = [x for x in clsmembers if x[0].endswith("Extension") and not x[0].startswith("ShiftExtension")]
        assert len(clsmembers) == 1, f"Found more than one extension class in {extension}"
        registry[extension] = clsmembers[0][1]

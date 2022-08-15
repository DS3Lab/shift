# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

common_imports = [
    "../../server",
    "../../server/common",
    "../../server/common/db_tools",
    "../../server/common/finetune",
    "../../server/common/optimizations",
    "../../server/common/pipeline",
    "../../server/common/schemas",
    "../../server/common/task2vec",
    "../../server/common/telemetry",
]

for each in common_imports:
    sys.path.insert(0, os.path.abspath(each))


# -- Project information -----------------------------------------------------

project = "SHiFT"
copyright = "2022, DS3Lab, ETH Zurich"
author = "DS3Lab"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "private-members": False,
    # "special-members": "__init__",
    "special-members": "__iter__,__next__,__getitem__,__setitem__,__contains__",
    "undoc-members": True,
    # "show-inheritance": True,
}

autodoc_mock_imports = [
    "tensorflow",
    "tensorflow_datasets",
    "tensorflow_hub",
    "tensorflow_addons",
    # sphinx does not support inheritance + ABC for Python < 3.7
    # https://github.com/sphinx-doc/sphinx/issues/5995
    "schemas",
    "celery",
    "datasets",
    "torch",
    "torchvision",
    "numpy",
    "pydantic",
    "pottery",
    "redis",
    "transformers",
    "PIL",
    "sklearn",
    "typing_extensions",
    "faiss",
    "scipy",
    "dstool",
]

autodoc_typehints = "none"
add_module_names = False
autodoc_member_order = "bysource"

source_suffix = ['.rst', '.md']

# sphinx-autodoc-typehints
set_type_checking_flag = True
always_document_param_types = True

# intersphinx_mapping = {
#     "worker_general": (
#         "../../../worker_general/doc/_build/html",
#         "../../worker_general/doc/_build/html/objects.inv",
#     ),
# }

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

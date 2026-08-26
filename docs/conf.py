"""Sphinx configuration.

Build the HTML documentation with::

    pip install -e '.[docs]'
    sphinx-build -b html docs docs/_build/html

then open ``docs/_build/html/index.html``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "fitting"
author = "Lucas A. N. Melo"
copyright = "Fordyce Lab, Stanford University"  # pylint: disable=redefined-builtin
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Google-style docstrings, with the type information already in the signatures.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}
# Equinox rewrites __init__; documenting it adds nothing over the field list.
autodoc_mock_imports: list[str] = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

html_theme = "sphinx_rtd_theme"
html_title = "fitting"
html_static_path = ["_static"]

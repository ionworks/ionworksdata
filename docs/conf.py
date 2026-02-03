import os
import sys
import ionworksdata as iwd

# Path for repository root
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "Ionworks Data Processing"
copyright = "2025, Ionworks Technologies Inc"
author = "Ionworks Technologies Inc"

# Note: Both version and release are used in the build
version = iwd.__version__
release = iwd.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    # Sphinx extensions
    "sphinx.ext.autodoc",
    # Third-party extensions
    "sphinx_design",
    "myst_nb",
]
myst_dmath_double_inline = True
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
root_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "source/_static/iw-icon.png"
html_static_path = ["source/_static"]
html_favicon = "source/_static/iw-icon.png"
html_permalinks_icon = "<span>¶</span>"

# Base URL for the documentation
html_baseurl = "https://data.docs.ionworks.com/"

# Include CNAME file in build output
html_extra_path = ["CNAME"]

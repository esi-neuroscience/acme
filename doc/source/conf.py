#
# Configuration file for the Sphinx documentation builder.
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
import time
sys.path.insert(0, os.path.abspath(".." + os.sep + ".." + os.sep))
import sphinx_bootstrap_theme
import acme

# -- Project information -----------------------------------------------------
project = ''
author = 'Ernst Strüngmann Institute for Neuroscience in Cooperation with Max Planck Society'
copyright = f'2020-{time.strftime("%Y")}, {author}'

# The short X.Y version
version = acme.__version__.partition("-")[0]
# The full version, including alpha/beta/rc tags
release = acme.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram',
    'sphinx_automodapi.automodapi',
    'sphinx_tabs.tabs',
]

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'exclude-members': '__weakref__',
    'ignore-module-all': False,
}

def setup(app):
    app.add_css_file("esi-style.css")

numpydoc_show_class_members = False
autosectionlabel_maxdepth = 2
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Create stub pages for autosummary entries
autosummary_generate = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# Custom sidebar templates, maps document names to template names.
html_sidebars = {"**" : ["localtoc.html"]}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'

# Do not display permalink symbol next to headers
html_permalinks = False

# Path to favicon
html_favicon = "_static/acme_icon.ico"

# Logo displayed in navbar
html_logo = "_static/acme_logo.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_theme_options = {
    "navbar_title": "",
    "navbar_site_name": "Documentation",
    # Render the next and previous page links in navbar. (Default: true)
    "navbar_sidebarrel": False,
    # Render the current pages TOC in the navbar. (Default: true)
    "navbar_pagenav": False,
    # Tab name for the current pages TOC. (Default: "Page")
    "navbar_pagenav_name": None,

    'globaltoc_includehidden': True,

    # Global TOC depth for "site" navbar tab. (Default: 1)
    # Switching to -1 shows all levels.
    "globaltoc_depth": 1,
    "bootswatch_theme": "lumen",
    "navbar_links": [
        ("GitHub", "https://www.github.com/esi-neuroscience/acme", True),
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3",
               "https://docs.python.org/3/objects.inv"
    ),
    "numpy": (
        "https://docs.scipy.org/doc/numpy/",
        "https://docs.scipy.org/doc/numpy/objects.inv",
    ),
    "dask": (
        "https://docs.dask.org/en/latest",
        "https://docs.dask.org/en/latest/objects.inv",

    ),
    "distributed": (
        "https://distributed.dask.org/en/stable/",
        "https://distributed.dask.org/en/stable/objects.inv",
    ),
    "jobqueue": (
        "https://jobqueue.dask.org/en/stable/",
        "https://jobqueue.dask.org/en/stable/objects.inv"
    ),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference",
              "https://docs.scipy.org/doc/scipy/reference/objects.inv"
    ),
    "h5py": ("https://docs.h5py.org/en/latest/",
             "https://docs.h5py.org/en/latest/objects.inv"
    ),
    "sklearn": ("https://scikit-learn.org/stable/",
                "https://scikit-learn.org/stable/objects.inv"
    ),
    "torch": ("https://pytorch.org/docs/stable/",
              "https://pytorch.org/docs/stable/objects.inv"
    ),
    "nilearn": ("https://nilearn.github.io/stable/",
                "https://nilearn.github.io/stable/objects.inv"
    )
}

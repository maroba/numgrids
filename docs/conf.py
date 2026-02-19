# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import datetime
import os
import re

project = 'numgrids'
author = 'Matthias Baer'

this_year = datetime.date.today().year
if this_year == 2023:
    copyright = f'2023, {author}'
else:
    copyright = f'2023 - {this_year}, {author}'

with open(os.path.join('..', 'numgrids', '__init__.py'), 'r') as fh:
    match = re.match('__version__ *= *([^ ]+)', fh.readline())
    if match:
        release = match.group(1)
    else:
        raise ValueError('numgrids.__init__.py must define version string as first line.')


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.intersphinx',
    'IPython.sphinxext.ipython_console_highlighting', 'sphinx.ext.napoleon',
    'IPython.sphinxext.ipython_directive', 'myst_parser', 'sphinx.ext.mathjax',
    'nbsphinx',
]

autoclass_content = 'both'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'pypi_description.md']

nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nature'
html_static_path = ['_static']

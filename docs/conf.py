# Configuration file for the Sphinx documentation builder.

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

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'myst_parser',
    'nbsphinx',
    'sphinx_design',
]

autoclass_content = 'both'
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'pypi_description.md']

nbsphinx_execute = 'always'
nbsphinx_allow_errors = False

# MyST configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'fieldlist',
    'dollarmath',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#2962FF',
        'color-brand-content': '#2962FF',
    },
    'dark_css_variables': {
        'color-brand-primary': '#82B1FF',
        'color-brand-content': '#82B1FF',
    },
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
    'source_repository': 'https://github.com/maroba/numgrids',
    'source_branch': 'main',
    'source_directory': 'docs/',
}

html_title = 'numgrids'

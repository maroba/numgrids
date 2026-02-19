# Development Setup

This page walks you through setting up a local development environment for
**numgrids**.

## Prerequisites

- Python 3.10 or newer
- Git

## Clone the Repository

```bash
git clone https://github.com/maroba/numgrids.git
cd numgrids
```

## Create a Virtual Environment

It is recommended to use a dedicated virtual environment to isolate dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

## Install in Development Mode

Install numgrids as an editable package so that local changes are reflected
immediately:

```bash
pip install -e .
```

## Install Development Dependencies

Install the tools needed for testing, building documentation, and running notebooks:

```bash
pip install pytest sphinx furo sphinx-design myst-parser nbsphinx ipython ipykernel
```

| Package | Purpose |
|---|---|
| `pytest` | Test runner |
| `sphinx` | Documentation generator |
| `furo` | Sphinx theme |
| `sphinx-design` | Cards, tabs, and other design elements for Sphinx |
| `myst-parser` | Markdown support for Sphinx |
| `nbsphinx` | Jupyter notebook integration in Sphinx |
| `ipython` / `ipykernel` | Running Jupyter notebooks |

## Building the Documentation Locally

From the repository root:

```bash
cd docs
make html
```

Then open the generated site in your browser:

```bash
open _build/html/index.html        # macOS
xdg-open _build/html/index.html    # Linux
```

The documentation is built with **Sphinx** using the **Furo** theme and
**MyST Markdown** for content files.

## Running Notebooks

Some documentation pages are Jupyter notebooks. To run them locally you need an
IPython kernel registered in your virtual environment:

```bash
python -m ipykernel install --user --name numgrids --display-name "Python (numgrids)"
```

After that you can open and execute the notebooks with Jupyter:

```bash
jupyter notebook
```

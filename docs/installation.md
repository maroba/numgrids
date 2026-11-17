# Installation

## From PyPI

The recommended way to install numgrids is via pip:

```bash
pip install numgrids
```

To upgrade to the latest release:

```bash
pip install --upgrade numgrids
```

## From source

For development or to use the latest unreleased code, clone the repository and
install in editable mode:

```bash
git clone https://github.com/maroba/numgrids.git
cd numgrids
pip install -e .
```

## Optional features

Plotting and xarray support are optional extras:

```bash
pip install "numgrids[plot]"     # matplotlib, for grid.plot(...)
pip install "numgrids[xarray]"   # xarray, for to_xarray / from_xarray
pip install "numgrids[all]"      # both
```

For development, `pip install -e ".[dev]"` installs the test, lint and
documentation tools as well.

## Dependencies

numgrids requires **Python 3.10 or later** and depends on the following
packages (installed automatically by pip):

| Package       | Minimum version | Purpose                                  |
|---------------|-----------------|------------------------------------------|
| numpy         | >= 1.22         | Array operations and meshgrid generation |
| scipy         | >= 1.11         | Sparse matrices, interpolation, linear algebra, ODE solvers |
| findiff       | >= 0.10         | Finite-difference stencil generation     |
| matplotlib    | >= 3.5          | Plotting (optional, `numgrids[plot]`)    |
| xarray        | >= 2023.1       | Labelled data and NetCDF (optional, `numgrids[xarray]`) |

```{note}
All dependencies are available as binary wheels on most platforms, so
installation typically requires no compiler.
```

## Verifying the installation

After installing, confirm that numgrids is importable and check the installed
version:

```bash
python -c "import numgrids; print(numgrids.__version__)"
```

You should see output like:

```text
0.6.0
```

```{tip}
If you are working in a virtual environment (recommended), make sure it is
activated before running `pip install`.
```

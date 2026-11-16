# Testing

numgrids uses [pytest](https://docs.pytest.org/) as its test runner. All tests live in
the `tests/` directory at the repository root; in addition, the examples in the
docstrings are run as doctests.

## Running the Full Test Suite

```bash
python -m pytest
```

This runs `tests/` and the doctests in `numgrids/` (configured in
`pyproject.toml`). With coverage:

```bash
python -m pytest --cov=numgrids --cov-report=term-missing
```

Lint the code with:

```bash
ruff check numgrids tests
```

To see verbose output with individual test names:

```bash
python -m pytest tests -v
```

## Running Specific Tests

Run a single test file:

```bash
python -m pytest tests/test_diff.py
```

Run a single test class or function:

```bash
python -m pytest tests/test_diff.py::TestDiffSomeClass
python -m pytest tests/test_diff.py::TestDiffSomeClass::test_specific_case
```

## Test Organization

The test suite is organized with roughly **one test file per module**:

| Test file | What it covers |
|---|---|
| `test_grids.py` | Grid creation and properties |
| `test_diff.py` | Differentiation operators |
| `test_boundary.py` | Boundary conditions (Dirichlet, Neumann, Robin) |
| `test_axes.py` | Axis types (uniform, logarithmic, etc.) |
| `test_curvilinear.py` | Curvilinear grids (spherical, cylindrical, polar) |
| `test_amr.py` | Adaptive mesh refinement |
| `test_integration.py` | Numerical integration |
| `test_interpol.py` | Interpolation |
| `test_io.py` | Save / load functionality |
| `test_api.py` | Public API surface |
| `test_plots.py` | Plotting utilities |

## Writing New Tests

When adding or modifying functionality, always include corresponding tests.

### Test structure

Tests can be plain pytest functions (preferred for new tests) or
`unittest.TestCase` classes, which the older test modules use:

```python
import numpy as np
from numpy.testing import assert_allclose

from numgrids import Grid, chebyshev


def test_second_derivative_of_polynomial():
    grid = Grid(chebyshev(20, -1, 1))
    x = grid.coords
    assert_allclose(grid.diff(x**3, order=2), 6 * x, atol=1e-10)
```

### Numerical assertions

Because numgrids deals with numerical computations, most assertions compare arrays
with a finite tolerance. Prefer helpers from `numpy.testing`:

- `numpy.testing.assert_array_almost_equal` -- element-wise comparison up to a number
  of decimal places.
- `numpy.testing.assert_allclose` -- relative and absolute tolerance comparison.

### Mathematical correctness

Tests should verify results against **known analytical solutions** whenever possible.
For example, a test for a second-order derivative operator might compare its output to
the exact second derivative of a polynomial on the same grid. This ensures that the
numerical discretization converges to the correct mathematical result.

# Changelog

All notable changes to **numgrids** are documented on this page.

## 0.5.0

### Breaking changes

- `integrate()` / `Integral` on curvilinear grids now include the volume
  element, i.e. compute $\int f\,dV$. Remove manual Jacobian factors from
  integrands, or use `Integral(grid, volume_element=False)` for the old
  coordinate-space behavior.
- matplotlib is no longer a required dependency; install
  `numgrids[plot]` for plotting.
- Axes need at least two points; unknown keyword arguments to axis
  constructors raise `TypeError`.
- `numgrids.Axis` (an alias of `create_axis`) is deprecated.

### Bug fixes

- Chebyshev derivatives of order 3 and higher were wrong.
- Derivatives of order 2 and higher on logarithmic axes were wrong.
- `refine()`, `coarsen()` and `refine_axis()` returned plain `Grid`
  objects for curvilinear grids (which also broke `adapt()` on them).
- Interpolation on periodic axes failed between the last grid point and
  the end of the period.
- Interpolating a 1D grid at several points returned only one value.
- `from numgrids import *` followed by `diff(...)` could fail because the
  `numgrids.diff` submodule shadowed the function.
- Chebyshev nodes are exactly symmetric with exact endpoints.

### New features

- Short-hand axis factories `chebyshev`, `equidistant`, `periodic`,
  `logarithmic`; `create_axis` accepts strings.
- Named axes via `Grid(x=..., y=...)`; names work wherever an axis index
  is expected, and as face keys (`grid.faces["x_low"]`).
- Grid methods `diff`, `integrate`, `interpolate`, `evaluate`,
  `laplacian`, `gradient`, `divergence`, `curl` (also on Cartesian grids),
  `diff_operator`, `laplacian_operator`, `save`, `load`, `interior`.
- `Operator` class with operator algebra, `solve()` for boundary value
  problems and `eig()` for eigenvalue problems without spurious modes.
- Compact boundary condition specifications (`bcs=0`,
  `{"x_low": 0, "x_high": Neumann(1)}`) via `Dirichlet`, `Neumann`,
  `Robin` and `resolve_bcs`.
- Boundary face operations: `restrict`, `normal_derivative`, `integrate`,
  `flux`.
- Spectral interpolation (`method="spectral"`) and quintic splines.
- Clenshaw--Curtis quadrature for Chebyshev axes (`Axis.quadrature_weights`).
- New coordinate systems: `EllipticCylindricalGrid`,
  `ProlateSpheroidalGrid`, `OblateSpheroidalGrid`; `cartesian_coords`,
  `scale_factors` and `volume_element` for all curvilinear grids.
- `evolve()` for method-of-lines time integration.
- `convergence_study()` for measuring convergence orders.
- `to_xarray()` / `from_xarray()` for xarray and NetCDF interoperability.
- `grid.plot(f)` for 1D, 2D (Cartesian space for curvilinear grids) and
  sliced higher-dimensional fields.
- All example notebooks rewritten for the new API, plus a new
  "Operators and solvers" notebook.

### Improvements

- Much faster integration, boundary condition application, FFT
  differentiation setup and multi-point interpolation.
- Clear error messages for shape mismatches, too-small finite-difference
  stencils and singular systems.
- Uses the current findiff API (no deprecation warnings).
- Type information (`py.typed`), modern CI (lint, tests on Linux, macOS
  and Windows, minimum dependency versions, example notebooks, docs).

## 0.4.0

- Curvilinear grids with vector calculus operators (`SphericalGrid`,
  `CylindricalGrid`, `PolarGrid`).
- Adaptive mesh refinement (`adapt`, `estimate_error`, `ErrorEstimator`).
- Sparse matrix representation for `LogDiff`.
- Documentation overhaul with Furo theme.

## 0.3.0

- Boundary conditions (`Dirichlet`, `Neumann`, `Robin`).
- Save / load grids to `.npz` files.
- Logarithmic axes.

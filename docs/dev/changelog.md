# Changelog

All notable changes to **numgrids** are documented on this page.

## 0.7.0

### New features

- **Nonlinear boundary value problems**: `solve_nonlinear(grid, residual, u0, bcs=...)`
  solves $F(u) = 0$ with a damped Newton method. The residual is an ordinary
  function of grid-shaped arrays. The Jacobian is computed by finite
  differences, or passed exactly via `jacobian=` (e.g. written in operator
  algebra: `lambda u: D2 + np.exp(u)`). It supports Dirichlet, Neumann and Robin
  conditions, curvilinear grids and coupled systems (with block Jacobians). It
  raises `NonlinearSolverError` (with the last iterate and the residual
  history) if Newton's method does not converge.
- **Neumann and Robin conditions in `evolve`**: the boundary values are
  computed from the interior values at every instant, so the conditions hold
  exactly (e.g. an insulated body conserves its heat). Coupled systems accept
  one boundary specification per field, e.g. `bcs=(Neumann(0), Neumann(0))`
  for reaction-diffusion systems with no-flux boundaries.
- **Generalized eigenvalue problems** $L u = \lambda M u$: `Operator.eig(..., M=rho)`
  with a mass density or weight function given as an array, a callable, an
  `Operator` or a matrix. It works with all boundary conditions and on the dense
  and sparse paths.

### Documentation

- New user guide page *Nonlinear problems* and example notebook
  *Nonlinear boundary value problems* (Bratu problem, continuation and
  bifurcation diagram, exact Jacobians, full disk, temperature-dependent
  conductivity).
- The heat equation notebook covers insulated and cooled ends, and the drum
  notebook covers a membrane with variable density.

## 0.6.1

### Bug fixes

- `Operator.eig` returns real eigenvectors for real operators, also for
  degenerate eigenvalues (e.g. symmetry-related modes on a disk, square or
  sphere). Previously, the eigenvectors of such modes could be complex
  combinations, and the sparse path never returned real arrays. A degenerate
  eigenspace is now returned as a real orthonormal basis. Genuinely complex
  eigenvalues and their eigenvectors are unchanged.

### Documentation

- New README and PyPI project description with a title image, a short
  introductory example and a gallery of the example notebooks.
- `CITATION.cff` for citing numgrids ("Cite this repository" on GitHub).

## 0.6.0

### Breaking changes

- The minimum supported SciPy version is now 1.11. SciPy 1.10's sparse LU
  solver (used by `Operator.solve` and `Operator.eig`) crashes with a
  segmentation fault on some CPUs, e.g. on GitHub's CI runners.

### Bug fixes

- `Operator.eig` (sparse path, more than 2000 unknowns): no loss of
  accuracy for operators with an exact zero eigenvalue, no missed members
  of degenerate eigenvalue clusters, and reproducible results.

### New features

- Full disks, cylinders and balls: `radial(n, R)` creates a radial axis
  through the origin for `PolarGrid`, `CylindricalGrid` and `SphericalGrid`.
  Radial derivatives use the whole diameter (no grid point and no
  singularity at the center, no boundary condition needed there), with
  spectral accuracy for derivatives, Laplacians, integrals and
  interpolation, no spurious eigenvalues, and stable time stepping.
- Full spheres including the poles: `colatitude(n)` creates a polar-angle
  axis whose derivatives continue across the poles ("double Fourier
  sphere"), with Fejér quadrature. Together with `radial(n, R)` this covers
  full balls without any coordinate singularity.
- `SphericalSurfaceGrid` for the surface of a sphere; its Laplacian is the
  Laplace--Beltrami operator (eigenvalues l(l+1)).

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
- Automated releases: pushing a version tag publishes to PyPI via Trusted
  Publishing and creates a GitHub release.

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

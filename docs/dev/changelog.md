# Changelog

All notable changes to **numgrids** are documented on this page.

## 0.7.0

### New features

**Grids**

- **Unbounded domains**: `semi_infinite(n, scale)` creates an axis on the
  half line $[a, \infty)$ and `infinite(n, scale)` one on the whole real
  line (mapped Chebyshev discretizations with spectral accuracy). There is
  no grid point at infinity and no condition to impose there: decaying
  solutions are selected by the problem itself. Derivatives, integrals,
  interpolation, boundary conditions at the finite end, save/load and
  xarray work as for the other axes, also in products with them and as the
  radial axis of exterior domains. Examples: the levels of the harmonic
  oscillator on the whole line to 14 digits with 64 points, and those of
  the hydrogen atom to 12 digits with 48 points.

**Solving equations**

- **Nonlinear boundary value problems**: `solve_nonlinear(grid, residual, u0, bcs=...)`
  solves $F(u) = 0$ with a damped Newton method. The residual is an ordinary
  function of grid-shaped arrays. The Jacobian is computed by finite
  differences, or passed exactly via `jacobian=` (e.g. written in operator
  algebra: `lambda u: D2 + np.exp(u)`). It supports Dirichlet, Neumann and Robin
  conditions, curvilinear grids and coupled systems (with block Jacobians). It
  raises `NonlinearSolverError` (with the last iterate and the residual
  history) if Newton's method does not converge.
- **Continuation**: `continuation(grid, residual, u0, param, ...)` follows a
  branch of solutions of $F(u, \lambda) = 0$ in the parameter by
  pseudo-arclength continuation. It passes folds (turning points), where
  solving at a fixed parameter fails, adapts its step size, and returns a
  `ContinuationResult` with the branch, its folds and `measure()` for
  bifurcation diagrams.
- **Linear stability**: `linear_stability(grid, residual, u, bcs)` computes
  the leading eigenvalues and modes of the linearization about a steady
  state of $u_t = F(u)$, with the boundary conditions built in, for single
  fields and systems (also with a mass matrix), and tells whether the state
  is stable.
- **Stability and bifurcations along a branch**: `continuation(..., stability=k)`
  follows the leading eigenvalues along the branch and reports where the
  stability changes: folds, branch points (pitchfork, transcritical) and
  Hopf bifurcations, each located on the branch (`result.bifurcations`,
  `result.stable`). Folds are located to solver accuracy
  (`result.fold_params`; the fold of the Bratu problem to 14 digits), and
  `tangent=` switches to the branch that bifurcates at a branch point.
- **Fast linear solvers**: `Operator.solve` now chooses between three direct
  methods (`method="auto"`, `"fourier"`, `"dense"`, `"sparse"`):
  - *Fourier decoupling*: problems that do not depend on a periodic axis
    (the azimuthal angle of a disk, cylinder, sphere or ball; periodic boxes)
    split into one small system per Fourier mode. A Poisson problem on a full
    ball with 8192 unknowns takes 0.2 s instead of 30 s.
  - *Dense LU* for spectral grids, where sparse LU fills in almost
    completely (several times faster).
  - *Sparse LU* for finite differences, as before.
- **Singular systems**, such as a Poisson problem with Neumann conditions on
  the whole boundary, are solved in the least-squares sense: `solve` returns
  one of the solutions instead of depending on round-off.
- **Reusable factorizations**: `Operator.factorize(bcs)` returns a
  `LinearSolver` that solves for many right-hand sides with one
  factorization, e.g. for implicit time stepping.
- **Faster eigenvalue problems**: `Operator.eig` uses the same Fourier
  decoupling: one small dense eigenproblem per Fourier mode instead of one
  large problem, with all eigenvalues to full accuracy (full ball with 8192
  unknowns: 1 s instead of 22 s). `normalize="l2"` returns eigenfunctions
  with $\int |u|^2 dV = 1$.
- **Generalized eigenvalue problems** $L u = \lambda M u$: `Operator.eig(..., M=rho)`
  with a mass density or weight function given as an array, a callable, an
  `Operator` or a matrix. It works with all boundary conditions and on the dense
  and sparse paths.

- **Multigrid solver**: `MultiGrid(...).solve(operator, rhs, bcs)` solves
  linear boundary value problems on finite-difference grids with geometric
  multigrid (as a preconditioner for GMRES), in time and memory
  proportional to the number of unknowns. This makes large 3D problems
  feasible, where sparse LU factorizations run out of time and memory: a
  Poisson problem on a 65³ grid (275 000 unknowns) takes a few seconds.
  `operator` is a function that builds the operator on a grid, so variable
  coefficients work; Dirichlet, Neumann and Robin conditions are supported.
  `MultiGrid` also accepts named axes now.

**Time evolution**

- **Neumann and Robin conditions in `evolve`**: the boundary values are
  computed from the interior values at every instant, so the conditions hold
  exactly (e.g. an insulated body conserves its heat). Coupled systems accept
  one boundary specification per field, e.g. `bcs=(Neumann(0), Neumann(0))`
  for reaction-diffusion systems with no-flux boundaries.
- **Time-dependent boundary values**: wrap a function of time in
  `TimeDependent`, e.g. `bcs={"x_low": TimeDependent(np.sin)}` or
  `Neumann(TimeDependent(flux))`.
- **Source terms**: `evolve(..., source=s)` with an array or a function of
  time. Operator right-hand sides stay linear, so implicit methods keep the
  exact Jacobian.
- **Linear systems as operator blocks**: `evolve(grid, [[None, 1], [L, None]], (u0, v0), ...)`
  for the wave equation and other coupled linear systems.
- Implicit methods (`"BDF"`, `"Radau"`) with an operator right-hand side are
  several times faster on spectral grids (they now get a dense Jacobian,
  which avoids the fill-in of sparse factorizations).
- **Stiff nonlinear problems**: `evolve(grid, N, u0, t_span, linear=L, method="ETDRK4", dt=...)`
  integrates semilinear equations $u_t = L u + N(t, u)$ such as the
  Korteweg--de Vries, nonlinear Schrödinger, Allen--Cahn, Swift--Hohenberg
  and reaction--diffusion equations. The stiff linear part is treated
  exactly by exponential time differencing (`"ETDRK4"`, `"ETDRK2"`,
  `"ETD1"`) or implicitly (`"IMEX2"`, which needs only linear solves), the
  nonlinear part explicitly, so the step size is not limited by the
  stiffness. On periodic grids the whole step runs in Fourier space, which
  makes two-dimensional pattern formation on 128 × 128 grids cheap; grids
  with boundaries support Dirichlet, Neumann and Robin conditions.
- **Exponential integrator**: `method="expm"` evaluates the solution of linear
  problems with matrix exponentials, directly at the output times. There is
  no time stepping error, complex fields and stiff operators work together,
  and the norm is conserved in Schrödinger-type problems.

**Operators and calculus**

- `grid.gradient_operator()` and `grid.divergence_operator()` return the
  gradient and divergence as operators (including the scale factors on
  curvilinear grids), and `grid.diffusion_operator(k)` builds
  $\nabla\cdot(k\,\nabla u)$ for variable coefficients.
- **Spectral coefficients**: `grid.coefficients(f)` and
  `grid.from_coefficients(c)` transform between grid values and expansion
  coefficients (Fourier on periodic axes, Chebyshev on Chebyshev and
  unbounded axes, and across the origin and the poles on radial and
  colatitude axes); `grid.resample(f, target)` changes the resolution
  exactly.
- **Resolution check**: `grid.resolution(f, tol)` tells from the decay of
  the coefficients, per axis, whether a function is resolved and how many
  points it needs, e.g. "axis x: NOT resolved, tail at 1e-03; about 107
  points needed"; `grid.is_resolved(f, tol)` is the short form, and
  `result.suggested_grid` the grid to continue with.
- **Filtering**: `grid.filter(f)` truncates or damps the high modes, e.g.
  for dealiasing nonlinear terms with the 2/3 rule.
- **Functions of operators**: `L.function(f, bcs)` builds $f(L)$ from the
  spectrum of the operator with the given boundary conditions, with the
  shortcuts `L.power(s, bcs)`, `L.sqrt(bcs)`, `L.exp(t, bcs)` and
  `L.inverse(bcs)`. `(-grid.laplacian_operator()).power(0.5, bcs=0)` is the
  fractional Laplacian with Dirichlet conditions. On periodic grids the
  function is applied to the Fourier symbol, on disks, spheres and balls to
  one small matrix per Fourier mode. The result is an ordinary operator
  for `solve`, `eig` and `evolve` (e.g. fractional diffusion).
- `Operator.T`, `Operator.H`, `Operator.adjoint()` (adjoint in the $L^2$ inner
  product of the grid) and `Operator.toarray()`.
- The coefficients of `Robin(a, b, value)` may vary along the face (arrays or
  functions of the face coordinates), e.g. for Newton's law of cooling with
  a position-dependent conductivity.
- `grid.inner(f, g)`, `grid.norm(f)` ($L^2$, $L^1$ or maximum norm),
  `grid.mean(f)` and `grid.quadrature_weights`.

**Visualization**

- **Animations**: `grid.animate(frames, t)` and `result.animate()` (for the
  result of `evolve`) return a matplotlib animation of a time-dependent
  field: a moving line in 1D, a color plot in 2D (in Cartesian space on
  curvilinear grids). Save it as GIF or HTML, or show it in a notebook.
- **VTK export**: `grid.to_vtk(path, **fields)` writes structured-grid files
  (`.vts`) for ParaView and VisIt, and `result.to_vtk(path)` /
  `grid.to_vtk_series(...)` a time series with a `.pvd` collection. No
  additional dependency is needed. Curvilinear grids are written in
  Cartesian space with periodic directions closed and the center and the
  poles filled in; vector fields are converted to Cartesian components.

### Bug fixes

- `divergence` and `curl` were wrong on full balls (a `SphericalGrid` with a
  radial axis through the origin): the components of a vector field change
  sign when the radial coordinate line is continued through the origin,
  which the derivative across the origin ignored. Derivatives of vector
  components now use the correct parity, on all grids with a radial or
  colatitude axis.

### Documentation

- New user guide page *Nonlinear problems* and example notebook
  *Nonlinear boundary value problems* (Bratu problem, continuation and
  bifurcation diagram, exact Jacobians, full disk, temperature-dependent
  conductivity).
- The heat equation notebook covers insulated and cooled ends, and the drum
  notebook covers a membrane with variable density.
- New example notebooks:
  - *Quantum dynamics*: a spreading wave packet, tunneling through a barrier
    (compared with stationary scattering) and a coherent state, computed
    with the exponential integrator.
  - *The hydrogen atom*: energy levels and wave functions on the half line,
    the role of the scale length, and a comparison with a truncated domain.
  - *Heat conduction*: variable conductivity, a seasonal temperature wave in
    the ground, a disk with a moving heat source, and implicit time
    stepping with a reused factorization.
  - *Solitons*: Korteweg--de Vries solitons and their collision, bright
    solitons and a breather of the nonlinear Schrödinger equation, compared
    with exact solutions.
  - *Pattern formation*: labyrinths and dividing spots in the Gray--Scott
    model, stripes and hexagons in the Swift--Hohenberg equation.
  - *Is my function resolved?*: coefficient spectra, the resolution check
    in one and two dimensions, choosing the number of points, aliasing.
- The nonlinear problems notebook traces the bifurcation diagram of the
  Bratu problem with `continuation()`, including the stability of the two
  branches.
- New user guide sections on unbounded domains, continuation, solver
  methods and factorizations, operators in divergence form, inner products
  and norms, time-dependent boundary values, source terms and the
  exponential integrator.

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

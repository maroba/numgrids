# MultiGrid

The `MultiGrid` class creates a hierarchy of grids at progressively coarser
resolutions over the same domain. This is the data structure needed for
multigrid iterative solvers, which accelerate convergence of elliptic PDE
solvers by transferring corrections between resolution levels. `MultiGrid`
also contains such a solver: [`MultiGrid.solve`](#solving-with-multigrid)
solves linear boundary value problems on the finest grid in a time that
grows only linearly with the number of grid points.

```python
from numgrids import *
import numpy as np
```

## Creating a MultiGrid

Pass the axes for the finest grid and an optional `min_size` parameter that
sets the minimum number of points per axis on the coarsest level:

```python
ax_x = create_axis(AxisType.EQUIDISTANT, 64, 0.0, 1.0)
ax_y = create_axis(AxisType.EQUIDISTANT, 64, 0.0, 1.0)

mg = MultiGrid(ax_x, ax_y, min_size=4)
```

Starting from the finest grid (64 x 64), the constructor repeatedly halves
each axis until one of them would drop below `min_size`. The default
`min_size` is 2.

As for `Grid`, axes can also be passed as keyword arguments, which names
them: `MultiGrid(x=equidistant(64, 0, 1), y=equidistant(64, 0, 1))`.

## Accessing levels

The `levels` property returns a list of `Grid` objects, ordered from finest
to coarsest:

```python
for i, level in enumerate(mg.levels):
    print(f"Level {i}: shape = {level.shape}")
```

Output:

```text
Level 0: shape = (64, 64)
Level 1: shape = (32, 32)
Level 2: shape = (16, 16)
Level 3: shape = (8, 8)
Level 4: shape = (4, 4)
```

- `mg.levels[0]` is always the finest grid (the one you passed in).
- `mg.levels[-1]` is the coarsest grid.

## Transferring data between levels

The `transfer()` method moves an array from one level to an adjacent level
using interpolation:

```python
# Sample a function on the finest grid
X, Y = mg.levels[0].meshed_coords
f_fine = np.sin(np.pi * X) * np.cos(np.pi * Y)

# Restrict (fine -> coarse): level 0 -> level 1
f_coarse = mg.transfer(f_fine, level_from=0, level_to=1)
print(f_coarse.shape)  # (32, 32)

# Prolongate (coarse -> fine): level 1 -> level 0
f_back = mg.transfer(f_coarse, level_from=1, level_to=0)
print(f_back.shape)    # (64, 64)
```

```{note}
Transfers are only allowed between **adjacent** levels (i.e.
`|level_from - level_to| == 1`). To move data across multiple levels,
chain the transfers.
```

The `method` parameter controls the interpolation order used for the
transfer (default `"linear"`):

```python
f_coarse = mg.transfer(f_fine, level_from=0, level_to=1, method="cubic")
```

## Mixed axis types

`MultiGrid` supports all axis types. Each level preserves the axis type,
domain bounds, periodicity, and name of the original axes:

```python
mg = MultiGrid(
    create_axis(AxisType.CHEBYSHEV, 32, 0.0, 1.0, name="x"),
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 64, 0.0, 2 * np.pi, name="theta"),
    min_size=4,
)

for i, level in enumerate(mg.levels):
    ax0 = level.get_axis(0)
    ax1 = level.get_axis(1)
    print(f"Level {i}: {type(ax0).__name__}({len(ax0)}), "
          f"{type(ax1).__name__}({len(ax1)}, periodic={ax1.periodic})")
```

## Solving with multigrid

`MultiGrid.solve` solves a linear boundary value problem $L u = f$ on the
finest grid with the multigrid method. You pass a *function that builds the
operator on a grid* -- it is called for every level of the hierarchy --, the
right-hand side and the boundary conditions:

```python
mg = MultiGrid(x=equidistant(129, 0, 1), y=equidistant(129, 0, 1))

exact = lambda x, y: np.sin(np.pi * x) * np.sin(2 * np.pi * y)
f = lambda x, y: -5 * np.pi**2 * exact(x, y)

# lap u = f with u = 0 on the boundary
u = mg.solve(lambda grid: grid.laplacian_operator(), f, bcs=0)

print(np.abs(u - mg.levels[0].evaluate(exact)).max())   # about 6e-08
```

The result is the solution of the same linear system that
`grid.laplacian_operator().solve(f, bcs=0)` solves, up to the tolerance of
the iteration.

### When to use it

`Operator.solve` factorizes the system matrix (sparse LU). That is exact and
fast for 1D and small 2D problems, and it is the only choice for spectral
grids. But the cost of the factorization grows much faster than the number
of unknowns, in particular in 3D: on a 33³ grid it already takes half a
minute, and the eight times larger 65³ grid was not timed any more.
Multigrid needs time and memory proportional to the number of unknowns:

| Poisson problem, Dirichlet | Unknowns | Iterations | `MultiGrid.solve` | `Operator.solve` |
|----------------------------|---------:|-----------:|------------------:|-----------------:|
| 2D, 129², `acc=4`          |   16 641 |         19 |             0.2 s |            0.3 s |
| 2D, 257², `acc=4`          |   66 049 |         19 |             0.6 s |            1.8 s |
| 2D, 513², `acc=4`          |  263 169 |         18 |             2.4 s |             14 s |
| 2D, 513², `acc=2`          |  263 169 |          7 |             2.1 s |            3.7 s |
| 3D, 33³, `acc=2`           |   35 937 |          8 |             0.2 s |            6.5 s |
| 3D, 33³, `acc=4`           |   35 937 |         30 |             0.5 s |             31 s |
| 3D, 65³, `acc=2`           |  274 625 |          8 |             3.2 s |        not timed |
| 3D, 65³, `acc=4`           |  274 625 |         29 |             4.2 s |        not timed |

(Total wall time including the setup, relative residual $10^{-8}$, one CPU
core. Most of the multigrid time is spent assembling the operators on the
levels, not iterating.) Rule of thumb: use `Operator.solve` in 1D and for 2D
grids up to about a hundred points per axis, `MultiGrid.solve` for larger 2D
and for 3D finite-difference problems.

### Operators, boundary conditions, right-hand sides

The operator function receives a `Grid` and returns an `Operator`. Build
coefficients from the grid that is passed in, so that they exist on every
level:

```python
def operator(grid):
    x, y = grid.meshed_coords
    k = 1 + 0.5 * x * y                                    # diffusion coefficient
    div_k_grad = (k * grid.laplacian_operator()
                  + 0.5 * y * grid.diff_operator(1, "x")   # dk/dx * d/dx
                  + 0.5 * x * grid.diff_operator(1, "y"))  # dk/dy * d/dy
    return -div_k_grad + (1 + x**2)                        # -div(k grad u) + c u

u = mg.solve(
    operator,
    rhs=1.0,
    bcs={"all": 0, "x_low": lambda coords: np.sin(np.pi * coords[1]), "y_high": Neumann(0)},
)
```

- `rhs` is a scalar, an array on the finest grid or a function of the
  coordinates.
- `bcs` accepts everything that `Operator.solve` accepts (see
  [Boundary conditions](boundary-conditions.md)): Dirichlet, Neumann and
  Robin conditions with scalar, array or callable values. Every boundary
  face needs a condition.

### Monitoring and options

```python
u, info = mg.solve(lambda grid: grid.laplacian_operator(), f, bcs=0,
                   tol=1e-10, return_info=True)

print(info.iterations)     # 20
print(info.residuals[-1])  # relative residual of the linear system, below 1e-10
print(info.shapes)         # ((129, 129), (65, 65), (33, 33), (17, 17), (9, 9), (5, 5))
```

| Option        | Default   | Meaning |
|---------------|-----------|---------|
| `tol`         | `1e-8`    | Tolerance for the residual norm, relative to the norm of the right-hand side. |
| `maxiter`     | `100`     | Maximum number of iterations (multigrid cycles). |
| `u0`          | zero      | Initial guess. |
| `pre`, `post` | `2`, `2`  | Smoothing sweeps before and after the coarse-grid correction. |
| `cycle`       | `"V"`     | `"V"` or `"W"` cycle. |
| `method`      | `"gmres"` | `"gmres"`: the multigrid cycle preconditions GMRES. `"plain"`: the cycle is iterated on its own. |
| `low_order`   | `True`    | Build the cycle from second-order stencils (see below). |
| `return_info` | `False`   | Also return a `MultiGridInfo` (iterations, residual history, levels). |

If the tolerance is not reached, a `MultiGridError` is raised; it carries
the last iterate (`err.u`) and the residual history (`err.residuals`):

```python
from numgrids.multigrid import MultiGridError

try:
    mg.solve(lambda grid: grid.laplacian_operator(), f, bcs=0, maxiter=5)
except MultiGridError as err:
    print(err.residuals[-1])   # about 3e-04 after 5 iterations
```

To solve for several right-hand sides, set the solver up once -- the setup
costs more than a solve:

```python
solver = mg.solver(lambda grid: grid.laplacian_operator(), bcs=0)
u1 = solver.solve(f)
u2 = solver.solve(1.0)
```

### How it works, and what to expect

One multigrid cycle smooths the error with a few damped Jacobi sweeps,
transfers the residual to the next coarser grid (full weighting), solves
there recursively -- by sparse LU on the coarsest grid --, interpolates the
correction back and smooths again. Boundary conditions are not smoothed but
satisfied exactly after every sweep.

Two things make this robust for the operators numgrids produces:

- The cycle is not iterated on its own but used as a preconditioner for
  GMRES (`method="gmres"`).
- The cycle is built from *second-order* stencils, whatever accuracy the
  operator asks for (`low_order=True`). Higher-order finite differences,
  including the default `acc=4`, use wide one-sided stencils next to the
  boundary, for which a simple smoother diverges. The solution is still
  that of your operator: the second-order cycle only preconditions it. To
  build the cycle, the operator function is also called with grids whose
  finite differences are limited to second order.

The number of iterations does not grow with the grid size (relative
residual $10^{-10}$):

| Laplacian                         | 33² | 129² | 513² | 17³ | 33³ | 65³ |
|-----------------------------------|----:|-----:|-----:|----:|----:|----:|
| `acc=2`, Dirichlet                |   8 |    8 |    8 |   9 |  10 |  10 |
| `acc=2`, Neumann on one face      |  11 |   10 |    – |  12 |  13 |   – |
| `acc=2`, Robin on all faces       |  11 |   11 |    – |  12 |  13 |   – |
| `acc=4`, Dirichlet                |  26 |   23 |   22 |  43 |  38 |  37 |
| `acc=4`, Neumann on one face      |  32 |   31 |    – |  45 |  42 |   – |
| `acc=4`, Robin on all faces       |  26 |   25 |    – |  42 |  38 |   – |

(– : not measured.)

Second-order operators converge fastest: the residual drops by a factor of
about 20 per iteration (with `method="plain"`, the classical V-cycle, by a
factor of 6 per cycle). Fourth-order operators need three to four times as
many iterations, because the second-order cycle approximates their boundary
stencils less well. If second-order accuracy is enough,
`grid.laplacian_operator(acc=2)` is the fastest choice.

### Limitations

- **Finite-difference axes only.** Equidistant (non-periodic) and
  logarithmic axes work. Chebyshev and periodic axes have dense spectral
  differentiation matrices, for which multigrid is pointless; `solve`
  raises a `TypeError` -- use `Operator.solve` there.
- **Grid sizes.** Any size works, also non-nested ones (64 → 32 → 16).
  Sizes of the form $2^k + 1$ (65, 129, 257, ...) are preferable: the coarse
  grid points then coincide with fine grid points and the hierarchy reaches
  down to 5 points per axis. Levels with fewer than 5 points along an axis
  are not used; a larger `min_size` stops the coarsening earlier.
- **Anisotropy.** The grid spacing should be similar in all directions.
  With spacings (or diffusion coefficients) that differ by a factor of 2,
  4 and 8, the 2D Poisson problem needs 35, 55 and 92 iterations instead of
  26. Choose the number of points per axis in proportion to the side
  lengths of the domain. A logarithmic axis combined with other axes is
  anisotropic in part of the domain, too.
- **Elliptic problems only.** Indefinite problems (Helmholtz with several
  wavelengths in the domain), convection-dominated problems and operators
  whose second derivatives are written as products of first derivatives
  converge slowly or not at all. Write second derivatives with
  `diff_operator(2, axis)` or `laplacian_operator()`.
- **Singular problems.** With Neumann conditions on all faces the solution
  is only determined up to a constant; `solve` raises a `ValueError`. Use a
  Dirichlet or Robin condition on at least one face.
- **The operator must be a function of the grid.** A ready-made `Operator`
  on the finest grid is not accepted, because it cannot be rebuilt on the
  coarser grids.

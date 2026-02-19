# Using *numgrids*

## Installation

To install the latest version of *numgrids*, simply use *pip*:

```
pip install --upgrade numgrids
```

## Axes and Grids

The basic entities in *numgrids* are axes and grids.
An **axis** represents a 1-D discretisation of a coordinate direction.
A **grid** is the tensor product of one or more axes.

```python
from numgrids import *
import numpy as np

# Equidistant axis with periodic boundary conditions
axis_phi = create_axis(AxisType.EQUIDISTANT, 50, 0, 2 * np.pi, periodic=True)

# Chebyshev axis (non-uniform, higher density near edges)
axis_r = create_axis(AxisType.CHEBYSHEV, 20, 0, 1)

# Combine into a 2-D grid
grid = Grid(axis_r, axis_phi)
R, Phi = grid.meshed_coords
```

### Choosing an axis type

*numgrids* offers four axis types. Each one selects a different point
distribution and differentiation strategy:

| Axis type | Best for | Differentiation method |
|---|---|---|
| `EQUIDISTANT` | Uniform sampling, finite-difference problems | Finite differences (configurable accuracy) |
| `EQUIDISTANT` with `periodic=True` | Periodic directions (angles, Fourier problems) | FFT spectral method |
| `CHEBYSHEV` | Smooth non-periodic functions where high accuracy is needed | Chebyshev spectral method |
| `LOGARITHMIC` | Domains where fine resolution is needed near the lower bound (e.g. radial coordinates near zero) | Finite differences on log-scale + chain rule |

**Rules of thumb:**

- For **periodic** coordinates (e.g. azimuthal angle *φ*), always use
  `EQUIDISTANT` with `periodic=True` — the FFT spectral method gives
  exponential convergence.
- For **non-periodic smooth** functions, `CHEBYSHEV` typically reaches a
  given accuracy with far fewer points than equidistant spacing.
- For **large dynamic ranges** (e.g. *r* from 0.01 to 100), `LOGARITHMIC`
  packs more points where the function varies fastest.
- If you don't know yet, start with `EQUIDISTANT` — it is the most
  straightforward and works well as a baseline.

### Logarithmic axes

Logarithmic axes place grid points on a log-scale, giving high resolution
near the lower boundary. The lower bound must be strictly positive:

```python
# 40 points from 0.01 to 100, dense near 0.01
axis_r = create_axis(AxisType.LOGARITHMIC, 40, 0.01, 100)

grid = Grid(axis_r)
R = grid.meshed_coords[0]
f = 1 / R  # fine resolution where f varies fastest
```

Differentiation on a `LOGARITHMIC` axis uses finite differences on the
internal log-scale coordinate and applies the chain rule automatically —
no manual coordinate transformation needed.

## Differentiation and Integration

```python
# Partial derivative ∂/∂r applied to f = r²
f = R ** 2
d_dr = Diff(grid, 1, 0)   # order=1, axis_index=0
df_dr = d_dr(f)            # ≈ 2r

# Integration
I = Integral(grid)
I(f * R)  # ∫ f r dr dφ
```

For equidistant (non-periodic) axes the optional `acc` parameter controls the
finite-difference accuracy order (default 4):

```python
d_dr_high = Diff(grid, 1, 0, acc=6)  # 6th-order finite differences
```

Spectral axes (`CHEBYSHEV` and periodic `EQUIDISTANT`) ignore `acc` and always
use their native spectral method.

Every `Diff` operator can also be exported as a sparse matrix, which is useful
for building linear systems (e.g. PDE solves):

```python
D = Diff(grid, 2, 0)   # ∂²/∂r²
D.as_matrix()           # returns a scipy.sparse matrix
```

## Interpolation

```python
inter = Interpolator(grid, f)
inter((0.5, 1.0))  # interpolate at r=0.5, φ=1.0
```

## Curvilinear Grids & Vector Calculus

*numgrids* provides specialised grid classes for the most common curvilinear
coordinate systems.  Each one ships with a **Laplacian**, **gradient**,
**divergence**, and **curl** that correctly account for metric scale factors
and handle coordinate singularities gracefully.

### SphericalGrid *(r, θ, φ)*

```python
grid = SphericalGrid(
    create_axis(AxisType.CHEBYSHEV, 25, 0.1, 5),              # r
    create_axis(AxisType.CHEBYSHEV, 20, 0.1, np.pi - 0.1),    # θ
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2*np.pi),  # φ
)
R, Theta, Phi = grid.meshed_coords
f = R ** 2

lap_f = grid.laplacian(f)             # ∇²f = 6
gr, gt, gp = grid.gradient(f)         # (2r, 0, 0)
div_v = grid.divergence(gr, gt, gp)   # ∇·(∇f) = ∇²f
cr, ct, cp = grid.curl(gr, gt, gp)    # ∇×(∇f) = 0
```

### CylindricalGrid *(r, φ, z)*

```python
grid = CylindricalGrid(
    create_axis(AxisType.CHEBYSHEV, 20, 0.1, 3),
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2*np.pi),
    create_axis(AxisType.CHEBYSHEV, 20, -1, 1),
)
R, Phi, Z = grid.meshed_coords
f = R ** 2 + Z ** 2

grid.laplacian(f)   # = 6
grid.gradient(f)    # (2r, 0, 2z)
```

### PolarGrid *(r, φ)*

```python
grid = PolarGrid(
    create_axis(AxisType.CHEBYSHEV, 30, 0.1, 1),
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 40, 0, 2*np.pi),
)
R, Phi = grid.meshed_coords

grid.laplacian(R * np.cos(Phi))   # = 0  (harmonic function)
grid.gradient(R * np.cos(Phi))    # (cos φ, −sin φ)
grid.curl(R * 0, R)               # scalar z-component = 2
```

Singularities at *r = 0* or *θ = 0, π* are handled automatically —
non-finite values are replaced by zero.

## Boundary Conditions

*numgrids* provides classes for applying Dirichlet, Neumann, and Robin
boundary conditions — both at the array level and at the sparse-system level
for linear PDE solves.

Each non-periodic axis contributes two **faces** (low end and high end).
Access them via `grid.faces`:

```python
from numgrids import *
from numgrids.boundary import apply_bcs
from scipy.sparse.linalg import spsolve
import numpy as np

x_ax = create_axis(AxisType.EQUIDISTANT, 101, 0, 1)
y_ax = create_axis(AxisType.EQUIDISTANT, 101, 0, 1)
grid = Grid(x_ax, y_ax)
X, Y = grid.meshed_coords
```

### Function-level (array modification)

```python
u = np.zeros(grid.shape)
DirichletBC(grid.faces["0_low"], value=1.0).apply(u)           # u = 1 on left
DirichletBC(grid.faces["0_high"],
            value=lambda c: np.sin(np.pi * c[1])).apply(u)     # u = sin(πy) on right
```

### System-level (sparse matrix + RHS)

Build a Laplacian matrix, set up boundary conditions, and solve:

```python
L = Diff(grid, 2, 0).as_matrix() + Diff(grid, 2, 1).as_matrix()
rhs = -2 * np.ones(grid.size)

bcs = [
    DirichletBC(grid.faces["0_low"],  0.0),
    DirichletBC(grid.faces["0_high"], 0.0),
    DirichletBC(grid.faces["1_low"],  0.0),
    NeumannBC(grid.faces["1_high"],   0.0),   # du/dn = 0 on top
]
L, rhs = apply_bcs(bcs, L, rhs)
u = spsolve(L, rhs).reshape(grid.shape)
```

Robin conditions (`a u + b du/dn = g`) are also supported via `RobinBC`.
See the API reference for the full details.

## Save / Load

Grids (and any number of meshed data arrays) can be persisted to NumPy
`.npz` files and loaded back:

```python
from numgrids import save_grid, load_grid

save_grid("simulation.npz", grid, temperature=T, pressure=P)

grid2, data = load_grid("simulation.npz")
T = data["temperature"]
P = data["pressure"]
```

The file stores a compact JSON description of the grid definition (axis
types, parameters) so the grid is fully reconstructed — including its type
(`SphericalGrid`, `PolarGrid`, etc.) and all axis properties.

## Refinement, Coarsening, and Multigrid

Every grid can be refined or coarsened:

```python
grid = Grid(
    create_axis(AxisType.EQUIDISTANT, 20, 0, 1),
    create_axis(AxisType.EQUIDISTANT, 20, 0, 1),
)

fine   = grid.refine()             # double all axis resolutions
coarse = grid.coarsen()            # halve all axis resolutions
fine_r = grid.refine_axis(0, 3)    # triple resolution along axis 0 only
```

The `MultiGrid` class builds a full hierarchy of grids automatically and
can transfer meshed functions between levels via interpolation:

```python
mg = MultiGrid(
    create_axis(AxisType.EQUIDISTANT, 32, 0, 1),
    create_axis(AxisType.EQUIDISTANT, 32, 0, 1),
)
# mg.levels[0] is the finest grid, mg.levels[-1] the coarsest

f_fine = mg.levels[0].meshed_coords[0] ** 2
f_coarse = mg.transfer(f_fine, level_from=0, level_to=1)
```

## Adaptive Mesh Refinement

When you don't know in advance how many grid points you need,
*numgrids* can determine the resolution for you. The `adapt` function
iteratively refines the axis that contributes the most discretization
error until a prescribed tolerance is met.

The core idea is **Richardson-extrapolation-style error estimation**:
for each axis, the function is evaluated on a grid refined along that
axis, interpolated back, and compared. The axis with the largest
difference is the resolution bottleneck and gets refined first.

### One-shot error estimation

Use `estimate_error` for a quick diagnostic without running the full
adaptation loop:

```python
from numgrids import *
import numpy as np

grid = Grid(
    create_axis(AxisType.EQUIDISTANT, 10, 0, 1),
    create_axis(AxisType.EQUIDISTANT, 10, 0, 1),
)

def my_func(g):
    X, Y = g.meshed_coords
    return np.sin(10 * X) * Y ** 2

result = estimate_error(grid, my_func)
print(result["global"])      # scalar error estimate
print(result["per_axis"])    # {0: ..., 1: ...}
```

### Automatic adaptation

The `adapt` function runs the estimation–refinement loop until the
global error drops below a tolerance:

```python
grid = Grid(
    create_axis(AxisType.EQUIDISTANT, 10, 0, 1),
    create_axis(AxisType.EQUIDISTANT, 10, 0, 1),
)

result = adapt(grid, my_func, tol=1e-4)

print(result.converged)      # True if tolerance was met
print(result.grid.shape)     # final resolution per axis
print(result.global_error)   # error on the final grid
print(result.iterations)     # number of refinement steps
```

Because *numgrids* uses tensor-product grids, refinement is
**per-axis** rather than per-cell. This keeps all existing operators
(differentiation, integration, interpolation) working unchanged while
still allowing anisotropic resolution — e.g. fine radial resolution with
coarse angular resolution.

Key parameters of `adapt`:

| Parameter | Default | Description |
|---|---|---|
| `tol` | `1e-6` | Target error tolerance |
| `norm` | `"max"` | Error norm (`"max"`, `"l2"`, or `"mean"`) |
| `max_iterations` | `20` | Safety limit on refinement steps |
| `max_points_per_axis` | `1024` | Cap on per-axis resolution |
| `refinement_factor` | `2.0` | Factor by which to increase resolution |
| `refine_all` | `False` | Refine all underresolved axes at once instead of only the worst |

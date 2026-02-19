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

## Refinement and Coarsening

The `MultiGrid` class lets you define a hierarchy of grids at different
resolutions and transfer meshed functions between them via interpolation.
See the API reference for details.

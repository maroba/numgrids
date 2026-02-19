# Boundary conditions

numgrids provides three boundary condition types -- Dirichlet, Neumann, and
Robin -- that can be applied either directly to array data or to sparse
linear systems built from differentiation matrices. This makes it
straightforward to set up and solve boundary value problems.

```python
from numgrids import *
import numpy as np
```

## Boundary faces

Before applying a boundary condition you need a `BoundaryFace`, which
identifies one side of the grid boundary. The easiest way to get them is
through `grid.faces`:

```python
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
)

print(grid.faces.keys())
# dict_keys(['0_low', '0_high', '1_low', '1_high'])
```

The key format is `"{axis_index}_{side}"`, where `side` is either `"low"`
(first point along that axis) or `"high"` (last point). Periodic axes have
no boundary faces and are automatically excluded.

Each `BoundaryFace` exposes:

- `mask` -- a boolean array of shape `grid.shape`, `True` on the face
- `flat_indices` -- 1D indices of the face points in the ravelled grid
- `normal_sign` -- `-1` for `"low"`, `+1` for `"high"` (outward normal direction)

## DirichletBC

A Dirichlet condition prescribes the function value on a boundary face:
$u = g$.

### Applying to arrays

The `apply()` method sets the boundary values in-place:

```python
face = grid.faces["0_low"]
bc = DirichletBC(face, value=0.0)

u = np.ones(grid.shape)
bc.apply(u)
# u is now 0.0 along the x=0 boundary
```

### Applying to a linear system

The `apply_to_system()` method modifies a sparse matrix and right-hand-side
vector. Boundary rows in the matrix are replaced with identity rows, and the
corresponding RHS entries are set to $g$:

```python
D_xx = Diff(grid, order=2, axis_index=0).as_matrix()
D_yy = Diff(grid, order=2, axis_index=1).as_matrix()
L = D_xx + D_yy                       # Laplacian matrix
rhs = np.zeros(grid.size)

L, rhs = bc.apply_to_system(L, rhs)   # returns modified copies
```

## NeumannBC

A Neumann condition prescribes the outward normal derivative:
$\partial u / \partial n = g$.

### Applying to arrays

The `apply()` method adjusts the boundary layer using a first-order
finite-difference approximation of the normal derivative:

```python
face = grid.faces["0_high"]
bc = NeumannBC(face, value=1.0)
bc.apply(u)
```

### Applying to a linear system

For higher accuracy, use `apply_to_system()`. It replaces boundary rows with
the differentiation-matrix rows scaled by the outward normal sign:

```python
L, rhs = bc.apply_to_system(L, rhs)
```

## RobinBC

A Robin condition is a linear combination of Dirichlet and Neumann:
$a\,u + b\,\partial u / \partial n = g$.

```python
face = grid.faces["1_high"]
bc = RobinBC(face, a=1.0, b=0.5, value=2.0)
```

```{warning}
Robin boundary conditions support only `apply_to_system()`, not `apply()`.
Calling `apply()` raises `NotImplementedError`.
```

```python
L, rhs = bc.apply_to_system(L, rhs)
```

## Applying multiple BCs at once

The `apply_bcs()` function takes a list of boundary conditions and applies
them to a linear system in order. For overlapping points (e.g. corners), the
last condition in the list wins:

```python
bcs = [
    DirichletBC(grid.faces["0_low"], value=0.0),
    DirichletBC(grid.faces["0_high"], value=0.0),
    NeumannBC(grid.faces["1_low"], value=0.0),
    NeumannBC(grid.faces["1_high"], value=0.0),
]

L, rhs = apply_bcs(bcs, L, rhs)
```

## Callable boundary values

All BC types accept a callable for `value`. The callable receives a tuple of
coordinate arrays restricted to the boundary face and must return an array:

```python
# u = sin(pi * y) on the left boundary
bc = DirichletBC(
    grid.faces["0_low"],
    value=lambda coords: np.sin(np.pi * coords[1]),
)
```

Here `coords[1]` is the y-coordinate array along the face. This is useful
for spatially varying boundary data.

## Worked example: solving the Poisson equation

Solve $\nabla^2 u = -2\pi^2 \sin(\pi x)\sin(\pi y)$ on $[0,1]^2$ with
$u = 0$ on all boundaries. The exact solution is
$u(x, y) = \sin(\pi x)\sin(\pi y)$.

```python
from numgrids import *
import numpy as np
from scipy.sparse.linalg import spsolve

# Build grid
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
)
X, Y = grid.meshed_coords

# Build Laplacian operator as sparse matrix
D_xx = Diff(grid, order=2, axis_index=0).as_matrix()
D_yy = Diff(grid, order=2, axis_index=1).as_matrix()
L = D_xx + D_yy

# Right-hand side
rhs = (-2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)).ravel()

# Dirichlet u=0 on all four boundaries
bcs = [DirichletBC(face, value=0.0) for face in grid.faces.values()]
L, rhs = apply_bcs(bcs, L, rhs)

# Solve
u = spsolve(L, rhs).reshape(grid.shape)

# Check error
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
print(f"Max error: {np.max(np.abs(u - u_exact)):.2e}")
```

The Chebyshev spectral method should give an error on the order of $10^{-10}$
or smaller with 30 points per side.

```{tip}
For mixed boundary conditions, simply put different BC types in the list.
For example, Dirichlet on the left and right, Neumann on top and bottom:

    bcs = [
        DirichletBC(grid.faces["0_low"], value=0.0),
        DirichletBC(grid.faces["0_high"], value=1.0),
        NeumannBC(grid.faces["1_low"], value=0.0),
        NeumannBC(grid.faces["1_high"], value=0.0),
    ]
```

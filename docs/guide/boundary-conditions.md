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
no boundary faces and are automatically excluded; so are ends that are not a
boundary, such as the ends at infinity of
[unbounded axes](axes.md#unbounded-domains), where no condition is imposed.
For named axes, the key
`"{name}_{side}"` works as well:

```python
grid = Grid(x=chebyshev(30, 0.0, 1.0), y=chebyshev(30, 0.0, 1.0))
grid.faces["x_low"] is grid.faces["0_low"]   # True
```

Each `BoundaryFace` exposes:

- `mask` -- a boolean array of shape `grid.shape`, `True` on the face
- `flat_indices` -- 1D indices of the face points in the ravelled grid
- `normal_sign` -- `-1` for `"low"`, `+1` for `"high"` (outward normal direction)
- `key`, `shape` and `coords` -- the face key, the shape of face arrays,
  and the meshed coordinates restricted to the face

## Compact boundary condition specifications

Most of the time you do not need to construct `DirichletBC` objects by
hand. [`Operator.solve`](operators.md), `Operator.eig`,
[`solve_nonlinear`](nonlinear.md), [`evolve`](time-evolution.md) and
`apply_bcs(..., grid=grid)` accept compact specifications:

```python
grid = Grid(x=chebyshev(30, 0.0, 1.0), y=chebyshev(30, 0.0, 1.0))
L = grid.laplacian_operator()
rhs = np.ones(grid.shape)

u = L.solve(rhs, bcs=0)                       # u = 0 on all faces
u = L.solve(rhs, bcs=lambda c: c[0] * c[1])   # u = x*y on all faces
u = L.solve(rhs, bcs={
    "all": 0,                                 # default for all faces ...
    "x_high": 1.0,                            # ... overridden: u = 1
    "y_low": Neumann(0.0),                    # du/dn = 0
    "y_high": Robin(a=1.0, b=0.5, value=0.0), # u + 0.5 du/dn = 0
})
```

Plain values (numbers, arrays, callables) mean Dirichlet conditions;
`Dirichlet`, `Neumann` and `Robin` are *unbound* specs that are attached to
the face given by the dictionary key. `resolve_bcs(grid, spec)` turns any
such specification into a list of bound conditions.

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

The coefficients `a` and `b` may vary along the face. Like the value, they
can be arrays matching the face or functions of the face coordinates. An
example is Newton's law of cooling, $h\,u + k\,\partial u/\partial n = h\,u_\infty$,
at a surface whose conductivity $k$ depends on the position:

```python
bcs = {"y_high": Robin(2.0, lambda c: 1 + c[0], 0.0)}      # k = 1 + x along the face y = y_max
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

## Working with boundary faces

Faces also provide operations for post-processing, e.g. fluxes and
surface integrals:

```python
X, Y = grid.meshed_coords
f = X**2 * Y

face = grid.faces["x_high"]
face.restrict(f)              # values on the face, shape face.shape
face.normal_derivative(f)     # outward normal derivative on the face
face.integrate(f)             # surface integral
face.flux(*grid.gradient(f))  # outward flux of a vector field

# Gauss's theorem: total outward flux = volume integral of the divergence
sum(face.flux(*grid.gradient(f)) for face in grid.faces.values())
```

On curvilinear grids, `normal_derivative` returns the physical derivative
$\pm(1/h_i)\,\partial f/\partial q_i$ and `integrate` includes the
surface element (e.g. $r^2\sin\theta$ on a sphere).

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

With the [Operator](operators.md) API, the same problem is three lines:

```python
grid = Grid(x=chebyshev(30, 0.0, 1.0), y=chebyshev(30, 0.0, 1.0))
rhs = lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
u = grid.laplacian_operator().solve(rhs, bcs=0)
```

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

# Operators and solvers

Linear differential operators are the heart of most grid-based PDE solvers.
numgrids represents them as `Operator` objects: sparse matrices that know
their grid, can be combined with ordinary arithmetic, and can solve
boundary value and eigenvalue problems directly.

```python
from numgrids import *
import numpy as np
```

## Creating operators

```python
grid = Grid(x=chebyshev(30, 0, 1), y=chebyshev(30, 0, 1))

Dx = grid.diff_operator(1, "x")      # d/dx
Dyy = grid.diff_operator(2, "y")     # d^2/dy^2
L = grid.laplacian_operator()        # Laplacian (also on curvilinear grids)
I = Operator.identity(grid)
```

`Diff` objects work as well -- any arithmetic with them yields an
`Operator`:

```python
L = Diff(grid, 2, "x") + Diff(grid, 2, "y")
```

Apply an operator like a function, or get its sparse matrix:

```python
X, Y = grid.meshed_coords
f = np.sin(X) * Y**2
L(f)                 # array of shape grid.shape
L.as_matrix()        # scipy.sparse CSR matrix of shape (grid.size, grid.size)
```

## Operator algebra

| Expression        | Meaning                                         |
|-------------------|-------------------------------------------------|
| `A + B`, `A - B`  | sum / difference of operators                   |
| `A + c`           | `A + c*I` for a scalar `c`                      |
| `c * A`, `A / c`  | scaling                                         |
| `A + V`           | `A + diag(V)` for an array `V` of grid shape    |
| `V * A`           | apply `A`, then multiply by `V`                 |
| `A * V`           | multiply by `V`, then apply `A`                 |
| `A * B`, `A @ B`  | composition (apply `B` first)                   |
| `A ** n`          | `n`-fold composition                            |

This makes it easy to write down operators with variable coefficients:

```python
x = grid.coords
# Schroedinger operator for the harmonic oscillator
grid = Grid(x=chebyshev(80, -10, 10))
x = grid.coords
H = -0.5 * grid.diff_operator(2) + 0.5 * x**2
```

## Boundary value problems: `solve`

`L.solve(rhs, bcs)` solves $L u = \text{rhs}$ with the given boundary
conditions and returns `u` with the grid's shape:

```python
# Poisson equation  u_xx + u_yy = -2 pi^2 sin(pi x) sin(pi y),  u = 0 on the boundary
grid = Grid(x=chebyshev(30, 0, 1), y=chebyshev(30, 0, 1))
L = grid.laplacian_operator()
u = L.solve(lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y), bcs=0)
```

The right-hand side may be a scalar, an array of grid shape, or a callable
evaluated on the meshed coordinates. Boundary conditions accept the
[compact forms](boundary-conditions.md#compact-boundary-condition-specifications):

```python
u = L.solve(0.0, bcs={
    "x_low": 0.0,
    "x_high": lambda c: np.sin(np.pi * c[1]),
    "y_low": Neumann(0.0),
    "y_high": Neumann(0.0),
})
```

For fully periodic grids no boundary conditions are needed (as long as the
operator is invertible, e.g. a Helmholtz operator):

```python
grid = Grid(periodic(64))
x = grid.coords
u = (grid.diff_operator(2) - 1).solve(-2 * np.sin(x))   # u = sin(x)
```

If you want to call a solver yourself, `L.with_bcs(rhs, bcs)` returns the
modified sparse system `(matrix, rhs_vector)`.

## Eigenvalue problems: `eig`

`L.eig(k, bcs)` computes `k` eigenpairs of $L u = \lambda u$:

```python
grid = Grid(x=chebyshev(80, -10, 10))
x = grid.coords
H = -0.5 * grid.diff_operator(2) + 0.5 * x**2

energies, states = H.eig(k=5, bcs=0)
print(energies)        # [0.5 1.5 2.5 3.5 4.5]
states.shape           # (5, 80): states[n] belongs to energies[n]
```

Boundary conditions are treated as homogeneous. Unlike the naive approach
of replacing matrix rows, `eig` produces **no spurious eigenvalues**:
Dirichlet points are eliminated from the problem, and Neumann/Robin rows
turn it into a generalized eigenvalue problem $L u = \lambda B u$.

Selecting eigenvalues:

| Argument        | Returns                                    |
|-----------------|--------------------------------------------|
| `which="SM"`    | smallest magnitude (default)               |
| `which="LM"`    | largest magnitude                          |
| `which="SR"`    | smallest real part                         |
| `which="LR"`    | largest real part                          |
| `sigma=s`       | the `k` eigenvalues closest to `s`         |

Problems with up to `dense_threshold` (default 2000) unknowns are solved
with dense LAPACK routines; larger problems use sparse shift-invert Arnoldi
iterations. Eigenvectors are normalized to unit Euclidean norm with a
positive largest component; normalize them with `grid.integrate` if you
need $\int |u|^2\,dV = 1$.

Example: vibration modes of a square drum $[0, \pi]^2$ with fixed rim,
whose eigenvalues are $m^2 + n^2$:

```python
grid = Grid(x=chebyshev(30, 0, np.pi), y=chebyshev(30, 0, np.pi))
vals, modes = (-grid.laplacian_operator()).eig(k=6, bcs=0)
vals                   # [2, 5, 5, 8, 10, 10]
grid.plot(modes[1])
```

The same works on curvilinear grids, e.g. for an annular drum:

```python
grid = PolarGrid(chebyshev(30, 0.5, 1.0), periodic(40))
vals, modes = (-grid.laplacian_operator()).eig(k=6, bcs=0)
grid.plot(modes[0])
```

```{note}
Grid points *on* a coordinate singularity (such as $r = 0$) yield empty
operator rows. `solve` raises an error and `eig` warns about spurious zero
eigenvalues in that case: start the axis slightly away from the singularity
and impose a suitable boundary condition there.
```

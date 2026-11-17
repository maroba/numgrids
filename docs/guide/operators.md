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

`A.T` is the transposed operator, `A.H` its conjugate transpose, and
`A.adjoint()` the adjoint with respect to the $L^2$
[inner product](integration.md#inner-products-and-norms) of the grid.
`A.toarray()` returns the matrix as a dense array.

This makes it easy to write down operators with variable coefficients:

```python
x = grid.coords
# Schroedinger operator for the harmonic oscillator
grid = Grid(x=chebyshev(80, -10, 10))
x = grid.coords
H = -0.5 * grid.diff_operator(2) + 0.5 * x**2
```

### Gradient, divergence and operators in divergence form

`grid.gradient_operator()` and `grid.divergence_operator()` return one
operator per axis. On curvilinear grids they include the scale factors,
like `grid.gradient` and `grid.divergence`, and they also work across the
center of a disk or ball. Composed, they give operators in divergence form
such as $\nabla\cdot(k\,\nabla u)$ for heat conduction or diffusion with a
variable coefficient:

```python
grid = Grid(x=chebyshev(30, 0, 1), y=chebyshev(30, 0, 1))
X, Y = grid.meshed_coords
k = 1 + X * Y                                   # conductivity

G = grid.gradient_operator()                    # (d/dx, d/dy)
D = grid.divergence_operator()
L = sum(Di @ (k * Gi) for Di, Gi in zip(D, G))  # u -> div(k grad u)
```

For this common case there is a shortcut, `grid.diffusion_operator(k)`,
where `k` is a number, an array or a function of the coordinates:

```python
L = grid.diffusion_operator(lambda x, y: 1 + x * y)
u = L.solve(-1.0, bcs=0)                        # steady temperature of a heated plate
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

### How the system is solved

`solve` picks the fastest of three direct methods; `method=` selects one
explicitly:

| `method`    | What it does                                                    | Best for |
|-------------|-----------------------------------------------------------------|----------|
| `"fourier"` | Fourier transform along the periodic axes that neither the operator nor the boundary conditions depend on; one small system per Fourier mode | disks, cylinders, spheres, balls, periodic boxes |
| `"dense"`   | dense LU factorization                                          | spectral grids of moderate size |
| `"sparse"`  | sparse LU factorization                                         | finite differences |
| `"auto"`    | the first of these that applies (default)                       | |

Spectral differentiation matrices are dense along each axis, so sparse LU
factorizations fill in almost completely on them and dense LU is several
times faster. The Fourier decoupling is far faster still. On a full ball,
the Laplacian does not depend on the azimuthal angle, so the problem splits
into independent problems in $(r, \theta)$:

```python
ball = SphericalGrid(radial(16, 1.0), colatitude(16), periodic(32))   # 8192 unknowns
u = ball.laplacian_operator().solve(1.0, bcs=0)                       # a fraction of a second
```

The boundary *values* may depend on the angle; only the operator and the
type of the boundary conditions must not. If a coefficient depends on the
angle, `"auto"` falls back to the other methods.

Singular systems are solved in the least-squares sense. The typical case
is a Poisson problem with Neumann conditions on the whole boundary, whose
solution is only determined up to a constant: `solve` then returns one of
the solutions (provided the data are compatible), and you can fix the
constant afterwards, e.g. with `u - grid.mean(u)`.

### Many right-hand sides: `factorize`

`L.factorize(bcs)` returns a solver that keeps the factorization. Calling
it with a right-hand side costs only a back substitution. This is what
implicit time stepping schemes, iterations and parameter studies need:

```python
grid = Grid(x=chebyshev(40, 0, 1))
x = grid.coords
dt = 0.01
step = (1 - dt * grid.laplacian_operator()).factorize(bcs=0)   # implicit Euler for u_t = u_xx

u = np.sin(np.pi * x)
for _ in range(100):
    u = step(u)
```

The boundary values are fixed when the solver is created; `step.method`
tells which method it uses.

## Eigenvalue problems: `eig`

`L.eig(k, bcs)` computes `k` eigenpairs of $L u = \lambda u$ (or of
$L u = \lambda M u$, see [below](#generalized-eigenvalue-problems)):

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

Problems that do not depend on a periodic axis (the azimuthal angle of a
disk, cylinder, sphere or ball) are split into one small eigenproblem per
Fourier mode, as in `solve`. This is fast and yields all eigenvalues to
full accuracy. Otherwise, problems with up to `dense_threshold` (default
2000) unknowns are solved with dense LAPACK routines, and larger problems
with sparse shift-invert Arnoldi iterations. `method="direct"` switches the
Fourier decoupling off.

Eigenvectors have a positive largest component and unit Euclidean norm (of
the array of values). `normalize="l2"` normalizes them on the grid instead,
$\int |u|^2\,dV = 1$, as needed for wave functions, and `normalize="max"`
scales the largest value to 1:

```python
energies, states = H.eig(k=5, bcs=0, normalize="l2")
grid.norm(states[0])   # 1.0
```

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
eigenvalues in that case. For full disks, cylinders and balls use a radial
axis instead, which handles the center without a singularity (see
[Curvilinear grids](curvilinear.md#full-disks-cylinders-spheres-and-balls)).
```

A circular drum:

```python
grid = PolarGrid(radial(24, 1.0), periodic(32))
vals, modes = (-grid.laplacian_operator()).eig(k=6, bcs=0)
np.sqrt(vals)     # 2.4048, 3.8317, 3.8317, 5.1356, 5.1356, 5.5201 (Bessel zeros)
```

For real operators, the eigenvectors are real, even for degenerate
eigenvalues such as the pairs above. Each degenerate eigenspace comes as a
real orthonormal basis.

### Generalized eigenvalue problems

Many physical eigenvalue problems have the form $L u = \lambda M u$, e.g.
a string or a membrane with a varying mass density $\rho$:
$-\nabla^2 u = \lambda \rho u$. Pass $M$ to `eig`. It can be an array of
grid shape or a function of the coordinates (multiplication by that
function), an `Operator`, or a matrix:

```python
# A string on [1, e] with density 1/x^2: eigenvalues 1/4 + n^2 pi^2
grid = Grid(x=chebyshev(40, 1, np.e))
vals, modes = (-grid.diff_operator(2)).eig(k=3, bcs=0, M=lambda x: 1 / x**2)
vals                   # [10.1196, 39.7284, 89.0764]
```

A circular drum whose density decreases from the center to the rim as
$\rho = 4 / (1 + r^2)^2$ sounds like a hemisphere. Stereographic projection
maps it onto one, and the clamped rim becomes the equator. Its eigenvalues
are $l(l+1)$:

```python
grid = PolarGrid(radial(24, 1.0), periodic(32))
R, _ = grid.meshed_coords
vals, modes = (-grid.laplacian_operator()).eig(k=6, bcs=0, M=4 / (1 + R**2)**2)
vals                   # [2, 6, 6, 12, 12, 12]
```

Boundary conditions and the selection of eigenvalues (`which`, `sigma`)
work as for the standard problem.

For **nonlinear** boundary value problems, see
[Nonlinear problems](nonlinear.md).

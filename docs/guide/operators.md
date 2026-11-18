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

## Functions of operators

An operator $L$ with the eigenpairs $L \varphi_n = \lambda_n \varphi_n$
defines the operator $f(L)$ for any scalar function $f$: it has the same
eigenfunctions and the eigenvalues $f(\lambda_n)$, so it maps
$u = \sum_n c_n \varphi_n$ to $\sum_n f(\lambda_n)\, c_n \varphi_n$. The
eigenfunctions depend on the boundary conditions, and so does $f(L)$.

| Method                 | Operator                                             |
|------------------------|------------------------------------------------------|
| `L.power(s, bcs)`      | $L^s$ for any real $s$, e.g. the fractional Laplacian |
| `L.sqrt(bcs)`          | $L^{1/2}$                                            |
| `L.exp(t, bcs)`        | $e^{tL}$, the propagator of $u_t = L u$              |
| `L.inverse(bcs)`       | $L^{-1}$, the Green's operator                       |
| `L.function(f, bcs)`   | $f(L)$ for a function `f` of the eigenvalues         |

The results are operators again. Apply them to functions, combine them
with other operators, or pass them to `solve`, `eig` and `evolve`.

### Fractional Laplacian

On a periodic grid, the Fourier modes are the eigenfunctions of the
Laplacian. $(-\nabla^2)^s$ multiplies the mode with the wave vector
$\mathbf{k}$ by $|\mathbf{k}|^{2s}$:

```python
grid = Grid(periodic(64), periodic(64))
X, Y = grid.meshed_coords
L = -grid.laplacian_operator()

Ls = L.power(0.5)                      # (-Laplacian)^(1/2)
u = np.sin(3 * X) * np.cos(4 * Y)
Ls(u)                                  # 5 * u
```

On a bounded domain, pass the boundary conditions. This gives the
*spectral* fractional Laplacian, built from the eigenfunctions of the
Laplacian with these conditions:

```python
grid = Grid(x=chebyshev(40, 0, 1))
x = grid.coords
L = -grid.laplacian_operator()

Ls = L.power(0.5, bcs=0)               # with Dirichlet conditions
Ls(np.sin(np.pi * x))                  # pi * sin(pi x)

Ls = L.power(0.5, bcs=Neumann(0))      # with Neumann conditions
Ls(np.cos(np.pi * x))                  # pi * cos(pi x)
```

Note the sign: fractional powers need an operator without negative
eigenvalues, so take the power of $-\nabla^2$, not of $\nabla^2$ (which
raises an error).

Negative exponents give powers of the inverse. The fractional Poisson
problem $(-\nabla^2)^s u = f$ is solved by the operator $(-\nabla^2)^{-s}$,
or by `solve`:

```python
f = np.sin(np.pi * x) + np.sin(3 * np.pi * x)
u = L.power(-0.5, bcs=0)(f)            # sin(pi x) / pi + sin(3 pi x) / (3 pi)
u = L.power(0.5, bcs=0).solve(f, bcs=0)
```

The same works on curvilinear grids, e.g. on a disk:

```python
grid = PolarGrid(radial(24, 1.0), periodic(32))
R, Phi = grid.meshed_coords
Ls = (-grid.laplacian_operator()).power(0.5, bcs=0)
u = Ls.solve(1.0, bcs=0)               # (-Laplacian)^(1/2) u = 1 on the unit disk
```

### Boundary conditions

The boundary conditions are homogeneous, as in `eig`: their values are
ignored. The result acts on functions that satisfy the conditions and
returns functions that satisfy them:

- Only the values of the argument at the points *without* a condition
  enter. Its boundary values are not used, since they follow from the
  conditions.
- At these points, the result is $f(L) u$. On the boundary it is zero for
  Dirichlet conditions; for Neumann and Robin conditions it takes the
  values for which the condition holds.

So `L.function(lambda z: z, bcs)` reproduces `L` on functions that satisfy
the conditions, except on the boundary points themselves (where $L u$ does
not satisfy the conditions in general). The functional calculus holds:
`L.power(a, bcs) @ L.power(b, bcs)` is `L.power(a + b, bcs)`, and
`L.sqrt(bcs) @ L.sqrt(bcs)` acts as `L`. When you pass the result to
`solve`, `eig` or `evolve`, pass the same conditions there.

Where $L$ has a zero eigenvalue, negative powers use the pseudo-inverse:
the zero modes are mapped to zero. The Laplacian on a periodic grid or
with Neumann conditions on the whole boundary annihilates the constants,
so its inverse solves the Poisson equation for the part of the right-hand
side with zero mean and returns the solution with zero mean:

```python
grid = Grid(x=periodic(64))
x = grid.coords
G = (-grid.laplacian_operator()).inverse()
G(np.cos(3 * x) + 1)                   # cos(3 x) / 9
```

### Fractional diffusion

Anomalous diffusion $u_t = -(-\nabla^2)^s u$ replaces the Laplacian of the
heat equation by a fractional one. Since the fractional Laplacian is an
operator, `evolve` integrates the equation like any other linear one,
without time stepping error if you choose the exponential integrator:

```python
grid = Grid(x=chebyshev(60, -1, 1))
x = grid.coords
Ls = (-grid.laplacian_operator()).power(0.5, bcs=0)

u0 = np.exp(-50 * x**2)
result = evolve(grid, -Ls, u0, (0, 0.2), bcs=0, method="expm",
                t_eval=np.linspace(0, 0.2, 11))
```

For a single time, the propagator $e^{-t(-\nabla^2)^s}$ does the same:

```python
u = Ls.exp(-0.2, bcs=0)(u0)            # equals result.final
```

`exp` takes complex factors as well, e.g. `H.exp(-1j * t, bcs=0)` for the
propagator $e^{-itH}$ of the Schrödinger equation.

### Square root, inverse and other functions

The square root and the inverse are the powers $1/2$ and $-1$:

```python
grid = Grid(x=chebyshev(40, 0, 1))
x = grid.coords
L = -grid.laplacian_operator()

root = L.sqrt(bcs=0)
(root @ root)(np.sin(np.pi * x))       # pi^2 sin(pi x), like L

G = L.inverse(bcs=0)                   # Green's operator of -u'' = f, u(0) = u(1) = 0
G(np.ones(grid.shape))                 # x (1 - x) / 2
G.toarray()                            # the discrete Green's function
```

`L.function(f, bcs)` takes any function of the eigenvalues. `f` is called
with an array of eigenvalues and returns the array of its values. This
covers pseudo-differential operators such as the relativistic kinetic
energy $\sqrt{p^2 + m^2} - m$ with $p^2 = -d^2/dx^2$:

```python
grid = Grid(x=chebyshev(80, -10, 10))
x = grid.coords
p2 = -grid.diff_operator(2)

T = p2.function(lambda lam: np.sqrt(lam + 1) - 1, bcs=0)     # m = 1
H = T + 0.5 * x**2                     # relativistic harmonic oscillator
energies, states = H.eig(k=3, bcs=0)   # [0.4411, 1.2371, 1.9099]: below 1/2, 3/2, 5/2
```

### Cost and limits

Functions of operators need dense matrices, so they are meant for
problems of moderate size:

- An operator with constant coefficients on a **periodic grid** is
  diagonal in Fourier space. `f` is applied to its Fourier symbol, which
  is exact and costs next to nothing.
- If the operator and the boundary conditions do not depend on a periodic
  axis (the **angles of disks, cylinders, spheres and balls**), the problem
  splits into one small matrix per Fourier mode, as in `solve` and `eig`.
- Otherwise the matrix has as many rows as the grid has points. The cost
  grows with the third power of the size: about a second for 1000 points
  and several seconds for 2000. Larger problems raise an error unless you
  raise the limit `max_size` (default 4000).

In the first two cases the result is applied with fast Fourier transforms
and never needs its matrix, which is dense. Sums, products, multiples and
functions of such operators keep this form. Anything else, e.g. adding a
potential, `solve`, `eig` or `toarray`, builds the matrix, which limits the
grid to about 9000 points.

The functions are evaluated with an eigendecomposition. This is accurate
for operators with well-conditioned eigenvectors, such as Laplacians,
diffusion operators and Schrödinger operators on any grid. For operators
far from normal, e.g. convection-dominated ones, `power`, `sqrt`, `exp`
and `inverse` switch to Schur-based algorithms that stay accurate;
`function` falls back to the less robust Schur-Parlett method and warns if
the estimated error is large.

For **nonlinear** boundary value problems, see
[Nonlinear problems](nonlinear.md).

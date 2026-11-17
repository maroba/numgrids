# Nonlinear problems

Many boundary value problems are nonlinear: reaction terms like $e^u$ or
$u^3$, conductivities that depend on the temperature, equilibria of
reaction-diffusion systems. `solve_nonlinear` finds a solution of
$F(u) = 0$ with Newton's method. You only write down the residual $F$ as an
ordinary function of grid-shaped arrays.

```python
from numgrids import *
import numpy as np
```

## A first example: the Bratu problem

$u'' + \lambda e^u = 0$ on $[0, 1]$ with $u(0) = u(1) = 0$ models, e.g.,
thermal runaway in a reactive slab:

```python
grid = Grid(x=chebyshev(31, 0, 1))
lam = 1.0

u = solve_nonlinear(
    grid,
    lambda u: grid.diff(u, 2) + lam * np.exp(u),   # the residual F(u)
    u0=0,                                          # initial guess
    bcs=0,                                         # u = 0 at both ends
)
u.max()        # 0.14053921...
```

The arguments mirror those of [`solve`](operators.md#boundary-value-problems-solve)
and [`evolve`](time-evolution.md):

- the **residual** `F(u)` returns an array of grid shape; write it with
  `grid.diff`, `grid.laplacian`, operators, NumPy functions, ...;
- the **initial guess** `u0` is a scalar, an array of grid shape or a
  function of the coordinates;
- the **boundary conditions** `bcs` take all the
  [compact forms](boundary-conditions.md) (Dirichlet, Neumann, Robin). On
  boundary points they replace the residual.

Newton stops when the update is smaller than `tol * (1 + max|u|)` (default
`tol=1e-10`). With `verbose=True` it reports the residual after every step:

```python
u = solve_nonlinear(grid, lambda u: grid.diff(u, 2) + np.exp(u), 0, bcs=0, verbose=True)
# Newton step 1: |F| = 1.020e-02, |du| = 1.395e-01, damping = 1
# Newton step 2: |F| = 6.266e-07, |du| = 1.045e-03, damping = 1
# ...
```

Each step is damped (shortened) if the full Newton step would not reduce
the residual, which makes the method robust for poor initial guesses.

## Several solutions: the initial guess decides

Nonlinear problems can have several solutions. Newton's method converges to
one *near* the initial guess. The Bratu problem with $\lambda = 1$ has a
second, much larger solution:

```python
x = grid.coords
upper = solve_nonlinear(grid, lambda u: grid.diff(u, 2) + np.exp(u),
                        u0=4 * np.sin(np.pi * x), bcs=0)
upper.max()    # 4.0915...
```

A good initial guess often is the solution of a nearby problem. To follow
a solution while a parameter changes, use the previous solution as the next
initial guess:

```python
u = np.zeros(grid.shape)
for lam in np.linspace(0.1, 3.5, 35):
    u = solve_nonlinear(grid, lambda u: grid.diff(u, 2) + lam * np.exp(u), u, bcs=0)
```

[`continuation`](#continuation-and-bifurcation-diagrams) automates this and
also finds how the two solutions are connected.

If Newton's method does not converge, `solve_nonlinear` raises a
`NonlinearSolverError`. Its attributes `u` (the last iterate) and
`residuals` (the history) help to find out what went wrong. Beyond
$\lambda \approx 3.51$ the Bratu problem has no solution at all:

```python
try:
    solve_nonlinear(grid, lambda u: grid.diff(u, 2) + 4 * np.exp(u), 0, bcs=0)
except NonlinearSolverError as err:
    print(err.residuals[-3:])
```

## The Jacobian

Newton's method needs the Jacobian $J = \partial F / \partial u$. By
default, `solve_nonlinear` approximates it by finite differences. This
works for any residual, but it costs one evaluation of the residual per
grid point and Newton step. For grids with a few thousand points that is
fine, but for larger problems pass the exact Jacobian. With
[operator algebra](operators.md#operator-algebra) it reads like the
formula: for $F(u) = u'' + \lambda e^u$, the Jacobian is
$J = d^2/dx^2 + \lambda e^u$:

```python
D2 = grid.diff_operator(2)
u = solve_nonlinear(
    grid,
    lambda u: D2(u) + lam * np.exp(u),
    u0=0,
    bcs=0,
    jacobian=lambda u: D2 + lam * np.exp(u),   # Operator + array = Operator + diag
)
```

`jacobian(u)` may return an `Operator`, a SciPy sparse matrix or a dense
array. The rows of the boundary conditions are added automatically.

A nonlinear term inside a derivative works the same way. For steady heat
conduction with a temperature-dependent conductivity,
$\nabla \cdot (k(u) \nabla u) = 0$, here in 1D with $k = 1 + u^2$:

```python
grid = Grid(x=chebyshev(30, 0, 1))
x = grid.coords
u = solve_nonlinear(grid, lambda u: grid.diff((1 + u**2) * grid.diff(u, 1), 1),
                    u0=x, bcs={"x_low": 0, "x_high": 1})
# exact: u + u^3 / 3 = 4 x / 3
```

## Curvilinear grids

Everything works on [curvilinear grids](curvilinear.md). The Bratu problem
on the full unit disk, $\nabla^2 u + e^u = 0$ with $u = 0$ on the rim, has
the exact solution $u = \ln\bigl(8\mu / (1 + \mu r^2)^2\bigr)$ with
$\mu = 3 - 2\sqrt{2}$:

```python
disk = PolarGrid(radial(16, 1.0), periodic(16))
L = disk.laplacian_operator()
u = solve_nonlinear(disk, lambda u: L(u) + np.exp(u), 0, bcs=0,
                    jacobian=lambda u: L + np.exp(u))

R, _ = disk.meshed_coords
mu = 3 - 2 * np.sqrt(2)
np.max(np.abs(u - np.log(8 * mu / (1 + mu * R**2)**2)))   # ~ 1e-14
```

## Coupled fields

For systems, pass a tuple of initial guesses. The residual then receives
and returns one array per field. `bcs` takes one specification per field
(`None` for none) or a single one for all fields:

```python
grid = Grid(x=chebyshev(30, 0, 1))
D2 = grid.diff_operator(2)

# u'' + e^v = 0,  v = u  (the Bratu problem, written as a system)
u, v = solve_nonlinear(
    grid,
    lambda u, v: (D2(u) + np.exp(v), v - u),
    (0, 0),
    bcs=(0, None),
)
```

An exact Jacobian for a system is a nested list of blocks,
`J[i][j]` $= \partial F_i / \partial u_j$. Each block is an `Operator`, a
matrix, an array of grid shape (multiplication), a scalar, or `None` for
zero:

```python
u, v = solve_nonlinear(
    grid,
    lambda u, v: (D2(u) + np.exp(v), v - u),
    (0, 0),
    bcs=(0, None),
    jacobian=lambda u, v: [[D2, np.exp(v)],
                           [-1, 1]],
)
```

## Continuation and bifurcation diagrams

How do the solutions change with a parameter, and how many are there?
`continuation` follows a branch of solutions of $F(u, \lambda) = 0$,
starting from the solution for one value of the parameter. The residual
takes the parameter as its last argument. For the Bratu problem
$u'' + \lambda e^u = 0$:

```python
grid = Grid(x=chebyshev(31, 0, 1))
D2 = grid.diff_operator(2)

branch = continuation(
    grid,
    lambda u, lam: D2(u) + lam * np.exp(u),         # the residual F(u, lam)
    u0=0,                                           # initial guess for the first solution
    param=0,                                        # the parameter to start from
    bcs=0,
    jacobian=lambda u, lam: D2 + lam * np.exp(u),   # optional, as for solve_nonlinear
    ds=0.05,                                        # step along the branch
    max_steps=150,
)
branch.params        # the parameters along the branch, shape (151,)
branch.u             # the solutions, shape (151, 31)
```

A *bifurcation diagram* shows one number per solution against the
parameter. `branch.measure(func)` evaluates a function on every solution:

```python
import matplotlib.pyplot as plt

plt.plot(branch.params, branch.measure(np.max))
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\max u$")
```

The curve starts at $u = 0$, runs up to $\lambda_c \approx 3.5138$ and
*turns back* there: the small and the large solution from
[above](#several-solutions-the-initial-guess-decides) are the lower and the
upper part of a single branch, which meet in a *fold*. Beyond the fold there
is no solution. `branch.folds` holds the indices of the points where the
parameter turns around:

```python
k = branch.folds[0]
branch.params[k]             # 3.5135  (exact: 3.51383...)
branch.measure(np.max)[k]    # 1.17...
```

Stepping in the parameter, as in the loop above, cannot pass a fold: the
Jacobian is singular there, and behind it the parameter decreases again.
`continuation` therefore steps along the curve $(u(s), \lambda(s))$ itself
(*pseudo-arclength continuation*): it predicts the next point along the
tangent of the curve and corrects it with Newton's method, in which the
parameter is an unknown like the solution.

The step `ds` is measured in the arclength
$ds^2 = \overline{du^2} + d\lambda^2$ (with the mean over the grid points,
so that it does not depend on the resolution). It is halved automatically
where Newton's method does not converge and recovers afterwards; with
`ds_max` larger than `ds`, the steps also grow on the easy parts of a
branch. The continuation stops

- after `max_steps` steps,
- when the parameter leaves `param_range=(low, high)`; the last point then
  lies exactly on the end of the range,
- when `callback(u, lam)`, which is called for every point, returns `True`,
- or when the step size falls below `ds_min`. In this case
  `branch.success` is `False`, `branch.message` tells where it happened,
  and the branch up to that point is kept.

`direction=-1` starts towards decreasing parameters. Starting on the lower
branch at $\lambda = 2$, the continuation arrives at the large solution for
$\lambda = 1$ by itself, without a clever initial guess:

```python
branch = continuation(grid, lambda u, lam: D2(u) + lam * np.exp(u), u0=0, param=2,
                      bcs=0, param_range=(1, 4), max_steps=200)
branch.params[-1]     # 1.0
branch.u[-1].max()    # 4.0915...
```

Boundary conditions and coupled fields work as for `solve_nonlinear`: with
a tuple of initial guesses, the residual is `residual(u, v, ..., lam)` and
`branch.u` is a tuple with one array per field.

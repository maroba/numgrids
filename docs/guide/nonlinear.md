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
branch.params[k]             # 3.5135, the largest parameter among the points
branch.measure(np.max)[k]    # 1.17...
```

The fold itself lies between two points of the branch. `continuation`
locates it there: it solves for the point of the branch where the parameter
is stationary, $d\lambda / ds = 0$. The result is as accurate as the
solutions, however large the step `ds` is:

```python
branch.fold_params           # array([3.51383072])   (exact: 3.513830719...)
branch.fold_u                # the solutions at the folds, shape (1, 31)
branch.fold_u[0].max()       # 1.18684...
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

## Stability of steady states

A solution of $F(u) = 0$ is a steady state of the evolution equation
$u_t = F(u)$ (which [`evolve`](time-evolution.md) integrates in time). It
is only observed if it is *stable*: if small perturbations decay. A
perturbation $e^{\sigma t} v(x)$ of the steady state $u$ solves the
linearized equation,

$$
F'(u)\, v = \sigma v ,
$$

so the eigenvalues $\sigma$ of the Jacobian decide: the steady state is
stable if all of them have a negative real part. `linear_stability`
computes the leading eigenvalues, those with the largest real part. It
takes the residual, the solution and the boundary conditions as they were
passed to `solve_nonlinear`. Of the two solutions of the Bratu problem for
$\lambda = 1$, only the small one is stable:

```python
grid = Grid(x=chebyshev(31, 0, 1))
x = grid.coords
F = lambda u: grid.diff(u, 2) + np.exp(u)

lower = solve_nonlinear(grid, F, 0, bcs=0)
upper = solve_nonlinear(grid, F, 4 * np.sin(np.pi * x), bcs=0)

linear_stability(grid, F, lower, bcs=0)
# StabilityResult(stable, growth rate -8.7397)

result = linear_stability(grid, F, upper, bcs=0)
# StabilityResult(unstable, 1 unstable mode, growth rate 28.873)
```

The result holds

- `eigenvalues`: the `k` leading eigenvalues (default `k=6`), largest real
  part first; complex ones come in conjugate pairs,
- `eigenfunctions`: the corresponding modes $v$, shape `(k, *grid.shape)`,
  normalized to unit Euclidean norm,
- `stable`, the `growth_rate` (the largest real part) and `n_unstable`, the
  number of growing modes.

```python
result.eigenvalues          # array([  28.87..., -20.56..., -68.28..., ...])
result.eigenfunctions[0]    # the growing mode
```

Perturbations satisfy the *homogeneous* boundary conditions: they vanish
where the solution is prescribed, and their normal derivative vanishes
where that of the solution is. An exact `jacobian` is passed as for
`solve_nonlinear`; without it, the Jacobian is approximated by finite
differences, which limits the accuracy of the eigenvalues to about six
digits.

```{warning}
The eigenvalues belong to the evolution equation $u_t = +F(u)$. The
residuals $F$ and $-F$ have the same solutions, but opposite stability. If
your residual is the negative of the right-hand side of the evolution
equation (e.g. $F(u) = -u'' - \lambda e^u$), pass `sign=-1`.
```

### Coupled fields and constraints

For coupled systems, pass the tuple of fields. `eigenfunctions` then is a
tuple with one array per field. Not every equation of a system needs to
have a time derivative. `mass` tells which ones do: it is the operator $M$
of the evolution equation $M u_t = F(u)$, given as one entry per field (a
number, an array of grid shape or an operator; `0` for an equation without
time derivative). The Bratu problem as a system with the algebraic
constraint $v = u$:

```python
D2 = grid.diff_operator(2)

result = linear_stability(
    grid,
    lambda u, v: (D2(u) + np.exp(v), v - u),    # u_t = u'' + e^v,  0 = v - u
    (lower, lower),
    bcs=(0, None),
    jacobian=lambda u, v: [[D2, np.exp(v)], [-1, 1]],
    mass=(1, 0),
)
result.growth_rate    # -8.7397..., as for the single equation
```

Problems with more than 2000 unknowns are solved with shift-invert Arnoldi
iterations, which find the eigenvalues closest to a shift `sigma` (default
0). This is where eigenvalues change their sign, but an unstable
eigenvalue far from the origin can be missed; choose `sigma` accordingly.

## Stability and bifurcations along a branch

With `stability=k`, `continuation` computes the `k` leading eigenvalues at
every point of the branch:

```python
D2 = grid.diff_operator(2)
F = lambda u, lam: D2(u) + lam * np.exp(u)
J = lambda u, lam: D2 + lam * np.exp(u)

branch = continuation(grid, F, u0=0, param=0, bcs=0, jacobian=J,
                      ds=0.05, max_steps=150, stability=4)
branch
# ContinuationResult(151 points, parameter from 0 to 0.437245, 1 fold(s), 74 stable point(s), success=True)

branch.eigenvalues    # shape (151, 4), largest real part first
branch.stable         # True for the points with a stable solution
branch.n_unstable     # the number of growing modes: 0, ..., 0, 1, ..., 1
```

In a bifurcation diagram, stable solutions are usually drawn with solid
lines and unstable ones with dashed lines:

```python
max_u = branch.measure(np.max)
plt.plot(np.where(branch.stable, branch.params, np.nan), max_u, "-")
plt.plot(np.where(branch.stable, np.nan, branch.params), max_u, "--")
plt.plot(branch.fold_params, branch.fold_u.max(axis=1), "o")
```

The lower part of the Bratu branch is stable, the upper part has one
growing mode: the leading eigenvalue crosses zero exactly at the fold.

The points of a branch where its stability changes, where eigenvalues cross
the imaginary axis, are its *bifurcations*. `branch.bifurcations` lists
them in the order in which they occur:

```python
branch.bifurcations
# [Bifurcation(fold at param = 3.513830719, between the points 73 and 74)]

fold = branch.bifurcations[0]
fold.kind       # "fold"
fold.param      # 3.5138307191...
fold.u          # the solution at the fold
fold.index      # 73: the fold lies between the points 73 and 74
```

There are three kinds:

- `"fold"`: the branch turns back in the parameter; a real eigenvalue is
  zero there. Folds are found from the shape of the branch, also without
  `stability`.
- `"branch_point"`: a real eigenvalue crosses zero away from a fold.
  Another branch of solutions crosses the branch there (a transcritical or
  pitchfork bifurcation).
- `"hopf"`: a complex conjugate pair of eigenvalues $\pm i\omega$ crosses
  the imaginary axis. The steady state gives way to an oscillation with the
  angular frequency `frequency` $= \omega$.

Every bifurcation is located by solving for points of the branch between
its two neighbours, until the eigenvalue (or $d\lambda / ds$ for a fold)
vanishes. `mode` holds the critical eigenfunction. With `refine=False`,
the bifurcations are only interpolated between the points of the branch,
which saves a few solves for each of them.

### Branch points: a buckling rod

A rod under the load $\lambda$ obeys $u'' + \lambda \sin u = 0$ with
$u(0) = u(1) = 0$, where $u$ is the angle of its deflection. The straight
rod, $u = 0$, is a solution for every load:

```python
F = lambda u, lam: D2(u) + lam * np.sin(u)
J = lambda u, lam: D2 + lam * np.cos(u)

straight = continuation(grid, F, u0=0, param=0, bcs=0, jacobian=J,
                        ds=1.0, param_range=(0, 50), stability=4)
straight.bifurcations
# [Bifurcation(branch_point at param = 9.869604401, between the points 9 and 10),
#  Bifurcation(branch_point at param = 39.4784176, between the points 39 and 40)]
```

It loses its stability at $\lambda = \pi^2$, the buckling load of Euler,
and a second mode becomes unstable at $4\pi^2$. At a branch point, another
branch of solutions starts along the critical mode. To follow it, start a
continuation at the branch point with `tangent`:

```python
bp = straight.bifurcations[0]
buckled = continuation(grid, F, u0=bp.u, param=bp.param, bcs=0, jacobian=J,
                       tangent=bp.mode, ds=0.2, max_steps=40, stability=4)
buckled.u[-1].max()        # 2.02...: the rod is strongly bent
buckled.stable[1:].all()   # True: the buckled rod is stable
```

`direction=-1` follows the branch to the other side (the rod buckles the
other way). This kind of branch switching assumes that the new branch
leaves the branch point roughly along the critical mode, as at a pitchfork
bifurcation like this one.

### Hopf bifurcations

The Brusselator, a model of an oscillating chemical reaction, with
diffusion and no-flux boundaries has the uniform steady state
$u = a$, $v = b / a$. It becomes unstable to oscillations at
$b = 1 + a^2$:

```python
a, Du, Dv = 1.0, 0.05, 0.05
grid = Grid(x=chebyshev(17, 0, 1))
D2 = grid.diff_operator(2)

def brusselator(u, v, b):
    return (Du * D2(u) + a - (b + 1) * u + u**2 * v,
            Dv * D2(v) + b * u - u**2 * v)

branch = continuation(grid, brusselator, (a, 1.5 / a), param=1.5, bcs=Neumann(0),
                      ds=0.1, param_range=(1.5, 2.5), stability=4)
hopf = branch.bifurcations[0]
hopf.kind         # "hopf"
hopf.param        # 2.0000000...
hopf.frequency    # 1.0000000...
```

### Limits

- Bifurcations are detected between neighbouring points of the branch. An
  eigenvalue that crosses the imaginary axis and returns within one step
  goes unnoticed; use a smaller `ds` if in doubt.
- Only the `k` leading eigenvalues are known. `n_unstable` cannot exceed
  `k`, and where all `k` eigenvalues are unstable, further crossings are
  missed. Choose `k` larger than the number of growing modes you expect.
- With a finite-difference Jacobian, the eigenvalues are accurate to about
  six digits, and so are the parameters of branch points and Hopf
  bifurcations. Folds do not suffer: they are found from the branch itself.
- The continuation stays on its branch at a branch point. With large steps
  it can still jump to the crossing branch near a transcritical
  bifurcation; a smaller `ds` avoids that.

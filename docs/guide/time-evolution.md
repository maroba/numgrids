# Time evolution

Time-dependent PDEs such as the heat, wave or Schrödinger equation can be
solved with the *method of lines*: discretize space on a grid and integrate
the resulting system of ODEs in time. `evolve` does this with
{func}`scipy.integrate.solve_ivp`, while letting you write the right-hand
side in terms of grid-shaped arrays.

```python
from numgrids import *
import numpy as np
```

## A first example: the heat equation

$\partial_t u = \kappa\, \partial_x^2 u$ on $[0, 1]$ with $u = 0$ at both
ends:

```python
grid = Grid(x=chebyshev(40, 0, 1))
x = grid.coords
kappa = 0.1

result = evolve(
    grid,
    lambda t, u: kappa * grid.laplacian(u),   # right-hand side
    np.sin(np.pi * x),                        # initial condition
    t_span=(0, 1),
    t_eval=np.linspace(0, 1, 11),
    bcs=0,
    method="BDF",                             # implicit: diffusion is stiff
)

result.t          # the 11 time points
result.u.shape    # (11, 40)
result.final      # solution at t = 1
```

For linear problems you can pass an [Operator](operators.md) instead of a
function. Implicit methods then use its sparse matrix as the Jacobian, which
makes them fast:

```python
result = evolve(grid, kappa * grid.laplacian_operator(), np.sin(np.pi * x),
                (0, 1), bcs=0, method="BDF")
```

## Boundary conditions

`bcs` accepts the [compact specifications](boundary-conditions.md) of
**Dirichlet, Neumann and Robin** conditions. The conditions hold at every
instant: the boundary values are not evolved themselves but computed from
the interior values so that the conditions are satisfied. The initial state
is adjusted accordingly. For Dirichlet conditions this simply means the
prescribed values.

```python
# Temperature 1 on the left, 3 on the right; relaxes to a linear profile
result = evolve(grid, grid.laplacian_operator(), np.zeros(grid.shape), (0, 5),
                bcs={"x_low": 1, "x_high": 3}, method="Radau")
```

An insulated rod ($\partial u / \partial n = 0$ at both ends) keeps its
total heat, and the temperature levels out:

```python
u0 = 1 + np.cos(np.pi * x)
result = evolve(grid, grid.laplacian_operator(), u0, (0, 2), bcs=Neumann(0), method="BDF")
grid.integrate(result.final)       # 1.0, as at t = 0
```

Newton's law of cooling at a surface, $-k\,\partial u / \partial n = h\,(u - u_\infty)$,
is a Robin condition $h u + k\,\partial u / \partial n = h u_\infty$:

```python
h, k, u_inf = 2.0, 1.0, 0.0
result = evolve(grid, grid.laplacian_operator(), np.ones(grid.shape), (0, 1),
                bcs={"x_low": Neumann(0), "x_high": Robin(h, k, u_inf)}, method="BDF")
```

### Boundary values that change in time

Wrap a function of time in `TimeDependent` to prescribe boundary values
that change in time. It works for Dirichlet, Neumann and Robin conditions:

```python
# The left end is heated periodically, the right end is insulated
result = evolve(grid, grid.laplacian_operator(), np.zeros(grid.shape), (0, 10),
                bcs={"x_low": TimeDependent(lambda t: np.sin(t)), "x_high": Neumann(0)},
                method="BDF")

# A prescribed heat flux that is switched off at t = 1
bcs = {"x_low": 0, "x_high": Neumann(TimeDependent(lambda t: 1.0 if t < 1 else 0.0))}
```

On faces of two- and three-dimensional grids, the function of time may
return a function of the face coordinates (or an array), just like
time-independent [boundary values](boundary-conditions.md#callable-boundary-values):

```python
g = TimeDependent(lambda t: lambda c: np.exp(-t) * np.sin(np.pi * c[1]))
```

## Source terms

`source=` adds a source term $s$ to the right-hand side,
$\partial_t u = F(t, u) + s$. It is an array of grid shape or a function of
time returning one. With an operator as right-hand side the problem stays
linear, and implicit methods keep the exact Jacobian:

```python
grid = Grid(x=chebyshev(40, 0, 1))
x = grid.coords
heating = lambda t: np.exp(-50 * (x - 0.5)**2) * np.sin(t)**2
result = evolve(grid, 0.1 * grid.laplacian_operator(), np.zeros(grid.shape), (0, 10),
                bcs=0, method="BDF", source=heating)
```

## Coupled fields: the wave equation

Pass a tuple of initial fields to integrate a system; the right-hand side
then receives and returns one array per field. The wave equation
$\partial_t^2 u = c^2 \nabla^2 u$ becomes $\partial_t u = v$,
$\partial_t v = c^2 \nabla^2 u$, here on a circular drum (a full disk with a
[radial axis](curvilinear.md#full-disks-cylinders-spheres-and-balls)):

```python
grid = PolarGrid(radial(24, 1.0), periodic(32))
X, Y = grid.cartesian_coords
u0 = np.exp(-20 * ((X - 0.3)**2 + Y**2))

result = evolve(
    grid,
    lambda t, u, v: (v, grid.laplacian(u)),
    (u0, np.zeros_like(u0)),
    t_span=(0, 2),
    t_eval=np.linspace(0, 2, 50),
    bcs=0,
)
u, v = result.u          # each of shape (50, 24, 32)
grid.plot(u[-1])
```

Linear systems can also be written as a nested list of operator blocks,
`rhs[i][j]` acting on field `j` in the equation of field `i`. A block is
an operator, an array of grid shape (multiplication), a number, or `None`
for zero. Implicit methods then use the exact Jacobian of the system, and
the [exponential integrator](#linear-problems-without-time-stepping-error)
becomes available:

```python
L = grid.laplacian_operator()
result = evolve(grid, [[None, 1], [L, None]],        # u_t = v,  v_t = L u
                (u0, np.zeros_like(u0)), (0, 2), bcs=0)
```

For systems, a single boundary condition specification applies to the
first field, and the other fields are held fixed on the same boundary
points. That is what $\partial_t u = v$ formulations need. To give each
field its own conditions, pass a tuple with one specification per field
(`None` for none). Here is a reaction-diffusion system with no-flux
boundaries for both species:

```python
grid = Grid(x=chebyshev(40, 0, 10))
x = grid.coords
result = evolve(
    grid,
    lambda t, a, b: (grid.laplacian(a) + a - a * b, 0.5 * grid.laplacian(b) + a * b - b),
    (1 + 0.5 * np.cos(np.pi * x / 10), np.ones(grid.shape)),
    t_span=(0, 5),
    bcs=(Neumann(0), Neumann(0)),
)
```

## Complex fields: the Schrödinger equation

Complex initial states are supported by the explicit methods (`RK45`,
`DOP853`, ...):

```python
grid = Grid(x=periodic(128, -20, 20))
x = grid.coords
psi0 = np.exp(-x**2 + 2j * x)
result = evolve(grid, lambda t, psi: 0.5j * grid.diff(psi, 2), psi0, (0, 2),
                t_eval=[0, 1, 2], rtol=1e-8)
```

## Linear problems without time stepping error

For linear problems with constant boundary values and sources, the
solution is known in closed form, $u(t) = e^{tL} u_0$ (plus the
contribution of sources and boundary values). `method="expm"` evaluates it
with matrix exponentials, directly at the requested times:

```python
grid = Grid(x=chebyshev(40, 0, 1))
x = grid.coords
result = evolve(grid, 0.1 * grid.laplacian_operator(), np.sin(np.pi * x), (0, 5),
                t_eval=[0, 0.1, 1, 5], bcs=0, method="expm")
```

There is no time step and hence no time stepping error, however stiff the
problem and however long the time interval. The right-hand side must be an
operator (or a list of operator blocks for systems).

This is the method of choice for the Schrödinger equation
$i\,\partial_t \psi = H \psi$: the fields are complex, the operator is
stiff, and the norm is conserved to round-off, also over long times:

```python
grid = Grid(x=chebyshev(120, -15, 15))
x = grid.coords
H = -0.5 * grid.diff_operator(2) + 0.5 * x**2             # harmonic oscillator
psi0 = np.pi**-0.25 * np.exp(-(x - 2)**2 / 2)             # displaced ground state

result = evolve(grid, -1j * H, psi0, (0, 2 * np.pi), t_eval=np.linspace(0, 2 * np.pi, 9),
                bcs=0, method="expm")
grid.norm(result.final)                                   # 1.0
abs(grid.inner(psi0, result.final))                       # 1.0: back after one period
```

Up to 2000 unknowns the matrix exponential is computed once per distinct
time step (equally spaced output times need only one), which also makes
long series of outputs cheap. Larger problems use
{func}`scipy.sparse.linalg.expm_multiply`, whose cost grows with the
stiffness of the operator.

## Stiff nonlinear problems

Many nonlinear PDEs are *semilinear*,

$$
\partial_t u = L u + N(t, u),
$$

with a linear part $L$ that contains the highest derivatives and is
therefore stiff, and a nonlinear part $N$ that is not: the Korteweg-de
Vries, nonlinear Schrödinger, Kuramoto-Sivashinsky, Allen-Cahn and
Swift-Hohenberg equations, and reaction-diffusion systems. Explicit
methods need tiny time steps for $L$, and implicit methods have to solve
nonlinear systems in every step. If you pass the linear part separately,
as `linear=`, `evolve` can treat the two parts differently. The right-hand
side `rhs` is then only the nonlinear part:

```python
# Korteweg-de Vries equation u_t + 6 u u_x + u_xxx = 0
grid = Grid(x=periodic(256, -20, 20))
x = grid.coords
c = 4.0
soliton = lambda t: c / 2 / np.cosh(np.sqrt(c) / 2 * (x + 10 - c * t))**2
D = grid.diff_operator()

result = evolve(
    grid,
    lambda t, u: -3 * D(u**2),            # nonlinear part: -6 u u_x
    soliton(0),
    t_span=(0, 5),
    t_eval=np.linspace(0, 5, 11),
    linear=-grid.diff_operator(3),        # stiff linear part: -u_xxx
    method="ETDRK4",
    dt=0.005,
)
abs(result.final - soliton(5)).max()      # 7e-5: the soliton has moved by 20
```

The methods integrate with a fixed time step `dt`:

| Method | Order | Linear part | Cost per step |
|---|---|---|---|
| `"ETDRK4"` | 4 | exact | 4 evaluations of $N$ |
| `"ETDRK2"` | 2 | exact | 2 evaluations of $N$ |
| `"ETD1"` | 1 | exact | 1 evaluation of $N$ |
| `"IMEX2"` | 2 | implicit (L-stable) | 2 evaluations of $N$, 2 linear solves |

The *exponential time differencing* (ETD) methods integrate the linear
part exactly, with the matrix exponential $e^{hL}$ and related functions,
and the nonlinear part with an explicit Runge-Kutta scheme. The time step
is limited by the nonlinear part only, not by the stiffness of $L$.
`"ETDRK4"` is the method of choice in most cases. `"IMEX2"` is an
implicit-explicit Runge-Kutta scheme: it only solves linear systems with
the matrix $1 - \gamma h L$, which makes it suitable for large problems on
grids with boundaries (see below).

There is no error control: choose `dt` by comparing with a run with half
the step. Every interval between two output times is divided into equal
steps that are not longer than `dt`, so the times in `t_eval` are hit
exactly; without `t_eval` the result holds the start and end time. If the
solution blows up, `result.success` is `False`, `result.message` tells
where, and the result ends with the last output time before that.

Complex fields work as well. For the nonlinear Schrödinger equation
$i\,\partial_t \psi + \tfrac12 \partial_x^2 \psi + |\psi|^2 \psi = 0$
with the soliton $\psi = \operatorname{sech}(x)\, e^{it/2}$:

```python
grid = Grid(x=periodic(256, -20, 20))
x = grid.coords
result = evolve(grid, lambda t, psi: 1j * abs(psi)**2 * psi, 1 / np.cosh(x), (0, 10),
                linear=0.5j * grid.diff_operator(2), method="ETDRK4", dt=0.05)
abs(result.final - np.exp(5j) / np.cosh(x)).max()      # 3e-6
grid.norm(result.final)**2                             # 2.0, as at t = 0
```

### Periodic grids: time stepping in Fourier space

On a grid that is periodic along all axes, a linear part with constant
coefficients is diagonal in Fourier space. `evolve` detects this and does
the time stepping on the Fourier coefficients, with fast Fourier
transforms and no matrices at all. Two-dimensional problems on large grids
are cheap this way. For systems, `linear` is a nested list of blocks like
a [linear right-hand side](#coupled-fields-the-wave-equation). Here is the
Gray-Scott reaction-diffusion system, in which a small seed of the species
$v$ grows into a labyrinth of stripes:

```python
grid = Grid(x=periodic(64, 0, 1.25), y=periodic(64, 0, 1.25))
X, Y = grid.meshed_coords
Lap = grid.laplacian_operator()
F, k = 0.037, 0.06

seed = np.exp(-((X - 0.6)**2 + (Y - 0.65)**2) / 0.005)
result = evolve(
    grid,
    lambda t, u, v: (-u * v**2 + F * (1 - u), u * v**2 - (F + k) * v),
    (1 - 0.5 * seed, 0.25 * seed),
    t_span=(0, 2000),
    linear=[[2e-5 * Lap, None], [None, 1e-5 * Lap]],
    method="ETDRK4",
    dt=2.0,
)
u, v = result.final
```

The blocks may also couple the fields, e.g. `[[None, 1], [L, None]]` for
a nonlinear wave equation $\partial_t^2 u = \nabla^2 u + f(u)$; each
Fourier mode then has a small matrix instead of a number.

Operators with mixed derivatives, such as $\nabla^4 = \partial_x^4 +
2\,\partial_x^2 \partial_y^2 + \partial_y^4$, have dense matrices in two
dimensions: $N^4$ entries on an $N \times N$ grid. On periodic grids you
can avoid the matrix and pass the *Fourier symbol* of the linear part
instead, a function of the meshed wave numbers: the factor by which the
operator multiplies $e^{i (k_x x + k_y y)}$ (so $\partial_x$ becomes
$i k_x$ and $\nabla^2$ becomes $-k_x^2 - k_y^2$). For the Swift-Hohenberg
equation $\partial_t u = r u - (1 + \nabla^2)^2 u - u^3$, which forms
stripes with the wave number 1:

```python
grid = Grid(x=periodic(128, 0, 32 * np.pi), y=periodic(128, 0, 32 * np.pi))
u0 = 0.1 * np.random.default_rng(0).standard_normal(grid.shape)
result = evolve(grid, lambda t, u: -u**3, u0, (0, 100),
                linear=lambda kx, ky: 0.3 - (1 - kx**2 - ky**2)**2,
                method="ETDRK4", dt=0.5)
```

The eigenvalues of this linear part reach $-960$, so explicit methods
would need steps of about 0.003 here. In a nested list for systems,
symbols and operators can be mixed.

### Grids with boundaries

On other grids, with boundary conditions, or with coefficients that vary
in space, the ETD methods compute the functions of the linear part as
dense matrices, once per step size (for equally spaced output times:
once). Dirichlet, Neumann and Robin conditions work as for the other
methods. The Allen-Cahn equation
$\partial_t u = \epsilon\, \partial_x^2 u + u - u^3$ forms plateaus
$u = \pm 1$ that last for a long time:

```python
grid = Grid(x=chebyshev(64, -1, 1))
x = grid.coords
u0 = 0.53 * x + 0.47 * np.sin(-1.5 * np.pi * x)
result = evolve(grid, lambda t, u: u - u**3, u0, (0, 30), t_eval=np.linspace(0, 30, 61),
                bcs={"x_low": -1, "x_high": 1},
                linear=0.01 * grid.diff_operator(2), method="ETDRK4", dt=0.25)
```

Constant boundary values and constant sources are part of the linear
problem and are integrated exactly: without a nonlinear part, the ETD
methods reproduce [`method="expm"`](#linear-problems-without-time-stepping-error).
Boundary values and sources that change in time are treated like the
nonlinear part; with [time-dependent boundary values](#boundary-values-that-change-in-time)
the observed order of `"ETDRK4"` drops from 4 to about 3, while the
second-order methods keep their order.

Dense matrix functions limit the ETD methods to 2500 unknowns on such
grids, and their setup takes about a hundred matrix products (a few
seconds for 1500 unknowns). Larger problems are the domain of `"IMEX2"`,
which solves its linear systems like [`Operator.solve`](operators.md):
with sparse or dense factorizations, or decoupled into Fourier modes where
the problem allows it (systems of several fields: with dense or sparse LU
factorizations). Here on a disk:

```python
grid = PolarGrid(radial(24, 1.0), periodic(32))
X, Y = grid.cartesian_coords
result = evolve(grid, lambda t, u: u * (1 - u), np.exp(-10 * ((X - 0.3)**2 + Y**2)), (0, 5),
                bcs=Neumann(0), linear=0.05 * grid.laplacian_operator(), method="IMEX2", dt=0.02)
```

## Looking at the result

`result.animate()` returns a matplotlib animation of the solution, and
`result.to_vtk("run")` writes a time series for ParaView; see
[Plotting](plotting.md#animations) and
[Saving and loading](io.md#vtk-export-for-paraview).

## Options

Additional keyword arguments (`rtol`, `atol`, `max_step`, `first_step`,
...) are passed on to {func}`scipy.integrate.solve_ivp`; the raw solver
result is available as `result.sol`. `method="expm"` and the fixed-step
methods of the previous section take no further options.

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

`bcs` accepts the [compact specifications](boundary-conditions.md) for
**Dirichlet** conditions. The boundary values are imposed on the initial
state and held fixed: the time derivative vanishes on those faces. Other
conditions (Neumann, Robin, time-dependent values) can be built into the
right-hand side.

```python
# Temperature 1 on the left, 3 on the right; relaxes to a linear profile
result = evolve(grid, grid.laplacian_operator(), np.zeros(grid.shape), (0, 5),
                bcs={"x_low": 1, "x_high": 3}, method="Radau")
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

## Options

Additional keyword arguments (`rtol`, `atol`, `max_step`, `first_step`,
...) are passed on to {func}`scipy.integrate.solve_ivp`; the raw solver
result is available as `result.sol`.

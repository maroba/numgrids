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

The values of the conditions are constant in time. Time-dependent boundary
data can be built into the right-hand side.

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

## Options

Additional keyword arguments (`rtol`, `atol`, `max_step`, `first_step`,
...) are passed on to {func}`scipy.integrate.solve_ivp`; the raw solver
result is available as `result.sol`.

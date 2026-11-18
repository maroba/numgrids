# Convergence studies

How fast does a numerical result approach the exact answer as the
resolution grows? `convergence_study` answers this in one call.

```python
from numgrids import *
import numpy as np
```

## Usage

Provide a grid factory, the quantity of interest, and the resolutions to
test:

```python
result = convergence_study(
    make_grid=lambda n: Grid(equidistant(n, 0, 1)),
    func=lambda grid: grid.diff(np.sin(grid.coords), acc=4),
    ns=[10, 20, 40, 80],
    exact=lambda grid: np.cos(grid.coords),
)

result.errors     # error for each n
result.order      # fitted algebraic order p in  error ~ n^-p  (here ~4)
result.orders     # pairwise orders between consecutive resolutions
result.plot()     # log-log plot (requires matplotlib)
```

`func` may return an array of grid shape (compared with the norm given by
`norm="max" | "l2" | "mean"`) or a scalar such as an integral or an
eigenvalue.

`exact` can be a callable `exact(grid)`, a number, or omitted. Without an
exact solution, the result on the finest grid serves as the reference (and
is excluded from the output); array results are interpolated from the
finest grid with spectral interpolation.

## Spectral convergence

Spectral methods converge faster than any power of $n$. For them, the
exponential rate $c$ in $\text{error} \sim e^{-c n}$ is more meaningful
and available as `result.rate`:

```python
result = convergence_study(
    lambda n: Grid(chebyshev(n, 0, 1)),
    lambda grid: grid.integrate(np.exp(3 * grid.coords)),
    ns=[4, 6, 8, 10, 12],
    exact=(np.exp(3) - 1) / 3,
)
result.rate
```

See also [Adaptive mesh refinement](amr.md), which uses similar error
estimates to choose the resolution of each axis automatically. On
spectral grids, the [resolution check](spectral.md#is-my-function-resolved)
tells from the values on a single grid whether a function is resolved and
how many points it needs.

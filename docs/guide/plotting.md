# Plotting

Plotting requires matplotlib, which is an optional dependency:

```bash
pip install numgrids[plot]
```

```python
from numgrids import *
import numpy as np
```

## Plotting functions on a grid

`grid.plot(f)` picks a suitable visualization based on the grid's
dimension and returns the matplotlib `Axes`:

```python
# 1D: line plot over the coordinate
grid = Grid(x=chebyshev(40, 0, np.pi))
grid.plot(np.sin(grid.coords))

# 2D: color plot
grid = Grid(x=chebyshev(30, 0, 1), y=periodic(40))
X, Y = grid.meshed_coords
ax = grid.plot(X * np.sin(Y), cmap="viridis")
ax.set_title("my field")
```

Curvilinear 2D grids are drawn in **Cartesian space**, with periodic axes
closed seamlessly -- a `PolarGrid` shows up as a disk:

```python
grid = PolarGrid(chebyshev(30, 0, 1), periodic(48))
R, Phi = grid.meshed_coords
grid.plot(R**2 * np.cos(3 * Phi))
grid.plot(R**2 * np.cos(3 * Phi), cartesian=False)   # in (r, phi) space
```

For grids with **three or more dimensions**, a 2D slice is shown. By
default the trailing axes are fixed at their middle index; choose the slice
with `fixed`:

```python
grid = Grid(x=chebyshev(20, 0, 1), y=chebyshev(20, 0, 1), z=chebyshev(20, 0, 1))
X, Y, Z = grid.meshed_coords
f = X * Y * Z
grid.plot(f)                     # slice at the middle of z
grid.plot(f, fixed={"x": 0})     # the y-z plane at the first x point
```

Further options: `ax=` to draw into existing axes, `colorbar=False`, and
any keyword argument of `ax.plot` (1D) or `ax.pcolormesh` (2D). The same
function is available as {func}`numgrids.plots.plot_field`.

## Visualizing grids and axes

Without arguments, `grid.plot()` shows the distribution of grid points and
`axis.plot()` shows the points of a single axis.

# Interpolation

numgrids can evaluate gridded data at arbitrary off-grid locations using the
`Interpolator` class. Under the hood it wraps scipy's
`RegularGridInterpolator` (for multi-dimensional grids) and `interp1d` (for
1D grids).

```python
from numgrids import *
import numpy as np
```

## The Interpolator class

Create an `Interpolator` by passing a grid, the function data, and
(optionally) the interpolation method:

```python
ax = create_axis(AxisType.CHEBYSHEV, 50, 0.0, 1.0)
grid = Grid(ax)
x = grid.coords
f = np.sin(np.pi * x)

interp = Interpolator(grid, f)
```

### Interpolation methods

The `method` parameter accepts three values:

| Method       | Order | Description |
|-------------|-------|-------------|
| `"linear"`  | 1     | Piecewise linear interpolation |
| `"quadratic"` | 2   | Piecewise quadratic interpolation |
| `"cubic"`   | 3     | Piecewise cubic interpolation (default) |

```python
interp_linear = Interpolator(grid, f, method="linear")
interp_cubic  = Interpolator(grid, f, method="cubic")
```

```{tip}
The default `"cubic"` method is a good choice for most smooth data. Use
`"linear"` when speed matters more than smoothness or when the data itself
is not smooth.
```

## Interpolating at points

### Single point

Pass a tuple of coordinates (one value per grid dimension):

```python
val = interp((0.3,))
print(val)  # sin(0.3 * pi) ~ 0.809
```

For 1D grids you can also pass a plain number:

```python
val = interp(0.3)
```

### List of points

Pass a list of coordinate tuples:

```python
points = [(0.1,), (0.5,), (0.9,)]
vals = interp(points)
print(vals)  # array of 3 values
```

### 2D example

```python
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
)
X, Y = grid.meshed_coords
f = np.sin(np.pi * X) * np.cos(np.pi * Y)

interp = Interpolator(grid, f)

# Single point
val = interp((0.25, 0.75))

# Multiple points
pts = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)]
vals = interp(pts)
```

### Zip objects

You can pass a `zip` of coordinate arrays, which is handy for constructing
point lists from separate x and y arrays:

```python
xs = [0.1, 0.3, 0.5]
ys = [0.2, 0.4, 0.6]
vals = interp(zip(xs, ys))
```

## Interpolating onto another Grid

Pass a `Grid` object as the location argument to interpolate the data onto an
entirely different grid. This is useful for transferring data between grids
of different resolutions or types:

```python
coarse_grid = Grid(
    create_axis(AxisType.EQUIDISTANT, 20, 0.0, 1.0),
    create_axis(AxisType.EQUIDISTANT, 20, 0.0, 1.0),
)
fine_grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 60, 0.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 60, 0.0, 1.0),
)

X, Y = coarse_grid.meshed_coords
f_coarse = np.sin(np.pi * X) * np.cos(np.pi * Y)

interp = Interpolator(coarse_grid, f_coarse, method="cubic")
f_fine = interp(fine_grid)
print(f_fine.shape)  # (60, 60)
```

The result has the shape of the target grid.

## The convenience function `interpolate()`

The module-level `interpolate()` function creates an `Interpolator`
internally and evaluates it in one step:

```python
val = interpolate(grid, f, (0.3, 0.7))
```

It accepts all the same location types (single tuple, list of tuples, zip,
or Grid):

```python
# Onto a fine grid
f_fine = interpolate(coarse_grid, f_coarse, fine_grid)
```

```{note}
Unlike `diff()` and `integrate()`, the `interpolate()` function creates a
fresh `Interpolator` on every call because the interpolant depends on
the data `f`, not just the grid. If you need to evaluate the same
interpolant at many different locations, create the `Interpolator`
explicitly and call it multiple times.
```

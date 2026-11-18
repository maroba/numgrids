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

The `method` parameter accepts these values:

| Method        | Order    | Description |
|---------------|----------|-------------|
| `"linear"`    | 1        | Piecewise linear interpolation |
| `"quadratic"` | 2        | Piecewise quadratic interpolation (1D grids only) |
| `"cubic"`     | 3        | Cubic spline interpolation (default) |
| `"quintic"`   | 5        | Quintic spline interpolation |
| `"spectral"`  | spectral | Matches the axis type, see below |

```python
interp_linear = Interpolator(grid, f, method="linear")
interp_cubic  = Interpolator(grid, f, method="cubic")
```

```{tip}
The default `"cubic"` method is a good choice for most smooth data. Use
`"linear"` when speed matters more than smoothness or when the data itself
is not smooth. Use `"spectral"` to keep the full accuracy of Chebyshev and
periodic grids.
```

### Spectral interpolation

With `method="spectral"` each axis is interpolated in the way that matches
its discretization:

* **Chebyshev axes** -- barycentric polynomial interpolation through all
  nodes (spectrally accurate, numerically stable),
* **periodic axes** -- trigonometric (Fourier) interpolation,
* **unbounded axes** (`semi_infinite`, `infinite`) -- barycentric
  interpolation in the computational variable, valid in the whole domain,
  also beyond the outermost grid point,
* **equidistant and logarithmic axes** -- local polynomial interpolation of
  degree 5 (in $\ln x$ for logarithmic axes).

```python
grid2d = Grid(chebyshev(20, 0, 1), periodic(32))
X, Y = grid2d.meshed_coords
spectral = Interpolator(grid2d, np.exp(X) * np.cos(Y), method="spectral")
spectral((0.37, 1.2))      # error ~ 1e-15
```

Spectral interpolation onto another grid is done axis by axis and is
therefore fast.

### Periodic axes

Periodic axes wrap around for all methods: points between the last grid
point and the end of the period, as well as points outside the period, are
handled correctly.

## Interpolating at points

### Single point

Pass a tuple of coordinates (one value per grid dimension):

```python
val = interp((0.3,))
print(val)  # sin(0.3 * pi) ~ 0.809
```

For 1D grids you can also pass a plain number (and a 1D array for many
points at once):

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

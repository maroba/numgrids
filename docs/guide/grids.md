# Grids

A `Grid` is the tensor product of one or more [axes](axes.md). It provides
meshed coordinate arrays, boundary information, and convenience methods for
refinement and coarsening. All numgrids operators (differentiation,
integration, interpolation) take a `Grid` as their first argument.

```python
from numgrids import *
import numpy as np
```

## Creating a grid

Pass one or more axes to the `Grid` constructor. The order of the axes
determines the axis indices (the first axis is index 0):

```python
ax_x = create_axis(AxisType.CHEBYSHEV, 40, 0.0, 1.0)
ax_y = create_axis(AxisType.EQUIDISTANT, 50, -1.0, 1.0)
grid = Grid(ax_x, ax_y)
```

### 1D grid

```python
ax = create_axis(AxisType.CHEBYSHEV, 100, 0.0, 1.0)
grid = Grid(ax)
print(grid.shape)  # (100,)
print(grid.ndims)  # 1
```

### 2D grid

```python
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 40, 0.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 60, 0.0, 2.0),
)
print(grid.shape)  # (40, 60)
```

### 3D grid

```python
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 20, 0.0, 1.0),
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0.0, 2 * np.pi),
    create_axis(AxisType.CHEBYSHEV, 20, -1.0, 1.0),
)
print(grid.shape)  # (20, 30, 20)
print(grid.ndims)  # 3
```

```{tip}
You can freely mix axis types within a single grid. For example, use a
Chebyshev axis for the radial direction, a periodic equidistant axis for
the angular direction, and a logarithmic axis for a third coordinate.
```

## Grid properties

### `shape`

A tuple of the number of points along each axis:

```python
print(grid.shape)  # (20, 30, 20)
```

### `size`

The total number of grid points (product of all axis lengths):

```python
print(grid.size)  # 12000
```

### `ndims`

The number of dimensions:

```python
print(grid.ndims)  # 3
```

### `coords`

For 1D grids, returns the 1D coordinate array directly. For multi-dimensional
grids, returns a tuple of 1D coordinate arrays (one per axis):

```python
# 1D
grid_1d = Grid(create_axis(AxisType.EQUIDISTANT, 10, 0.0, 1.0))
x = grid_1d.coords  # 1D array of length 10

# 2D
grid_2d = Grid(
    create_axis(AxisType.EQUIDISTANT, 10, 0.0, 1.0),
    create_axis(AxisType.EQUIDISTANT, 20, 0.0, 2.0),
)
x_coords, y_coords = grid_2d.coords  # two 1D arrays
```

### `meshed_coords`

Returns a tuple of N-dimensional arrays (one per axis) created by
`numpy.meshgrid` with `indexing="ij"`. These are the arrays you use to
evaluate functions on the grid:

```python
X, Y = grid_2d.meshed_coords
f = np.sin(X) * np.cos(Y)  # shape (10, 20)
```

### `boundary`

A boolean mask of shape `grid.shape` that is `True` on boundary points and
`False` in the interior. Periodic axes have no boundary points:

```python
grid = Grid(create_axis(AxisType.EQUIDISTANT, 5, 0.0, 1.0))
print(grid.boundary)
# [ True False False False  True]
```

### `faces`

A dictionary of `BoundaryFace` objects, one per non-periodic boundary face.
Keys follow the pattern `"{axis_index}_{side}"`:

```python
grid = Grid(
    create_axis(AxisType.EQUIDISTANT, 10, 0.0, 1.0),
    create_axis(AxisType.EQUIDISTANT, 10, 0.0, 1.0),
)
print(grid.faces.keys())
# dict_keys(['0_low', '0_high', '1_low', '1_high'])
```

Periodic axes are automatically excluded:

```python
grid = Grid(
    create_axis(AxisType.EQUIDISTANT, 10, 0.0, 1.0),
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 20, 0.0, 2 * np.pi),
)
print(grid.faces.keys())
# dict_keys(['0_low', '0_high'])
```

The `faces` dictionary is the starting point for applying
[boundary conditions](boundary-conditions.md).

## Accessing individual axes

Use `get_axis()` to retrieve an axis by index:

```python
axis_0 = grid.get_axis(0)
print(axis_0)
```

The `axes` property returns the full tuple:

```python
for ax in grid.axes:
    print(ax)
```

## Refinement and coarsening

### `refine()`

Returns a new grid with twice the number of points along every axis:

```python
fine = grid.refine()
print(grid.shape)  # (10, 10)
print(fine.shape)  # (20, 20)
```

### `coarsen()`

Returns a new grid with half the number of points along every axis:

```python
coarse = grid.coarsen()
print(coarse.shape)  # (5, 5)
```

### `refine_axis(axis_index, factor)`

Refines (or coarsens) a single axis by a multiplicative factor. This is
useful for anisotropic refinement:

```python
# Double only axis 0
refined = grid.refine_axis(0, factor=2.0)
print(refined.shape)  # (20, 10)

# Triple axis 1
refined = grid.refine_axis(1, factor=3.0)
print(refined.shape)  # (10, 30)
```

A factor less than 1 coarsens. The result is rounded to the nearest integer
with a minimum of 2 points.

## Grid caching

The `Grid` object maintains an internal cache dictionary. Operators created
by the convenience functions (`diff`, `integrate`, `interpolate`) store
themselves in this cache to avoid redundant construction:

```python
# First call builds the Diff operator and caches it
df = diff(grid, f, order=1, axis_index=0)

# Second call with the same parameters reuses the cached operator
dg = diff(grid, g, order=1, axis_index=0)
```

The cache is per-grid-instance: creating a new `Grid` (e.g. via `refine()`)
starts with a fresh cache. You do not need to manage the cache manually.

## Visualization

For 2D or higher grids, call `plot()` to see the grid point distribution:

```python
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 15, 0.0, 1.0, name="x"),
    create_axis(AxisType.CHEBYSHEV, 15, 0.0, 1.0, name="y"),
)
grid.plot()
```

Named axes produce labeled plot axes automatically.

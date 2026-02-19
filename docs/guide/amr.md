# Adaptive mesh refinement

numgrids includes tools for estimating discretization error and automatically
refining the grid until a prescribed tolerance is met. Because numgrids uses
tensor-product grids, refinement is performed **per-axis** rather than
per-cell. This keeps all operators (differentiation, integration,
interpolation) working unchanged while still allowing anisotropic resolution.

```python
from numgrids import *
import numpy as np
```

## The core idea: Richardson extrapolation

The error estimation strategy is based on Richardson extrapolation. For each
axis, numgrids:

1. Creates a refined copy of the grid (more points along that axis only).
2. Evaluates the user function on both the original and refined grids.
3. Interpolates the refined result back onto the original grid.
4. Measures the difference.

The axis whose refinement changes the answer the most is the **resolution
bottleneck** -- refining it will improve accuracy the most. This per-axis
approach naturally handles anisotropic problems: a function that varies
rapidly in $x$ but slowly in $y$ will only refine the $x$ axis.

## Quick diagnostics with `estimate_error()`

For a one-shot error report, use `estimate_error()`. It returns both the
global error and per-axis error contributions:

```python
grid = Grid(
    create_axis(AxisType.EQUIDISTANT, 20, 0.0, 1.0),
    create_axis(AxisType.EQUIDISTANT, 20, 0.0, 1.0),
)

def my_func(g):
    X, Y = g.meshed_coords
    return X**5 + Y**2

result = estimate_error(grid, my_func)
print(f"Global error:  {result['global']:.2e}")
print(f"Per-axis errors: {result['per_axis']}")
```

The per-axis dictionary shows how much the solution changes when each axis
is refined independently. A large value for axis 0 means that axis needs
more resolution.

### Parameters

| Parameter           | Default | Description |
|--------------------|---------|-------------|
| `grid`             | --      | The grid to evaluate |
| `func`             | --      | `func(grid) -> NDArray` |
| `norm`             | `"max"` | Error norm: `"max"`, `"l2"`, or `"mean"` |
| `refinement_factor`| `2.0`   | Factor by which to test-refine each axis |

## Automatic refinement with `adapt()`

The `adapt()` function runs an iterative refinement loop. At each iteration
it identifies the worst axis, refines it, and re-evaluates:

```python
grid = Grid(
    create_axis(AxisType.EQUIDISTANT, 10, 0.0, 1.0),
    create_axis(AxisType.EQUIDISTANT, 10, 0.0, 1.0),
)

result = adapt(
    grid,
    lambda g: g.meshed_coords[0]**5 + g.meshed_coords[1]**2,
    tol=1e-4,
)

print(f"Converged: {result.converged}")
print(f"Final shape: {result.grid.shape}")
print(f"Iterations: {result.iterations}")
print(f"Global error: {result.global_error:.2e}")
```

Because $x^5$ is harder to resolve than $y^2$, `adapt()` will refine
axis 0 more aggressively than axis 1.

### Parameters

| Parameter              | Default | Description |
|-----------------------|---------|-------------|
| `grid`                | --      | Initial grid |
| `func`                | --      | `func(grid) -> NDArray` |
| `tol`                 | `1e-6`  | Target error tolerance |
| `norm`                | `"max"` | Error norm: `"max"`, `"l2"`, or `"mean"` |
| `max_iterations`      | `20`    | Maximum refinement iterations |
| `max_points_per_axis` | `1024`  | Safety cap to prevent runaway refinement |
| `refinement_factor`   | `2.0`   | Factor by which to increase resolution |
| `refine_all`          | `False` | Refine all under-resolved axes per iteration |

### The `refine_all` option

By default, `adapt()` refines only the single worst axis per iteration. Set
`refine_all=True` to refine *all* axes whose error exceeds `tol / ndims` in
a single step. This can converge faster for problems that are equally
under-resolved in all directions, but uses more points per step:

```python
result = adapt(grid, my_func, tol=1e-4, refine_all=True)
```

## The AdaptationResult

`adapt()` returns an `AdaptationResult` dataclass with the following fields:

| Field          | Type               | Description |
|----------------|--------------------|-------------|
| `grid`         | `Grid`             | The adapted grid |
| `f`            | `NDArray`          | Function evaluated on the final grid |
| `errors`       | `dict[int, float]` | Per-axis error estimates on the final grid |
| `global_error` | `float`            | Global error estimate on the final grid |
| `iterations`   | `int`              | Number of adaptation iterations performed |
| `converged`    | `bool`             | Whether the tolerance was met |
| `history`      | `list[dict]`       | Per-iteration diagnostics |

### Inspecting the history

The `history` list contains one dictionary per iteration with the grid
shape, global error, axis errors, and which axes were refined:

```python
for record in result.history:
    print(f"Iteration {record['iteration']}: "
          f"shape={record['shape']}, "
          f"global_error={record['global_error']:.2e}")
```

## Manual control with ErrorEstimator

For full control over the refinement loop, use the `ErrorEstimator` class
directly. This is useful when you want to combine error estimation with
custom refinement logic:

```python
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 15, 0.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 15, 0.0, 1.0),
)

def my_func(g):
    X, Y = g.meshed_coords
    return np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2))

est = ErrorEstimator(grid, my_func, norm="max")

# Global error (refine all axes, compare)
print(f"Global error: {est.global_error():.2e}")

# Per-axis errors
errors = est.per_axis_errors()
for axis_idx, err in errors.items():
    print(f"  Axis {axis_idx}: {err:.2e}")

# Which axis needs refinement most?
worst = est.axis_needing_refinement()
print(f"Refine axis: {worst}")
```

### ErrorEstimator methods

- `global_error(refinement_factor=2.0)` -- refine all axes, measure total
  difference
- `per_axis_errors(refinement_factor=2.0)` -- refine each axis independently,
  return `{axis_index: error}` dict
- `axis_needing_refinement(refinement_factor=2.0)` -- return the index of
  the axis with the largest error, or `None` if fully resolved
- `f_current` -- the cached function evaluation on the current grid

### Custom refinement loop

```python
tol = 1e-5

while True:
    est = ErrorEstimator(grid, my_func, norm="max")
    if est.global_error() < tol:
        print(f"Converged with shape {grid.shape}")
        break

    worst_axis = est.axis_needing_refinement()
    if worst_axis is None:
        break

    grid = grid.refine_axis(worst_axis, factor=1.5)
    print(f"Refined axis {worst_axis} -> shape {grid.shape}")
```

## Why per-axis refinement?

Tensor-product grids have a key structural advantage: all numgrids operators
(differentiation, integration, interpolation) work directly on the grid
without any special treatment for non-uniform cell sizes. Refining individual
axes preserves this structure while allowing **anisotropic resolution**.

Consider a boundary layer problem where the solution changes rapidly near
$x = 0$ but is smooth in $y$. Per-axis refinement can give axis 0 two
hundred points while keeping axis 1 at twenty, for a total of 4,000 grid
points. Cell-based refinement of a uniform grid to the same resolution near
$x = 0$ would require a much larger point count or complex data structures.

```{note}
The error estimator compares solutions at different resolutions. Your
`func` callable is evaluated multiple times per iteration (once per axis
plus once for the global estimate). For expensive functions (e.g. PDE
solvers), consider using a coarser initial grid or a larger tolerance to
reduce the total number of function evaluations.
```

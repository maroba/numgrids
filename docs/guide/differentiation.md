# Differentiation

numgrids computes partial derivatives through the `Diff` class, which
automatically selects the best differentiation strategy based on the axis
type. You never need to worry about whether to use finite differences, FFT,
or Chebyshev spectral methods -- the grid tells `Diff` what to do.

```python
from numgrids import *
import numpy as np
```

## The Diff class

Create a `Diff` operator by specifying the grid, derivative order, and
(optionally) the axis index:

```python
ax = create_axis(AxisType.CHEBYSHEV, 50, 0.0, 1.0)
grid = Grid(ax)
x = grid.coords

d1 = Diff(grid, order=1)           # first derivative, axis 0
d2 = Diff(grid, order=2)           # second derivative, axis 0
```

### Constructor parameters

| Parameter    | Type  | Default | Description |
|-------------|-------|---------|-------------|
| `grid`      | `Grid` | --     | The grid on which to differentiate |
| `order`     | `int`  | --     | Derivative order (must be positive) |
| `axis_index`| `int`  | `0`    | Which axis to differentiate along |
| `acc`       | `int`  | `4`    | Accuracy order for finite-difference methods |

### Applying to arrays

A `Diff` object is callable. Pass it an array of shape `grid.shape`:

```python
f = np.sin(np.pi * x)
df = d1(f)                # should be close to pi * cos(pi * x)
d2f = d2(f)               # should be close to -pi^2 * sin(pi * x)
```

### Multi-dimensional example

```python
ax_x = create_axis(AxisType.CHEBYSHEV, 40, 0.0, 1.0)
ax_y = create_axis(AxisType.CHEBYSHEV, 40, 0.0, 1.0)
grid = Grid(ax_x, ax_y)
X, Y = grid.meshed_coords

f = np.sin(np.pi * X) * np.exp(Y)

d_dx = Diff(grid, order=1, axis_index=0)
d_dy = Diff(grid, order=1, axis_index=1)

df_dx = d_dx(f)   # pi * cos(pi*x) * exp(y)
df_dy = d_dy(f)   # sin(pi*x) * exp(y)
```

## Automatic strategy selection

`Diff` inspects the axis at `axis_index` and picks the differentiation
method accordingly:

| Axis type | Strategy | Notes |
|-----------|----------|-------|
| `EquidistantAxis` (non-periodic) | Finite differences | Accuracy controlled by `acc` |
| `EquidistantAxis` (periodic) | FFT spectral | Spectral accuracy for smooth periodic functions |
| `ChebyshevAxis` | Chebyshev spectral matrix | Spectral accuracy; `acc` is ignored |
| `LogAxis` | Finite differences on log-scale | Chain rule $df/dx = (1/x)\,df/d(\ln x)$ |

```{note}
The `acc` parameter only affects finite-difference-based strategies
(equidistant non-periodic and logarithmic axes). For Chebyshev and FFT
methods, the accuracy is determined by the number of grid points and `acc`
is silently ignored.
```

## The `acc` parameter

For finite-difference methods, `acc` controls the width of the stencil. A
higher value uses more neighboring points and yields a higher-order
approximation:

```python
ax = create_axis(AxisType.EQUIDISTANT, 50, 0.0, 1.0)
grid = Grid(ax)
x = grid.coords
f = np.sin(np.pi * x)

d_low  = Diff(grid, order=1, acc=2)
d_high = Diff(grid, order=1, acc=8)

err_low  = np.max(np.abs(d_low(f) - np.pi * np.cos(np.pi * x)))
err_high = np.max(np.abs(d_high(f) - np.pi * np.cos(np.pi * x)))
print(f"acc=2 error: {err_low:.2e}")   # larger
print(f"acc=8 error: {err_high:.2e}")  # smaller
```

```{tip}
The default `acc=4` is a good balance between accuracy and stencil width.
Increase it if you need higher accuracy on a coarse equidistant grid.
```

## Sparse matrix representation

Every `Diff` operator can export itself as a scipy sparse matrix via
`as_matrix()`. This is essential for building linear systems (e.g. for
solving PDEs):

```python
D = Diff(grid, order=2, axis_index=0)
L = D.as_matrix()
print(type(L))   # <class 'scipy.sparse._csc.csc_matrix'>
print(L.shape)   # (grid.size, grid.size)
```

The matrix operates on the *flattened* grid array. To apply it manually:

```python
f_flat = f.ravel()
d2f_flat = L @ f_flat
d2f = d2f_flat.reshape(grid.shape)
```

## Building linear operators

Because `as_matrix()` returns standard scipy sparse matrices, you can combine
them with ordinary linear algebra to build PDE operators. For example, the 2D
Laplacian $\nabla^2 = \partial^2/\partial x^2 + \partial^2/\partial y^2$:

```python
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
)

D_xx = Diff(grid, order=2, axis_index=0).as_matrix()
D_yy = Diff(grid, order=2, axis_index=1).as_matrix()

laplacian = D_xx + D_yy   # sparse matrix, shape (900, 900)
```

This sparse Laplacian matrix can be fed into `scipy.sparse.linalg.spsolve`
to solve elliptic PDEs. See the [Boundary conditions](boundary-conditions.md)
guide for a complete worked example.

## The convenience function `diff()`

For quick one-off derivatives, use the module-level `diff()` function. It
creates a `Diff` operator on the first call and **caches** it on the grid for
reuse:

```python
df_dx = diff(grid, f, order=1, axis_index=0)
df_dy = diff(grid, f, order=1, axis_index=1)
```

The cache key is the tuple `(order, axis_index, acc)`, so calls with the
same parameters hit the cache. This is convenient in interactive sessions or
when you differentiate many different functions on the same grid.

```{note}
The cache lives on the specific `Grid` instance. If you create a new grid
(e.g. via `grid.refine()`), the new grid has its own empty cache.
```

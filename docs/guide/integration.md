# Integration

numgrids provides numerical integration over the entire grid domain through
the `Integral` class and the convenience function `integrate()`.

```python
from numgrids import *
import numpy as np
```

## The Integral class

Create an `Integral` by passing a `Grid`:

```python
ax = create_axis(AxisType.CHEBYSHEV, 50, 0.0, 1.0)
grid = Grid(ax)
I = Integral(grid)
```

The constructor collects the quadrature weights of all axes, so reusing the
same `Integral` object for multiple functions is efficient.

### Calling the integrator

The `Integral` object is callable. Pass it an array of shape `grid.shape`:

```python
x = grid.coords
f = x ** 2
result = I(f)
print(result)  # close to 1/3
```

### 1D example

```python
ax = create_axis(AxisType.EQUIDISTANT, 100, 0.0, np.pi)
grid = Grid(ax)
I = Integral(grid)

x = grid.coords
print(I(np.sin(x)))  # close to 2.0
```

### 2D example

Integrate $f(x, y) = x^2 + y^2$ over $[0, 1] \times [0, 1]$:

```python
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 30, 0.0, 1.0),
)
X, Y = grid.meshed_coords
f = X ** 2 + Y ** 2
I = Integral(grid)
print(I(f))  # close to 2/3
```

### Quadrature rules

The integral is a tensor product of one-dimensional quadrature rules chosen
per axis type (see `Axis.quadrature_weights`):

| Axis type           | Rule                         | Accuracy                 |
|---------------------|------------------------------|--------------------------|
| Chebyshev           | Clenshaw--Curtis             | spectral                 |
| periodic            | trapezoidal                  | spectral (periodic *f*)  |
| equidistant, log    | inverse differentiation      | same order as `Diff`     |

The weights are computed once per axis and cached, so integration is a
cheap weighted sum.

## The convenience function `integrate()`

For quick usage, the module-level `integrate()` function -- or the
equivalent method `grid.integrate(f)` -- wraps `Integral` and caches the
operator on the grid:

```python
result = integrate(grid, f)
result = grid.integrate(f)
```

The first call constructs the `Integral` and stores it in `grid.cache`.
Subsequent calls for the same grid reuse the cached object.

## Integration in curvilinear coordinates

On curvilinear grids (`CurvilinearGrid`, `SphericalGrid`,
`CylindricalGrid`, `PolarGrid`, ...) the volume element
$J = h_1 h_2 \cdots h_n$ is included automatically, i.e. the integral is
$\int f \, dV$:

```python
grid = SphericalGrid(
    chebyshev(25, 0.0, 2.0),          # r
    chebyshev(20, 0.0, np.pi),        # theta
    periodic(30),                     # phi
)
volume = grid.integrate(np.ones(grid.shape))   # 4/3 pi 2^3 = 33.51...
```

The volume element is available as `grid.volume_element`. To integrate
with respect to the coordinate differentials $dq_1\,dq_2\cdots$ instead,
use `Integral(grid, volume_element=False)`.

```{versionchanged} 0.5.0
Earlier versions did not include the volume element; integrands had to be
multiplied by the Jacobian manually. Remove such factors when upgrading.
```

## Surface integrals

Each boundary face can integrate over itself, including the surface
element on curvilinear grids:

```python
sphere = grid.faces["r_high"]
sphere.integrate(np.ones(grid.shape))     # 4 pi 2^2
```

See [Boundary conditions](boundary-conditions.md#working-with-boundary-faces)
for more face operations.

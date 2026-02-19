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

The constructor pre-computes the integration weights (inverse differentiation
matrices), so reusing the same `Integral` object for multiple functions is
efficient.

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

### Periodic axes

For periodic equidistant axes the integrator uses the trapezoidal rule
(which is spectrally accurate for smooth periodic functions). For all other
axis types it uses an anti-differentiation approach based on the inverse of
the differentiation matrix.

## The convenience function `integrate()`

For quick usage, the module-level `integrate()` function wraps `Integral`
and caches the operator on the grid:

```python
result = integrate(grid, f)
```

The first call constructs the `Integral` and stores it in `grid.cache`.
Subsequent calls for the same grid reuse the cached object.

```python
# Both calls share the same Integral internally
a = integrate(grid, f)
b = integrate(grid, g)
```

## Integration in curvilinear coordinates

```{note}
The `Integral` class integrates with respect to the *coordinate*
differentials $dq_1\,dq_2\,\ldots$ It does **not** automatically include
the Jacobian of a curvilinear coordinate system.

If you are integrating on a `CurvilinearGrid` (or `SphericalGrid`,
`CylindricalGrid`, `PolarGrid`), you must multiply your integrand by the
appropriate volume element. For example, in spherical coordinates the
volume element is $r^2 \sin\theta\,dr\,d\theta\,d\phi$:
```

```python
grid = SphericalGrid(
    create_axis(AxisType.CHEBYSHEV, 25, 0.5, 2.0),
    create_axis(AxisType.CHEBYSHEV, 20, 0.01, np.pi - 0.01),
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2 * np.pi),
)
R, Theta, Phi = grid.meshed_coords

f = np.ones_like(R)                    # integrate 1 to get the volume
volume_element = R**2 * np.sin(Theta)  # Jacobian for spherical coords
result = integrate(grid, f * volume_element)
```

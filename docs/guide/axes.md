# Axes

An **axis** is the fundamental building block in numgrids. It represents a
single coordinate dimension -- a 1D set of grid points with a specific
spacing strategy. One or more axes are combined into a
[Grid](grids.md) via tensor product.

```python
from numgrids import *
import numpy as np
```

## Creating axes

The quickest way is one of the four short-hand factories, one per axis type:

```python
x = chebyshev(30, 0.0, 1.0)          # Chebyshev nodes on [0, 1]
y = equidistant(50, 0.0, 1.0)        # uniform spacing on [0, 1]
phi = periodic(64)                   # uniform, periodic on [0, 2*pi)
r = logarithmic(40, 1e-3, 10.0)      # log-spaced on [1e-3, 10]
```

All of them take `(num_points, low, high, name=None)`; sensible defaults
exist for the bounds (`[-1, 1]` for Chebyshev, `[0, 1]` for equidistant,
`[0, 2*pi)` for periodic).

Alternatively, use the generic `create_axis` factory:

```python
ax = create_axis(AxisType.EQUIDISTANT, 50, 0.0, 1.0)
ax = create_axis("chebyshev", 50, 0.0, 1.0)   # strings work, too
```

The arguments are:

1. **axis type** -- an `AxisType` member or its string value
   (`"equidistant"`, `"equidistant_periodic"` / `"periodic"`, `"chebyshev"`,
   `"log"`)
2. **num_points** -- number of grid points (at least 2)
3. **low** -- lower coordinate bound
4. **high** -- upper coordinate bound

## Axis types

### Equidistant (non-periodic)

```python
ax = create_axis(AxisType.EQUIDISTANT, 50, 0.0, 1.0)
```

Uniformly spaced points from `low` to `high` (both endpoints included).
Differentiation uses classical **finite differences** via the findiff library.
Best for problems where uniform resolution is sufficient and the solution does
not have sharp boundary layers.

### Equidistant periodic

```python
ax = create_axis(AxisType.EQUIDISTANT_PERIODIC, 64, 0.0, 2 * np.pi)
```

Uniformly spaced points on a periodic domain. The last point is *not*
included because it would coincide with the first after wrapping.
Differentiation uses **FFT spectral** methods, giving spectral accuracy for
smooth periodic functions.

### Chebyshev

```python
ax = create_axis(AxisType.CHEBYSHEV, 30, -1.0, 1.0)
```

Grid points are placed at Chebyshev nodes, which cluster near the domain
boundaries:

$$
x_k = \cos\!\left(\frac{k\pi}{N-1}\right), \quad k = 0, \ldots, N-1
$$

mapped linearly to $[\text{low}, \text{high}]$. Differentiation uses the
**Chebyshev spectral** differentiation matrix, achieving spectral accuracy
for smooth non-periodic functions. Excellent for resolving boundary layers.

### Logarithmic

```python
ax = create_axis(AxisType.LOGARITHMIC, 40, 0.01, 10.0)
```

Points are logarithmically spaced, clustering near the lower boundary. The
internal coordinate is $\xi = \ln(x)$, and differentiation applies the
chain rule:

$$
\frac{df}{dx} = \frac{1}{x}\,\frac{df}{d(\ln x)}
$$

```{warning}
The lower bound must be strictly positive (`low > 0`). A `ValueError` is
raised otherwise.
```

### Radial (through the origin)

```python
ax = radial(24, 1.0)            # or create_axis("radial", 24, 0, 1.0)
```

The radial coordinate of a *full* disk, cylinder or ball. The points are the
positive half of Chebyshev nodes on the diameter $[-R, R]$, so no point lies
on the origin. Derivatives use the whole diameter: the value at $-r$ is the
value at the antipodal point $(r, \varphi + \pi)$. This gives spectral
accuracy without a coordinate singularity, and only the outer end $r = R$ is
a boundary. A radial axis can only be used as the first axis of a
`PolarGrid`, `CylindricalGrid` or `SphericalGrid`; see
[Curvilinear grids](curvilinear.md#full-disks-cylinders-and-balls).

## Choosing the right axis type

| Scenario | Recommended type | Why |
|----------|-----------------|-----|
| Smooth, periodic function (e.g. Fourier modes) | `EQUIDISTANT_PERIODIC` | FFT spectral accuracy |
| Smooth, non-periodic function | `CHEBYSHEV` | Spectral accuracy without Runge phenomenon |
| Uniform resolution, moderate accuracy needs | `EQUIDISTANT` | Simple, controllable via `acc` parameter |
| Fine resolution near one boundary (e.g. $r \to 0$) | `LOGARITHMIC` | Concentrates points near `low` |
| Radial coordinate of a full disk, cylinder or ball | `RADIAL` | Includes the center without a singularity |
| Radial coordinate away from origin (annulus, shell) | `CHEBYSHEV` or `LOGARITHMIC` | Depends on whether the lower bound is positive |

```{tip}
When in doubt, start with `CHEBYSHEV`. It gives spectral accuracy and works
well for most smooth, non-periodic problems.
```

## Axis properties

Every axis exposes the following properties:

### `coords`

The 1D array of coordinate values in the user-specified domain:

```python
ax = create_axis(AxisType.CHEBYSHEV, 5, 0.0, 1.0)
print(ax.coords)
# array([0.  , 0.14644661, 0.5, 0.85355339, 1.  ])
```

### `coords_internal`

The raw coordinates in the axis's internal (canonical) coordinate system.
For Chebyshev axes, these are the nodes on $[-1, 1]$; for logarithmic axes,
the uniformly spaced values in $[\ln(\text{low}), \ln(\text{high})]$:

```python
ax = create_axis(AxisType.CHEBYSHEV, 5, 0.0, 1.0)
print(ax.coords_internal)
# array([ 1. ,  0.70710678,  0. , -0.70710678, -1. ])
```

### `boundary`

A `slice` that selects interior points (excluding boundary). For periodic
axes the full range is returned because there are no boundary points:

```python
ax = create_axis(AxisType.EQUIDISTANT, 10, 0.0, 1.0)
print(ax.boundary)   # slice(1, -1, None)

ax_p = create_axis(AxisType.EQUIDISTANT_PERIODIC, 10, 0.0, 2 * np.pi)
print(ax_p.boundary) # slice(None, None, None)
```

### `periodic`

Boolean flag indicating whether the axis uses periodic boundary conditions:

```python
print(ax.periodic)    # False
print(ax_p.periodic)  # True
```

## Resizing an axis

The `resized()` method returns a new axis of the same type and domain with a
different number of points. All metadata (periodicity, name) is preserved:

```python
ax = create_axis(AxisType.CHEBYSHEV, 20, 0.0, 1.0)
ax_fine = ax.resized(80)
print(len(ax_fine))  # 80
print(type(ax_fine))  # <class 'numgrids.axes.ChebyshevAxis'>
```

This is used internally by `Grid.refine()`, `Grid.coarsen()`, and the
adaptive mesh refinement module.

## Named axes

Pass the `name` keyword to give an axis a label:

```python
ax_r = chebyshev(30, 0.1, 5.0, name="r")
print(ax_r.name)  # "r"
```

Names can be used instead of axis indices everywhere on a grid
(`grid.diff(f, axis="r")`, `grid.faces["r_high"]`, ...), and they produce
nicer plot labels. `ax.renamed("rho")` returns a renamed copy. The easiest
way to name axes is to pass them as keyword arguments to
[`Grid`](grids.md).

## Quadrature weights

Every axis knows a quadrature rule matching its type, available as
`ax.quadrature_weights` (Clenshaw--Curtis for Chebyshev axes, the
trapezoidal rule for periodic axes, and a rule consistent with the
finite-difference stencils otherwise):

```python
x = chebyshev(20, 0, np.pi)
print(x.quadrature_weights @ np.sin(x.coords))   # 2.0
```

## Visualization

In Jupyter notebooks, call `plot()` to see where the grid points are placed:

```python
ax = create_axis(AxisType.CHEBYSHEV, 20, 0.0, 1.0)
ax.plot()
```

For periodic axes, `plot()` renders the points on a circle to emphasize the
wrap-around topology.

## Indexing

Axes support integer indexing. For periodic axes, the index wraps around
automatically:

```python
ax = create_axis(AxisType.EQUIDISTANT_PERIODIC, 8, 0.0, 2 * np.pi)
print(ax[0])   # 0.0
print(ax[8])   # 0.0  (wraps)
print(ax[-1])  # same as ax[7]
```

## Length

`len(ax)` returns the number of grid points:

```python
ax = create_axis(AxisType.EQUIDISTANT, 100, 0.0, 1.0)
print(len(ax))  # 100
```

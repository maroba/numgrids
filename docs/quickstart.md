# Quickstart

This page gives a five-minute tour of the main features in numgrids. By the
end you will know how to create axes and grids, compute derivatives, integrate,
interpolate, and run vector calculus on a spherical grid.

## Setup

Every example on this page uses the following imports:

```python
from numgrids import *
import numpy as np
```

## 1. Create an axis

An *axis* is a one-dimensional set of grid points. Use the `create_axis`
factory to build one:

```python
ax = create_axis(AxisType.EQUIDISTANT, 50, 0.0, 1.0)
print(ax)          # EquidistantAxis(50 points from 0.0 to 1.0)
print(ax.coords)   # array([0.  , 0.02040816, ..., 1.  ])
```

numgrids ships four axis types: `EQUIDISTANT`, `EQUIDISTANT_PERIODIC`,
`CHEBYSHEV`, and `LOGARITHMIC`. See the [Axes guide](guide/axes.md) for details
on when to pick which.

## 2. Build a grid

A `Grid` is the tensor product of one or more axes:

```python
ax_x = create_axis(AxisType.CHEBYSHEV, 40, -1.0, 1.0)
ax_y = create_axis(AxisType.CHEBYSHEV, 40, -1.0, 1.0)
grid = Grid(ax_x, ax_y)

print(grid.shape)  # (40, 40)
print(grid.ndims)  # 2
```

The `meshed_coords` property returns NumPy arrays ready for vectorized
evaluation:

```python
X, Y = grid.meshed_coords
```

## 3. Sample a function on the grid

Evaluate any function on the meshed coordinates:

```python
f = np.sin(np.pi * X) * np.cos(np.pi * Y)
```

The result `f` has shape `grid.shape` and can be passed to every numgrids
operator.

## 4. Compute derivatives

The `Diff` class creates a partial-derivative operator. Pass the grid,
derivative order, and (for multi-dimensional grids) the axis index:

```python
d_dx = Diff(grid, order=1, axis_index=0)
d_dy = Diff(grid, order=1, axis_index=1)

df_dx = d_dx(f)
df_dy = d_dy(f)
```

Because both axes are Chebyshev, numgrids automatically uses spectral
differentiation -- no finite-difference accuracy parameter needed.

For quick one-off derivatives the convenience function `diff` does the same
thing with caching:

```python
df_dx = diff(grid, f, order=1, axis_index=0)
```

## 5. Integrate over the domain

```python
result = integrate(grid, f)
print(result)  # close to 0.0 (sin * cos over symmetric domain)
```

Or use the class form for repeated integration on the same grid:

```python
I = Integral(grid)
print(I(f))
```

## 6. Interpolate

Evaluate the gridded data at arbitrary points:

```python
val = interpolate(grid, f, (0.3, 0.7))
print(val)  # interpolated value at x=0.3, y=0.7
```

You can also pass a list of points or an entirely different `Grid`:

```python
fine_grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 80, -1.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 80, -1.0, 1.0),
)
f_fine = interpolate(grid, f, fine_grid)
```

## 7. Vector calculus on a sphere

`SphericalGrid` provides gradient, divergence, curl, and Laplacian in
spherical coordinates $(r, \theta, \phi)$ out of the box:

```python
grid = SphericalGrid(
    create_axis(AxisType.CHEBYSHEV, 25, 0.5, 2.0),              # r
    create_axis(AxisType.CHEBYSHEV, 20, 0.1, np.pi - 0.1),      # theta
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2 * np.pi),  # phi
)

R, Theta, Phi = grid.meshed_coords

# A scalar field
f = R**2 * np.sin(Theta)

# Laplacian (should be close to 6 sin(theta) + r^2 * ... )
lap_f = grid.laplacian(f)

# Gradient returns one component per coordinate
grad_f = grid.gradient(f)   # (grad_r, grad_theta, grad_phi)
```

Cylindrical and polar grids work the same way -- see the
[Curvilinear guide](guide/curvilinear.md).

## Where to go next

| Topic | Page |
|-------|------|
| All axis types and when to use each | [Axes](guide/axes.md) |
| Grid properties, refinement, and caching | [Grids](guide/grids.md) |
| Differentiation strategies and sparse matrices | [Differentiation](guide/differentiation.md) |
| Numerical integration | [Integration](guide/integration.md) |
| Interpolating onto arbitrary points or grids | [Interpolation](guide/interpolation.md) |
| Spherical, cylindrical, polar, and custom coordinates | [Curvilinear grids](guide/curvilinear.md) |
| Dirichlet, Neumann, and Robin boundary conditions | [Boundary conditions](guide/boundary-conditions.md) |
| Saving and loading grids | [I/O](guide/io.md) |
| Multigrid hierarchies | [MultiGrid](guide/multigrid.md) |
| Adaptive mesh refinement | [AMR](guide/amr.md) |

# Saving and loading grids

numgrids can persist grids and associated data arrays to disk using NumPy's
`.npz` format. This is useful for checkpointing long-running computations,
sharing data between scripts, or archiving results.

```python
from numgrids import *
import numpy as np
```

## Saving a grid

Use `save_grid()` to write a grid and any number of named data arrays to a
file:

```python
grid = Grid(
    create_axis(AxisType.CHEBYSHEV, 40, 0.0, 1.0),
    create_axis(AxisType.CHEBYSHEV, 40, 0.0, 1.0),
)
X, Y = grid.meshed_coords

temperature = np.sin(np.pi * X) * np.cos(np.pi * Y)
pressure = X**2 + Y**2

save_grid("simulation.npz", grid, temperature=temperature, pressure=pressure)
```

The data arrays are passed as keyword arguments. Each must have the same
shape as `grid.shape`.

## Loading a grid

Use `load_grid()` to reconstruct the grid and retrieve the saved arrays:

```python
grid, data = load_grid("simulation.npz")

temperature = data["temperature"]
pressure = data["pressure"]

print(grid.shape)           # (40, 40)
print(temperature.shape)    # (40, 40)
```

The `.npz` suffix can be omitted -- it will be appended automatically if
missing.

## What gets stored

The file contains:

- **JSON metadata** describing the grid structure: grid class, axis types,
  axis parameters (number of points, low, high, periodic, name).
- **Named arrays** for each data keyword passed to `save_grid()`.

The grid is reconstructed from the metadata, not from raw coordinate arrays.
This means the loaded grid is an exact replica of the original (same axis
types, same parameters), not just a grid with matching coordinates.

## Roundtrip example

```python
# Create and save
ax_r = create_axis(AxisType.LOGARITHMIC, 30, 0.01, 10.0)
ax_phi = create_axis(AxisType.EQUIDISTANT_PERIODIC, 40, 0, 2 * np.pi)
grid = PolarGrid(ax_r, ax_phi)

R, Phi = grid.meshed_coords
field = R * np.cos(Phi)

save_grid("polar_field.npz", grid, field=field)

# Load and verify
grid2, data2 = load_grid("polar_field.npz")

print(type(grid2))                 # <class 'numgrids.api.PolarGrid'>
print(grid2.shape)                 # (30, 40)
print(np.allclose(data2["field"], field))  # True
```

## Supported grid types

`save_grid` and `load_grid` work with all grid types:

- `Grid`
- `SphericalGrid`, `CylindricalGrid`, `PolarGrid`
- `EllipticCylindricalGrid`, `ProlateSpheroidalGrid`, `OblateSpheroidalGrid`
  (including the focal parameter `a`)

All axis types (`EquidistantAxis`, `ChebyshevAxis`, `LogAxis`, including
periodic equidistant axes, the radial and colatitude axes of curvilinear
grids and the axes on unbounded domains) are fully supported. Axis metadata
such as `periodic`, `name` and the `scale` of unbounded axes is preserved
across the roundtrip.

```{note}
`CurvilinearGrid` with custom scale factors cannot be saved directly
because the scale-factor callables are not serializable. The built-in
coordinate grids (`SphericalGrid`, `CylindricalGrid`, `PolarGrid`) work
because their scale factors are reconstructed from the class type.
```

## Methods on the grid

`grid.save(path, **data)` and `Grid.load(path)` are shortcuts for
`save_grid` and `load_grid`:

```python
u = grid.evaluate(lambda *coords: coords[0] ** 2)
grid.save("result.npz", u=u)
grid, data = Grid.load("result.npz")
```

## xarray and NetCDF

With the optional [xarray](https://xarray.dev) dependency
(`pip install numgrids[xarray]`), grids and meshed data can be converted to
labelled `xarray.Dataset` objects -- convenient for analysis, plotting and
writing NetCDF files:

```python
grid = Grid(x=chebyshev(30, 0, 1), y=periodic(40))
X, Y = grid.meshed_coords

ds = grid.to_xarray(u=X * np.sin(Y), v=X**2)   # or to_xarray(grid, ...)
ds.to_netcdf("result.nc")
```

Each axis becomes a dimension coordinate named after the axis (`x0`, `x1`,
... for unnamed axes). The full grid definition is stored in the
attributes, so the grid can be reconstructed exactly:

```python
import xarray as xr

grid, data = Grid.from_xarray(xr.open_dataset("result.nc"))
u = data["u"]
```

`from_xarray` also accepts single `DataArray` objects and datasets that were
not created by numgrids; in that case the axis types are inferred from the
coordinate values (equidistant, Chebyshev or logarithmic; other axis types
are only restored from datasets written by `to_xarray`).

## VTK export for ParaView

For 3D rendering, slicing, streamlines and the like, fields can be written
to VTK files, which [ParaView](https://www.paraview.org),
[VisIt](https://visit-dav.github.io/visit-website/) and other VTK-based
tools read. No additional package is needed: numgrids writes the files
itself.

```python
disk = PolarGrid(radial(24), periodic(48))
R, Phi = disk.meshed_coords
u = (1 - R**2) * R**2 * np.cos(2 * Phi)

disk.to_vtk("disk.vts", u=u, weight=1 - R**2)
```

Every keyword argument is a field of the file, an array of grid shape. The
method returns the path of the file written. `write_vtk("disk.vts", disk,
u=u)` from `numgrids.vtk` does the same.

### Vector fields

A tuple with one array per axis is a vector field. As everywhere in
numgrids, it is given by its physical components along the coordinate lines
of the grid, for example $(v_r, v_\varphi)$ on a `PolarGrid`. Viewers expect
Cartesian components, so the vectors are converted, with the unit vectors
derived from the Cartesian mapping of the grid:

```python
velocity = (np.zeros(disk.shape), R)       # (v_r, v_phi): a rigid rotation
disk.to_vtk("flow.vts", u=u, velocity=velocity)
```

In the file, `velocity` is $(-y, x, 0)$. The conversion is also available
on its own:

```python
from numgrids.vtk import cartesian_components

vx, vy = cartesian_components(disk, *velocity)
```

### Time series

A time series is written as one `.vts` file per time step plus a small
`.pvd` collection file that lists the files with their times. Open the
`.pvd` file in ParaView to step through the series. The result of
[`evolve`](time-evolution.md) writes itself:

```python
X, Y = disk.cartesian_coords
u0 = np.exp(-20 * ((X - 0.3) ** 2 + Y**2))
result = evolve(disk, 0.05 * disk.laplacian_operator(), u0, (0, 0.5),
                bcs=0, method="BDF", t_eval=np.linspace(0, 0.5, 11))

result.to_vtk("heat.pvd", name="T")        # heat.pvd, heat_0000.vts, ..., heat_0010.vts
```

For a coupled system, pass one name per field (`name=("u", "v")`). Any
other sequence of frames is written with `grid.to_vtk_series`; the frames
of a field have the shape `(nt, *grid.shape)`:

```python
t = np.linspace(0, 1, 5)
omega = 1 + t[:, None, None]               # a rotation that speeds up
frames = np.cos(Phi - omega * t[:, None, None]) * R
spin = (np.zeros((5,) + disk.shape), omega * R)

disk.to_vtk_series("spin.pvd", t=t, u=frames, velocity=spin)
```

### What the file contains

- **Format:** VTK XML *StructuredGrid* (`.vts`): the points of the grid in
  their natural order, so that the viewer knows the cells without a
  connectivity list.
- **Points:** the Cartesian coordinates of the grid points, for curvilinear
  grids from their Cartesian mapping. 1D and 2D grids lie in the plane
  $z = 0$; a `SphericalSurfaceGrid` is the sphere in space.
- **Fields:** point data in double precision. Scalars are stored under
  their name, vectors with three Cartesian components. A complex field
  `psi` becomes the arrays `psi_real`, `psi_imag` and `psi_abs` (a complex
  vector field `E` becomes `E_real` and `E_imag`).
- **Closed surfaces and volumes:** a periodic direction ends one point
  before the period is complete, which would show as a gap. It is closed by
  repeating the first point. Likewise, a [`radial`](axes.md) axis has no
  point at the origin and a `colatitude` axis none at the poles; these
  points are added, with values from the spectral interpolation. The values
  at the grid points themselves are written unchanged. With `close=False`
  the file holds exactly the grid points.
- **Encoding:** the numbers are stored base64-encoded (`binary=True`, the
  default) or as text (`binary=False`). Both keep the full precision.

### Limits

- Grids with more than three dimensions cannot be written. Export a
  lower-dimensional slice on a grid of its own.
- A curvilinear grid needs a Cartesian mapping (all built-in grids have
  one; pass `cartesian=` to a `CurvilinearGrid` of your own).
- Only the origin of a `radial` axis and the poles of a `colatitude` axis
  are filled in. A grid that merely stops short of the origin or of the
  poles, such as `chebyshev(n, 0.01, 1)` for the radius, keeps its small
  hole.
- Where the coordinate system is singular *at a grid point* (for example
  at $r = 0$ of a `chebyshev(n, 0, 1)` radial axis), the unit vectors have
  no direction, and components along them are dropped there.
- The data is not compressed, and in a time series every file repeats the
  points. Large 3D series take a lot of space.
- The files are meant for visualization. They cannot be read back into
  numgrids; use `save_grid` or xarray for that.

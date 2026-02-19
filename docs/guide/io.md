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
- `SphericalGrid`
- `CylindricalGrid`
- `PolarGrid`

All axis types (`EquidistantAxis`, `ChebyshevAxis`, `LogAxis`, including
periodic equidistant axes) are fully supported. Axis metadata such as
`periodic` and `name` is preserved across the roundtrip.

```{note}
`CurvilinearGrid` with custom scale factors cannot be saved directly
because the scale-factor callables are not serializable. The built-in
coordinate grids (`SphericalGrid`, `CylindricalGrid`, `PolarGrid`) work
because their scale factors are reconstructed from the class type.
```

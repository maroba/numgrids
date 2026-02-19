# MultiGrid

The `MultiGrid` class creates a hierarchy of grids at progressively coarser
resolutions over the same domain. This is the data structure needed for
multigrid iterative solvers, which accelerate convergence of elliptic PDE
solvers by transferring corrections between resolution levels.

```python
from numgrids import *
import numpy as np
```

## Creating a MultiGrid

Pass the axes for the finest grid and an optional `min_size` parameter that
sets the minimum number of points per axis on the coarsest level:

```python
ax_x = create_axis(AxisType.EQUIDISTANT, 64, 0.0, 1.0)
ax_y = create_axis(AxisType.EQUIDISTANT, 64, 0.0, 1.0)

mg = MultiGrid(ax_x, ax_y, min_size=4)
```

Starting from the finest grid (64 x 64), the constructor repeatedly halves
each axis until one of them would drop below `min_size`. The default
`min_size` is 2.

## Accessing levels

The `levels` property returns a list of `Grid` objects, ordered from finest
to coarsest:

```python
for i, level in enumerate(mg.levels):
    print(f"Level {i}: shape = {level.shape}")
```

Output:

```text
Level 0: shape = (64, 64)
Level 1: shape = (32, 32)
Level 2: shape = (16, 16)
Level 3: shape = (8, 8)
Level 4: shape = (4, 4)
```

- `mg.levels[0]` is always the finest grid (the one you passed in).
- `mg.levels[-1]` is the coarsest grid.

## Transferring data between levels

The `transfer()` method moves an array from one level to an adjacent level
using interpolation:

```python
# Sample a function on the finest grid
X, Y = mg.levels[0].meshed_coords
f_fine = np.sin(np.pi * X) * np.cos(np.pi * Y)

# Restrict (fine -> coarse): level 0 -> level 1
f_coarse = mg.transfer(f_fine, level_from=0, level_to=1)
print(f_coarse.shape)  # (32, 32)

# Prolongate (coarse -> fine): level 1 -> level 0
f_back = mg.transfer(f_coarse, level_from=1, level_to=0)
print(f_back.shape)    # (64, 64)
```

```{note}
Transfers are only allowed between **adjacent** levels (i.e.
`|level_from - level_to| == 1`). To move data across multiple levels,
chain the transfers.
```

The `method` parameter controls the interpolation order used for the
transfer (default `"linear"`):

```python
f_coarse = mg.transfer(f_fine, level_from=0, level_to=1, method="cubic")
```

## Mixed axis types

`MultiGrid` supports all axis types. Each level preserves the axis type,
domain bounds, periodicity, and name of the original axes:

```python
mg = MultiGrid(
    create_axis(AxisType.CHEBYSHEV, 32, 0.0, 1.0, name="x"),
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 64, 0.0, 2 * np.pi, name="theta"),
    min_size=4,
)

for i, level in enumerate(mg.levels):
    ax0 = level.get_axis(0)
    ax1 = level.get_axis(1)
    print(f"Level {i}: {type(ax0).__name__}({len(ax0)}), "
          f"{type(ax1).__name__}({len(ax1)}, periodic={ax1.periodic})")
```

## Use case: multigrid solver skeleton

A typical V-cycle multigrid solver follows this pattern:

1. **Restrict** the residual from the fine grid to the coarse grid.
2. **Solve** (or smooth) on the coarse grid.
3. **Prolongate** the correction back to the fine grid.
4. **Update** the fine-grid solution.

`MultiGrid` provides the grid hierarchy and the `transfer()` method for
steps 1 and 3. The smoothing and solving steps are problem-specific and left
to the user.

```python
# Pseudocode for a single V-cycle
residual = compute_residual(mg.levels[0], u, rhs)

# Restrict residual to each coarser level
r = [residual]
for k in range(len(mg.levels) - 1):
    r.append(mg.transfer(r[-1], k, k + 1))

# Solve on the coarsest level
e = solve_coarse(mg.levels[-1], r[-1])

# Prolongate corrections back up
for k in range(len(mg.levels) - 2, -1, -1):
    e = mg.transfer(e, k + 1, k)
    u = u + e
    u = smooth(mg.levels[k], u, rhs)
```

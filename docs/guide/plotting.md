# Plotting

Plotting requires matplotlib, which is an optional dependency:

```bash
pip install numgrids[plot]
```

```python
from numgrids import *
import numpy as np
```

## Plotting functions on a grid

`grid.plot(f)` picks a suitable visualization based on the grid's
dimension and returns the matplotlib `Axes`:

```python
# 1D: line plot over the coordinate
grid = Grid(x=chebyshev(40, 0, np.pi))
grid.plot(np.sin(grid.coords))

# 2D: color plot
grid = Grid(x=chebyshev(30, 0, 1), y=periodic(40))
X, Y = grid.meshed_coords
ax = grid.plot(X * np.sin(Y), cmap="viridis")
ax.set_title("my field")
```

Curvilinear 2D grids are drawn in **Cartesian space**, with periodic axes
closed seamlessly -- a `PolarGrid` shows up as a disk:

```python
grid = PolarGrid(chebyshev(30, 0, 1), periodic(48))
R, Phi = grid.meshed_coords
grid.plot(R**2 * np.cos(3 * Phi))
grid.plot(R**2 * np.cos(3 * Phi), cartesian=False)   # in (r, phi) space
```

For grids with **three or more dimensions**, a 2D slice is shown. By
default the trailing axes are fixed at their middle index; choose the slice
with `fixed`:

```python
grid = Grid(x=chebyshev(20, 0, 1), y=chebyshev(20, 0, 1), z=chebyshev(20, 0, 1))
X, Y, Z = grid.meshed_coords
f = X * Y * Z
grid.plot(f)                     # slice at the middle of z
grid.plot(f, fixed={"x": 0})     # the y-z plane at the first x point
```

Further options: `ax=` to draw into existing axes, `colorbar=False`, and
any keyword argument of `ax.plot` (1D) or `ax.pcolormesh` (2D). The same
function is available as {func}`numgrids.plots.plot_field`.

## Animations

`grid.animate(frames, t)` animates a field that changes in time. `frames`
holds the field at successive times, with shape `(nt, *grid.shape)`, and
`t` the times, which are shown in the title. Every frame is drawn like
`grid.plot` draws a single field: a line for 1D grids, a color plot for 2D
grids (in Cartesian space for curvilinear grids), a 2D slice for higher
dimensions (choose it with `fixed`).

The result of [`evolve`](time-evolution.md) knows its grid, so
`result.animate()` is all it takes. The heat equation on a rod:

```python
grid = Grid(x=chebyshev(40, 0, 1))
result = evolve(grid, 0.1 * grid.laplacian_operator(), np.sin(np.pi * grid.coords),
                (0, 1), bcs=0, method="BDF", t_eval=np.linspace(0, 1, 41))

anim = result.animate()          # same as grid.animate(result.u, result.t)
```

and on a disk, where a hot spot spreads out:

```python
disk = PolarGrid(radial(16), periodic(32))
X, Y = disk.cartesian_coords
u0 = np.exp(-20 * ((X - 0.3) ** 2 + Y ** 2))
result = evolve(disk, 0.05 * disk.laplacian_operator(), u0, (0, 0.5),
                bcs=0, method="BDF", t_eval=np.linspace(0, 0.5, 26))

anim = result.animate(cmap="inferno", title="temperature", fps=10)
```

The return value is a matplotlib
[`FuncAnimation`](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html).
Keep a reference to it (here `anim`), otherwise the animation is discarded
before it is shown.

The vertical axis (1D) and the color scale (2D) are the same in all frames.
By default they cover the values of all frames; set them with `vmin` and
`vmax` (or `ylim` in 1D). Further options:

| Option | Meaning |
|---|---|
| `interval`, `fps` | delay between frames in milliseconds (default 50), or frames per second |
| `part` | for complex fields: `"real"` (default), `"imag"`, `"abs"`, `"abs2"` or a function |
| `field` | `result.animate(field=1)`: the field of a coupled system to show (default: the first) |
| `ax` | draw into existing axes |
| `title` | a title; the time is appended to it |
| `colorbar`, `cartesian`, `fixed` | as for `grid.plot` |

Other keyword arguments go to `ax.plot` (1D) or `ax.pcolormesh` (2D), for
example `cmap`.

### Saving an animation

Pass `save=` to write the animation to a file right away, or save it later:

```python
result.animate(save="disk.gif")

from numgrids.plots import save_animation

save_animation(anim, "disk.html")       # a self-contained page with playback controls
save_animation(anim, "slow.gif", fps=4)
```

The writer is chosen from the file name. GIF files are written with Pillow,
which is installed together with matplotlib, and HTML pages need nothing
else. Other formats such as `.mp4` use matplotlib's movie writer, for which
[ffmpeg](https://ffmpeg.org) has to be installed; if it is missing, you get
an error that says so. `anim.save(...)`, the method of matplotlib, works as
well.

### In a notebook

Jupyter shows an animation as an interactive player:

```python
from IPython.display import HTML

HTML(anim.to_jshtml())
```

The figure of the animation would be shown a second time as a still image
below the cell. Close it with `plt.close()` to avoid that.

## Visualizing grids and axes

Without arguments, `grid.plot()` shows the distribution of grid points and
`axis.plot()` shows the points of a single axis.

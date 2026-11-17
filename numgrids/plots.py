"""Plotting helpers (require matplotlib, installed via ``numgrids[plot]``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from numgrids.interpol import Interpolator

if TYPE_CHECKING:
    from numgrids.grids import Grid


def require_matplotlib():
    """Import and return ``matplotlib.pyplot`` with a helpful error message."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "Plotting requires matplotlib. Install it with `pip install numgrids[plot]`."
        ) from e
    return plt


def _label(grid: Grid, i: int) -> str:
    name = grid.axes[i].name
    return f"${name}$" if name else f"axis-{i}"


def plot_field(
    grid: Grid,
    f: NDArray,
    ax=None,
    fixed: dict | None = None,
    cartesian: bool = True,
    colorbar: bool = True,
    **kwargs,
):
    """Plot a meshed function on a grid.

    * **1D grids**: line plot of *f* over the coordinate.
    * **2D grids**: color plot (``pcolormesh``).  Curvilinear grids with a
      Cartesian mapping (e.g. :class:`~numgrids.PolarGrid`) are drawn in
      Cartesian space, with periodic axes closed seamlessly.
    * **Higher dimensions**: a 2D slice; all but two axes are fixed via
      *fixed* (default: the middle index of the trailing axes).

    Parameters
    ----------
    grid : Grid
        The grid.
    f : NDArray
        Meshed function values of shape ``grid.shape`` (real part is
        plotted for complex arrays).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    fixed : dict, optional
        For grids with more than two dimensions: ``{axis: index}`` for the
        axes to fix (axis given by index or name).  Exactly ``ndims - 2``
        axes must be fixed.
    cartesian : bool
        Draw curvilinear 2D grids in Cartesian space (default ``True``).
    colorbar : bool
        Add a colorbar for 2D plots (default ``True``).
    **kwargs
        Passed on to ``ax.plot`` (1D) or ``ax.pcolormesh`` (2D).

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = require_matplotlib()
    f = np.real(grid._check_field(f))
    if ax is None:
        _, ax = plt.subplots()

    if grid.ndims == 1:
        ax.plot(grid.coords, f, **kwargs)
        ax.set_xlabel(_label(grid, 0))
        return ax

    # Reduce to 2D
    keep = list(range(grid.ndims))
    index = [slice(None)] * grid.ndims
    if grid.ndims > 2:
        if fixed is None:
            fixed = {i: grid.shape[i] // 2 for i in range(2, grid.ndims)}
        fixed = {grid.axis_index(k): v for k, v in fixed.items()}
        if len(fixed) != grid.ndims - 2:
            raise ValueError(f"Fix exactly {grid.ndims - 2} axes via `fixed`, got {len(fixed)}.")
        for i, idx in fixed.items():
            index[i] = idx
            keep.remove(i)
    index = tuple(index)
    i0, i1 = keep
    F = f[index]

    use_cartesian = cartesian and grid.ndims == 2
    if use_cartesian:
        try:
            cart = grid.cartesian_coords
        except NotImplementedError:
            cart = ()
        # e.g. a 2D surface embedded in 3D space cannot be drawn flat
        use_cartesian = len(cart) == 2
        if use_cartesian:
            X, Y = cart
    if not use_cartesian:
        X, Y = grid.meshed_coords[i0][index], grid.meshed_coords[i1][index]

    # A radial axis through the origin has no point at r = 0: add the center
    # (value from spectral interpolation) so that disks are drawn without a hole.
    from numgrids.axes import RadialAxis
    if use_cartesian and isinstance(grid.axes[i0], RadialAxis):
        center = float(np.real(grid.interpolate(f, (0.0, grid.axes[i1].coords[0]), method="spectral")))
        F = np.vstack([np.full((1, F.shape[1]), center), F])
        X = np.vstack([np.zeros((1, X.shape[1])), X])
        Y = np.vstack([np.zeros((1, Y.shape[1])), Y])

    # Close periodic axes so that there is no gap in the plot.
    for local, i in enumerate((i0, i1)):
        axis = grid.axes[i]
        if axis.periodic:
            first = np.take(F, [0], axis=local)
            F = np.concatenate([F, first], axis=local)
            if use_cartesian:
                X = np.concatenate([X, np.take(X, [0], axis=local)], axis=local)
                Y = np.concatenate([Y, np.take(Y, [0], axis=local)], axis=local)
            else:
                X = np.concatenate([X, np.take(X, [0], axis=local) + (axis.period if local == 0 else 0)],
                                   axis=local)
                Y = np.concatenate([Y, np.take(Y, [0], axis=local) + (axis.period if local == 1 else 0)],
                                   axis=local)

    kwargs.setdefault("shading", "gouraud")
    mesh = ax.pcolormesh(X, Y, F, **kwargs)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax)
    if use_cartesian:
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    else:
        ax.set_xlabel(_label(grid, i0))
        ax.set_ylabel(_label(grid, i1))
    return ax


class Plotter:
    """Plotting utility for visualizing meshed functions on a grid.

    Wraps matplotlib to provide quick line plots of interpolated
    function values along user-specified coordinate slices.
    """

    def __init__(self, grid: Grid) -> None:
        """Create a plotter for the given grid.

        Parameters
        ----------
        grid : Grid
            The grid on which the data lives.
        """
        self.grid = grid
        self.fig = None
        self.ax = None

    def plot(self, f: NDArray, *coords) -> None:
        """Plot interpolated function values along a coordinate path.

        The values are plotted against the first coordinate that is given
        as an array.

        Parameters
        ----------
        f : NDArray
            Meshed function values on the grid.
        *coords
            Coordinate arrays or scalar values defining the path to plot.
            At least one coordinate must be an array; scalar coordinates
            are broadcast to match.
        """
        plt = require_matplotlib()

        num_points = None
        param = None
        for i, c in enumerate(coords):
            if hasattr(c, "__len__"):
                num_points = len(c)
                param = i
                break
        if num_points is None:
            raise ValueError("At least one coordinate must be an array.")

        parsed_coords = []
        for c in coords:
            if hasattr(c, "__len__"):
                parsed_coords.append(c)
            else:
                parsed_coords.append([c] * num_points)
        coords = parsed_coords

        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.subplots(1, 1)

        inter = Interpolator(self.grid, f)
        points = np.array(coords).T
        values = inter(points)

        self.ax.plot(np.asarray(coords[param]), values)
        self.ax.set_xlabel(_label(self.grid, param))

    def show(self) -> None:
        """Display the current plot."""
        if self.fig is not None:
            self.fig.show()

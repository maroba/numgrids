"""Plotting helpers (require matplotlib, installed via ``numgrids[plot]``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from numgrids.interpol import Interpolator

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation

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


class _Mesh2D:
    """Geometry of a 2D color plot of a field: the mesh and the values on it.

    Shared by :func:`plot_field` and :func:`animate_field`, so that both
    draw the same picture.
    """

    def __init__(self, grid: Grid, fixed: dict | None = None, cartesian: bool = True) -> None:
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
        self.grid = grid
        self.index = tuple(index)
        i0, i1 = self.axes = tuple(keep)

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
            X, Y = grid.meshed_coords[i0][self.index], grid.meshed_coords[i1][self.index]
        self.cartesian = use_cartesian

        # A radial axis through the origin has no point at r = 0: add the center
        # (value from spectral interpolation) so that disks are drawn without a hole.
        from numgrids.axes import RadialAxis
        self._center = use_cartesian and isinstance(grid.axes[i0], RadialAxis)
        if self._center:
            X = np.vstack([np.zeros((1, X.shape[1])), X])
            Y = np.vstack([np.zeros((1, Y.shape[1])), Y])

        # Close periodic axes so that there is no gap in the plot.
        self._periodic = [local for local, i in enumerate(self.axes) if grid.axes[i].periodic]
        for local in self._periodic:
            axis = grid.axes[self.axes[local]]
            if use_cartesian:
                X = np.concatenate([X, np.take(X, [0], axis=local)], axis=local)
                Y = np.concatenate([Y, np.take(Y, [0], axis=local)], axis=local)
            else:
                X = np.concatenate([X, np.take(X, [0], axis=local) + (axis.period if local == 0 else 0)],
                                   axis=local)
                Y = np.concatenate([Y, np.take(Y, [0], axis=local) + (axis.period if local == 1 else 0)],
                                   axis=local)
        self.X, self.Y = X, Y

    def values(self, f: NDArray) -> NDArray:
        """The values of a (real) field of grid shape on the mesh."""
        grid = self.grid
        F = f[self.index]
        if self._center:
            phi = grid.axes[self.axes[1]].coords[0]
            center = float(np.real(grid.interpolate(f, (0.0, phi), method="spectral")))
            F = np.vstack([np.full((1, F.shape[1]), center), F])
        for local in self._periodic:
            F = np.concatenate([F, np.take(F, [0], axis=local)], axis=local)
        return F

    def label_axes(self, ax) -> None:
        if self.cartesian:
            ax.set_aspect("equal")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
        else:
            ax.set_xlabel(_label(self.grid, self.axes[0]))
            ax.set_ylabel(_label(self.grid, self.axes[1]))


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
    geometry = _Mesh2D(grid, fixed, cartesian) if grid.ndims > 1 else None
    if ax is None:
        _, ax = plt.subplots()

    if geometry is None:
        ax.plot(grid.coords, f, **kwargs)
        ax.set_xlabel(_label(grid, 0))
        return ax

    kwargs.setdefault("shading", "gouraud")
    mesh = ax.pcolormesh(geometry.X, geometry.Y, geometry.values(f), **kwargs)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax)
    geometry.label_axes(ax)
    return ax


# -- animations ---------------------------------------------------------------

_PARTS: dict[str, Callable[[NDArray], NDArray]] = {
    "real": np.real,
    "imag": np.imag,
    "abs": np.abs,
    "abs2": lambda f: np.abs(f) ** 2,
}


def _time_format(t: NDArray) -> str:
    """A format with fixed width that resolves the time steps (the title should not jitter)."""
    largest = float(np.max(np.abs(t))) if t.size else 0.0
    steps = np.abs(np.diff(t))
    steps = steps[steps > 0]
    if not np.isfinite(largest) or (largest != 0 and not 1e-3 <= largest < 1e6):
        return "{:.3e}"
    decimals = 2 if not steps.size else int(np.clip(np.ceil(-np.log10(steps.min())) + 1, 0, 6))
    return f"{{:.{decimals}f}}"


def animate_field(
    grid: Grid,
    frames: NDArray,
    t: NDArray | None = None,
    ax=None,
    fixed: dict | None = None,
    cartesian: bool = True,
    colorbar: bool = True,
    part: str | Callable[[NDArray], NDArray] = "real",
    interval: float | None = None,
    fps: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    repeat: bool = True,
    save: str | Path | None = None,
    **kwargs,
) -> FuncAnimation:
    """Animate a time-dependent field on a grid.

    The frames are drawn like :func:`plot_field` draws a single field: a
    line for 1D grids, a color plot for 2D grids (in Cartesian space for
    curvilinear grids) and a 2D slice for higher dimensions.  The axis
    limits (1D) and the color scale (2D) are the same in all frames; by
    default they cover the values of all frames.

    Parameters
    ----------
    grid : Grid
        The grid.
    frames : NDArray
        The field at successive times, shape ``(nt, *grid.shape)``, e.g.
        the attribute ``u`` of an :class:`~numgrids.EvolutionResult`.
    t : array_like, optional
        The times of the frames, shape ``(nt,)``.  They are shown in the
        title as ``t = ...``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    fixed : dict, optional
        The slice to show for grids with more than two dimensions, see
        :func:`plot_field`.
    cartesian : bool
        Draw curvilinear 2D grids in Cartesian space (default ``True``).
    colorbar : bool
        Add a colorbar for 2D plots (default ``True``).
    part : str or callable
        What to show of the values: ``"real"`` (default), ``"imag"``,
        ``"abs"`` or ``"abs2"`` (the squared modulus), or a function that
        is applied to the frames.  Only matters for complex fields.
    interval : float, optional
        Delay between frames in milliseconds (default 50).
    fps : float, optional
        Frames per second, as an alternative to *interval*.
    vmin, vmax : float, optional
        Limits of the color scale (2D) or of the vertical axis (1D).
        Default: the smallest and largest value of all frames.
    ylim : (float, float), optional
        Limits of the vertical axis for 1D grids, in place of *vmin* and
        *vmax*.
    title : str, optional
        Title of the plot; the time is appended to it.
    repeat : bool
        Start again after the last frame (default ``True``).
    save : str or Path, optional
        Also save the animation to this file, see :func:`save_animation`.
    **kwargs
        Passed on to ``ax.plot`` (1D) or ``ax.pcolormesh`` (2D), e.g. ``cmap``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Keep a reference to it: the animation stops when it is garbage
        collected.  In a notebook, show it with
        ``HTML(anim.to_jshtml())``.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, periodic
    >>> grid = Grid(x=periodic(64))
    >>> t = np.linspace(0, 2 * np.pi, 40)
    >>> frames = np.sin(grid.coords[None, :] - t[:, None])      # a travelling wave
    >>> anim = grid.animate(frames, t)                          # doctest: +SKIP
    >>> anim.save("wave.gif")                                   # doctest: +SKIP
    """
    plt = require_matplotlib()
    from matplotlib.animation import FuncAnimation

    if isinstance(frames, tuple):
        raise TypeError(
            "Got a tuple of fields (a coupled system). Animate one of them, e.g. frames[0], "
            "or use the `field` argument of EvolutionResult.animate."
        )
    frames = np.asarray(frames)
    if frames.ndim != grid.ndims + 1 or frames.shape[1:] != grid.shape or not len(frames):
        raise ValueError(
            f"`frames` has shape {frames.shape}, expected (nt, *grid.shape) with grid shape {grid.shape}."
        )
    nt = len(frames)
    if t is not None:
        t = np.asarray(t, dtype=float)
        if t.shape != (nt,):
            raise ValueError(f"Expected {nt} time values (one per frame), got an array of shape {t.shape}.")
    if callable(part):
        select = part
    elif part in _PARTS:
        select = _PARTS[part]
    else:
        raise ValueError(f"Unknown part {part!r}. Use one of {sorted(_PARTS)} or a function.")
    if interval is not None and fps is not None:
        raise ValueError("Give either `interval` or `fps`, not both.")
    if fps is not None:
        interval = 1000.0 / fps
    elif interval is None:
        interval = 50.0

    geometry = _Mesh2D(grid, fixed, cartesian) if grid.ndims > 1 else None
    if ax is None:
        _, ax = plt.subplots()

    def frame(k: int) -> NDArray:
        return np.asarray(select(frames[k]), dtype=float)

    def limits(values: NDArray) -> tuple[float, float]:
        low = float(np.nanmin(values)) if vmin is None else vmin
        high = float(np.nanmax(values)) if vmax is None else vmax
        return low, high

    if geometry is None:
        (line,) = ax.plot(grid.coords, frame(0), **kwargs)
        ax.set_xlabel(_label(grid, 0))
        if ylim is None:
            low, high = limits(np.asarray(select(frames), dtype=float))
            margin = 0.05 * (high - low) or 0.5
            # a little room around the curve, but not beyond limits that were asked for
            ylim = (low - margin * (vmin is None), high + margin * (vmax is None))
        ax.set_ylim(*ylim)

        def draw(values: NDArray):
            line.set_ydata(values)
            return line
    else:
        kwargs.setdefault("shading", "gouraud")
        if "norm" not in kwargs:
            shown = np.asarray(select(frames[(slice(None),) + geometry.index]), dtype=float)
            kwargs["vmin"], kwargs["vmax"] = limits(shown)
        mesh = ax.pcolormesh(geometry.X, geometry.Y, geometry.values(frame(0)), **kwargs)
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax)
        geometry.label_axes(ax)

        def draw(values: NDArray):
            # Only the colors change: the mesh itself is built once.
            mesh.set_array(geometry.values(values))
            return mesh

    stamp = None if t is None else _time_format(t)

    def update(k: int):
        artists = [draw(frame(k))]
        if stamp is not None:
            time = "t = " + stamp.format(t[k])
            artists.append(ax.set_title(time if title is None else f"{title}, {time}"))
        return artists

    if title is not None:
        ax.set_title(title)
    anim = FuncAnimation(ax.figure, update, frames=nt, interval=interval, repeat=repeat, blit=False)
    if save is not None:
        save_animation(anim, save)
    return anim


def save_animation(anim: FuncAnimation, path: str | Path, fps: float | None = None, writer=None, **kwargs) -> Path:
    """Save an animation to a file, choosing the writer from the file name.

    * ``.gif``: written with Pillow (installed together with matplotlib).
    * ``.html``: a self-contained page with playback controls (the output
      of ``anim.to_jshtml()``); needs no additional software.
    * anything else (e.g. ``.mp4``): the default movie writer of
      matplotlib, usually `ffmpeg <https://ffmpeg.org>`_, which has to be
      installed separately.

    Parameters
    ----------
    anim : matplotlib.animation.Animation
        The animation, e.g. from :func:`animate_field`.
    path : str or Path
        Destination file.
    fps : float, optional
        Frames per second of the file.  Defaults to the speed of the
        animation.
    writer : str or matplotlib.animation.AbstractMovieWriter, optional
        The writer to use instead of the one chosen from the file name.
    **kwargs
        Passed on to ``anim.save`` (e.g. ``dpi``).

    Returns
    -------
    Path
        The path of the file written.

    Raises
    ------
    RuntimeError
        If the writer is not available on this system.
    """
    require_matplotlib()
    import matplotlib
    from matplotlib import animation

    path = Path(path)
    suffix = path.suffix.lower()
    if writer is None and suffix in (".html", ".htm"):
        path.write_text(anim.to_jshtml(fps=fps), encoding="utf-8")
        return path
    if writer is None:
        writer = "pillow" if suffix == ".gif" else matplotlib.rcParams["animation.writer"]
    if isinstance(writer, str) and not animation.writers.is_available(writer):
        raise RuntimeError(
            f"Cannot save '{path.name}': the matplotlib movie writer '{writer}' is not available on this "
            "system. Install it, or save the animation as a GIF ('.gif', needs Pillow) or as a "
            "self-contained HTML page ('.html')."
        )
    anim.save(str(path), writer=writer, fps=fps, **kwargs)
    return path


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

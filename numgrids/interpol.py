"""Interpolation of meshed functions at arbitrary locations."""

from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator, interp1d

from numgrids.axes import (
    Axis,
    ChebyshevAxis,
    ColatitudeAxis,
    EquidistantAxis,
    LogAxis,
    RadialAxis,
    UnboundedAxis,
)
from numgrids.grids import Grid

METHODS = ("linear", "quadratic", "cubic", "quintic", "spectral")


class Interpolator:
    """Interpolation of meshed function data on a grid.

    Available methods:

    * ``"linear"``, ``"cubic"`` (default), ``"quintic"`` -- spline
      interpolation via :class:`scipy.interpolate.RegularGridInterpolator`
      (``"quadratic"`` is available for 1D grids only),
    * ``"spectral"`` -- interpolation that matches the axis type and keeps
      spectral accuracy: barycentric polynomial interpolation on Chebyshev
      axes, trigonometric interpolation on periodic axes, barycentric
      interpolation in the computational variable on unbounded axes, and
      high-order local polynomial interpolation on all other axes.

    Periodic axes are handled correctly by all methods, including points
    between the last grid point and the end of the period.  On unbounded
    axes (:func:`~numgrids.semi_infinite`, :func:`~numgrids.infinite`) all
    methods work in the computational variable and cover the whole domain,
    also beyond the outermost grid point.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, Interpolator, chebyshev, periodic
    >>> grid = Grid(chebyshev(20, 0, 1), periodic(32, 0, 2 * np.pi))
    >>> X, Y = grid.meshed_coords
    >>> inter = Interpolator(grid, X ** 2 * np.cos(Y), method="spectral")
    >>> value = inter((0.5, 6.25))    # also between the last point and 2*pi
    >>> bool(abs(value - 0.25 * np.cos(6.25)) < 1e-12)
    True
    """

    def __init__(self, grid: Grid, f: NDArray, method: str = "cubic") -> None:
        """
        Create an interpolating function for the array data.

        Call the Interpolator object like a normal function to interpolate.

        Parameters
        ----------
        grid: Grid
            The grid on which the array f is meshed.
        f: numpy.ndarray
            The data to interpolate.
        method: str
            One of ``"linear"``, ``"quadratic"`` (1D only), ``"cubic"``,
            ``"quintic"`` or ``"spectral"``.
        """
        if method not in METHODS:
            raise ValueError(f"Unknown interpolation method {method!r}. Use one of {METHODS}.")
        if method == "quadratic" and grid.ndims > 1:
            raise ValueError("Method 'quadratic' is only available for 1D grids.")
        self.grid = grid
        self.method = method
        self.f = grid._check_field(f)
        # Radial axes through the origin are extended to the full diameter
        # [-R, R] (a Chebyshev axis) with the values at the antipodal points,
        # so that interpolation works across the center of the domain.
        self._axes, self._data = _extend_radial_axes(grid, self.f)

        if method == "spectral":
            self._inter = None
            return

        pad = 3
        coords = [axis.coords for axis in self._axes]
        data = self._data
        for i, axis in enumerate(self._axes):
            if axis.periodic:
                n = len(axis)
                p = min(pad, n)
                P = axis.period
                coords[i] = np.concatenate([axis.coords[-p:] - P, axis.coords, axis.coords[:p] + P])
                data = np.take(data, np.arange(-p, n + p), axis=i, mode="wrap")
            elif isinstance(axis, UnboundedAxis):
                # Splines in the computational variable xi: there the data is smooth
                # and the nodes are not spread over many orders of magnitude. The value
                # of the spectral interpolant at infinity closes the interval [-1, 1].
                coords[i], data = _compactify(axis, data, i)

        if grid.ndims > 1:
            self._inter = RegularGridInterpolator(tuple(coords), data, method=method)
        else:
            self._inter = interp1d(coords[0], data, kind=method)

    # ── public API ──────────────────────────────────────────────

    def __call__(self, locations) -> NDArray:
        """
        Return the interpolation for one or more points.

        Parameters
        ----------
        locations: float, tuple, list of tuples, array, zip, or Grid
            The location(s) where to interpolate:

            * a tuple with one coordinate per axis (or a single number for
              1D grids) -- a single point; a scalar is returned,
            * a list of tuples, an array of shape ``(M, ndims)``, a ``zip``
              of coordinate sequences, or (1D grids) a 1D array -- several
              points; an array of shape ``(M,)`` is returned,
            * a :class:`~numgrids.grids.Grid` -- all points of that grid;
              an array of shape ``grid.shape`` is returned.

        Returns
        -------
        The interpolated value or values.
        """
        if isinstance(locations, Grid):
            return self._on_grid(locations)

        points, single = _parse_locations(locations, self.grid.ndims)
        values = self._evaluate(points)
        return values[0] if single else values

    # ── internals ───────────────────────────────────────────────

    def _wrap(self, points: NDArray) -> NDArray:
        points = np.array(points, dtype=float, copy=True)
        for i, axis in enumerate(self._axes):
            if axis.periodic:
                points[:, i] = axis.low + np.mod(points[:, i] - axis.low, axis.period)
            elif self.method != "spectral" and isinstance(axis, UnboundedAxis):
                points[:, i] = axis.to_internal(points[:, i])
        return points

    def _evaluate(self, points: NDArray) -> NDArray:
        points = self._wrap(points)
        if self.method == "spectral":
            return self._evaluate_spectral(points)
        if self.grid.ndims > 1:
            return np.asarray(self._inter(points))
        return np.asarray(self._inter(points[:, 0]))

    def _evaluate_spectral(self, points: NDArray, chunk_elements: int = 4_000_000) -> NDArray:
        f = self._data
        M = points.shape[0]
        rest = int(np.prod(f.shape[1:])) if f.ndim > 1 else 1
        chunk = max(1, chunk_elements // max(rest, 1))
        out = np.empty(M, dtype=np.result_type(f, float))
        for start in range(0, M, chunk):
            pts = points[start:start + chunk]
            A = [interpolation_matrix(ax, pts[:, i]) for i, ax in enumerate(self._axes)]
            T = np.tensordot(A[0], f, axes=([1], [0]))  # (m, n1, ..., nd-1)
            for Ai in A[1:]:
                T = np.einsum("mj,mj...->m...", Ai, T)
            out[start:start + chunk] = T
        return out

    def _on_grid(self, target: Grid) -> NDArray:
        if target.ndims != self.grid.ndims:
            raise ValueError(
                f"Target grid has {target.ndims} dimensions, expected {self.grid.ndims}."
            )
        if self.method == "spectral":
            # Separable: apply 1D interpolation matrices axis by axis.
            result = self._data
            for i, (ax, tax) in enumerate(zip(self._axes, target.axes)):
                x = tax.coords
                if ax.periodic:
                    x = ax.low + np.mod(x - ax.low, ax.period)
                A = interpolation_matrix(ax, x)
                result = np.moveaxis(np.tensordot(A, result, axes=([1], [i])), 0, i)
            return result
        points = np.column_stack([c.reshape(-1) for c in target.meshed_coords])
        return self._evaluate(points).reshape(target.shape)


def _extend_radial_axes(grid: Grid, f: NDArray) -> tuple[list[Axis], NDArray]:
    """Extend folded axes to their full range.

    Radial axes through the origin become Chebyshev axes on the diameter,
    colatitude axes become periodic axes on the full great circle. The values
    on the added halves are taken from the antipodal points supplied by the
    grid.
    """
    axes = list(grid.axes)
    data = f
    for i, axis in enumerate(grid.axes):
        if isinstance(axis, RadialAxis):
            antipodal = grid._antipodal_map(i)
            data = np.concatenate([np.flip(antipodal(data), axis=i), data], axis=i)
            axes[i] = ChebyshevAxis(2 * len(axis), -axis.high, axis.high, name=axis.name)
    for i, axis in enumerate(grid.axes):
        if isinstance(axis, ColatitudeAxis):
            # The full great circle [0, 2 pi) is a periodic coordinate; its far
            # half holds the values at (theta, phi + pi) in reversed order.
            # (Composes with the radial extension: (s, 2 pi - theta, phi) is
            # (s, theta, phi + pi) for any signed radius s.)
            antipodal = grid._antipodal_map(i)
            data = np.concatenate([data, np.flip(antipodal(data), axis=i)], axis=i)
            low = axis.great_circle_coords[0]
            axes[i] = EquidistantAxis(2 * len(axis), low, low + 2 * np.pi, periodic=True, name=axis.name)
    return axes, data


def _compactify(axis: UnboundedAxis, data: NDArray, i: int) -> tuple[NDArray, NDArray]:
    """Nodes and data of an unbounded axis in the variable xi, with the ends at infinity added."""
    def value_at(x: float) -> NDArray:
        A = interpolation_matrix(axis, np.array([x]))
        return np.moveaxis(np.tensordot(A, data, axes=([1], [i])), 0, i)

    xi, parts = [axis.coords_internal, [1.0]], [data, value_at(np.inf)]
    if np.isinf(axis.low):
        xi.insert(0, [-1.0])
        parts.insert(0, value_at(-np.inf))
    return np.concatenate(xi), np.concatenate(parts, axis=i)


def _parse_locations(locations, ndims: int) -> tuple[NDArray, bool]:
    """Normalize *locations* to an array of shape ``(M, ndims)``."""
    if isinstance(locations, numbers.Number):
        if ndims != 1:
            raise ValueError(f"A single number is only a valid location on 1D grids, got {ndims}D.")
        return np.array([[float(locations)]]), True
    is_tuple = isinstance(locations, tuple)
    if not isinstance(locations, (np.ndarray, list, tuple)):
        locations = list(locations)  # zip, generators, ...
    arr = np.asarray(locations, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1), True
    if arr.ndim == 1:
        if ndims == 1:
            if is_tuple and len(arr) == 1:
                return arr.reshape(1, 1), True
            return arr.reshape(-1, 1), False
        if len(arr) != ndims:
            raise ValueError(f"Point {tuple(arr)} has {len(arr)} coordinates, expected {ndims}.")
        return arr.reshape(1, ndims), True
    if arr.ndim == 2 and arr.shape[1] == ndims:
        return arr, False
    raise ValueError(
        f"Cannot interpret locations of shape {arr.shape} for a {ndims}D grid. "
        f"Expected an array of shape (M, {ndims})."
    )


# ---------------------------------------------------------------------------
# 1D interpolation matrices
# ---------------------------------------------------------------------------

def interpolation_matrix(axis: Axis, x: NDArray, stencil: int = 6) -> NDArray:
    """Matrix ``A`` such that ``A @ f`` interpolates *f* (on *axis*) at points *x*.

    Parameters
    ----------
    axis : Axis
        The axis.
    x : NDArray
        1D array of target coordinates.  For periodic axes the values must
        already be wrapped into ``[low, high)``.  On unbounded axes any
        point of the domain is allowed, including infinity (the limit of
        the interpolant).
    stencil : int
        Number of points used for local polynomial interpolation on
        non-spectral axes (default 6, i.e. degree 5).

    Returns
    -------
    NDArray
        Dense array of shape ``(len(x), len(axis))``.
    """
    x = np.asarray(x, dtype=float)
    if axis.periodic:
        return _trig_matrix(axis, x)
    if isinstance(axis, UnboundedAxis):
        # Polynomial interpolation in xi (a rational function of x)
        return _barycentric(axis._internal_differences(x), axis.barycentric_weights)
    _check_range(axis, x)
    if isinstance(axis, ChebyshevAxis):
        return _barycentric_matrix(axis.coords, x)
    if isinstance(axis, LogAxis):
        return _lagrange_matrix(np.log(axis.coords), np.log(np.maximum(x, axis.low)), stencil)
    return _lagrange_matrix(axis.coords, x, stencil)


def _check_range(axis: Axis, x: NDArray) -> None:
    lo, hi = axis.coords[0], axis.coords[-1]
    tol = 1e-12 * max(1.0, abs(lo), abs(hi))
    if np.any(x < lo - tol) or np.any(x > hi + tol):
        raise ValueError(
            f"Interpolation point(s) outside the axis range [{lo}, {hi}] (no extrapolation)."
        )


def _barycentric_matrix(nodes: NDArray, x: NDArray) -> NDArray:
    n = len(nodes)
    w = (-1.0) ** np.arange(n)
    w[0] *= 0.5
    w[-1] *= 0.5
    return _barycentric(x[:, None] - nodes[None, :], w)


def _barycentric(d: NDArray, w: NDArray) -> NDArray:
    """Barycentric interpolation matrix from the differences ``d[m, j] = x_m - node_j``."""
    exact = d == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        W = w[None, :] / d
        W = W / W.sum(axis=1, keepdims=True)
    hit = exact.any(axis=1)
    W[hit] = exact[hit].astype(float)
    return W


def _trig_matrix(axis: Axis, x: NDArray) -> NDArray:
    n = len(axis)
    t = 2 * np.pi * (x - axis.low) / axis.period
    tj = 2 * np.pi * np.arange(n) / n
    theta = t[:, None] - tj[None, :]
    half = theta / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        if n % 2:
            S = np.sin(n * half) / (n * np.sin(half))
        else:
            S = np.sin(n * half) / (n * np.tan(half))
    hit = np.isclose(np.mod(theta + np.pi, 2 * np.pi) - np.pi, 0.0, atol=1e-14)
    S[hit] = 1.0
    S[~np.isfinite(S)] = 0.0
    return S


def _lagrange_matrix(nodes: NDArray, x: NDArray, stencil: int) -> NDArray:
    n = len(nodes)
    p = min(stencil, n)
    start = np.clip(np.searchsorted(nodes, x) - p // 2, 0, n - p)
    idx = start[:, None] + np.arange(p)[None, :]          # (M, p)
    s = nodes[idx]                                          # (M, p)
    L = np.ones((len(x), p))
    for k in range(p):
        for m in range(p):
            if m != k:
                L[:, k] *= (x - s[:, m]) / (s[:, k] - s[:, m])
    A = np.zeros((len(x), n))
    np.put_along_axis(A, idx, L, axis=1)
    return A

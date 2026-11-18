r"""Spectral coefficients, resolution diagnostics and filtering.

On a spectral axis a meshed function is, besides its values at the grid
points, a sum of basis functions: complex exponentials on periodic axes and
Chebyshev polynomials on Chebyshev axes and on the axes for unbounded
domains. The coefficients of this expansion tell how well the function is
represented on the grid. For a smooth function they decay rapidly; the
function is resolved as soon as the coefficients of the highest modes are
negligible. Coefficients that are still large at the end of the spectrum
mean that the grid is too coarse.

This module provides

* the transforms between values and coefficients (:func:`coefficients`,
  :func:`from_coefficients`) and a change of resolution in coefficient
  space (:func:`resample`),
* a resolution check that needs nothing but the values on the grid
  (:func:`resolution`, :func:`is_resolved`), and
* filters that act on the coefficients (:func:`spectral_filter`).

All of them are available as methods of :class:`~numgrids.grids.Grid`.

.. rubric:: Expansions

==========================================  ==============================================================
axis                                        expansion
==========================================  ==============================================================
periodic                                    :math:`f(x) = \sum_k c_k\, e^{i \kappa_k x}`,
                                            :math:`\kappa_k = 2\pi k / P`
Chebyshev                                   :math:`f(x) = \sum_k a_k\, T_k(\xi)`,
                                            :math:`\xi = (2x - a - b)/(b - a)`
semi-infinite, infinite                     :math:`f(x) = \sum_k a_k\, T_k(\xi(x))` with the
                                            computational variable :math:`\xi` of the axis
radial (through the origin)                 Chebyshev series in :math:`\xi = s/R` on the diameter
                                            :math:`-R \le s \le R`
colatitude (through the poles)              Fourier series in :math:`\theta` on the great circle
                                            :math:`0 \le \theta < 2\pi`
equidistant (not periodic), logarithmic     none: finite differences have no spectral expansion
==========================================  ==============================================================

Fourier coefficients are stored in the order of :func:`numpy.fft.fftfreq`
(mode numbers :math:`0, 1, \ldots, -2, -1`) and are normalized such that
:math:`|c_k|` is the amplitude of the mode: :math:`c_0` is the mean value,
and a real oscillation :math:`A\cos(\kappa_k x + \varphi)` contributes
:math:`\tfrac{A}{2} e^{\pm i\varphi}` to :math:`c_{\pm k}`. The phases refer
to the coordinate :math:`x` itself, not to the first grid point. For an
even number of points :math:`n`, the entry at index :math:`n/2` belongs to
the Nyquist mode, which the grid cannot tell from its mirror image; it is
the amplitude of :math:`\cos\bigl(\kappa_{n/2}(x - x_0)\bigr)`, which is
:math:`\pm 1` at the grid points.

Along a radial or colatitude axis with :math:`n` points there are
:math:`2n` coefficients: the field is continued to the far side of the
origin or the poles (as a scalar), and the coefficients are those of the
function on the whole diameter or great circle.

See :func:`mode_numbers` for the mode number that belongs to each
coefficient.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.fft
from numpy.typing import NDArray

from numgrids.axes import (
    Axis,
    ChebyshevAxis,
    ColatitudeAxis,
    EquidistantAxis,
    InfiniteAxis,
    RadialAxis,
    SemiInfiniteAxis,
)
from numgrids.grids import Grid

AxisSpec = int | str | Sequence[int | str] | None

_FOLDED = (RadialAxis, ColatitudeAxis)

#: Relative level, per mode, below which coefficients cannot be told from round-off.
_ROUNDOFF = 10 * np.finfo(float).eps


# ---------------------------------------------------------------------------
# Expansions along one axis
# ---------------------------------------------------------------------------

def _index(ndim: int, axis: int, key) -> tuple:
    """Index tuple that applies *key* along *axis* and takes everything else."""
    return (slice(None),) * axis + (key,) + (slice(None),) * (ndim - axis - 1)


def _along(values: NDArray, ndim: int, axis: int) -> NDArray:
    """Reshape a 1D array such that it broadcasts along *axis* of an array with *ndim* dimensions."""
    shape = [1] * ndim
    shape[axis] = -1
    return values.reshape(shape)


def _apply(matrix: NDArray, a: NDArray, axis: int) -> NDArray:
    return np.moveaxis(np.tensordot(matrix, a, axes=([1], [axis])), 0, axis)


class _Basis:
    """Expansion of the values on the nodes of one axis in basis functions.

    For axes that are folded across the origin or a pole, the nodes are
    those of the whole diameter or great circle.
    """

    kind = ""

    def __init__(self, size: int) -> None:
        self.size = size

    @property
    def num_modes(self) -> int:
        """Number of distinct (non-negative) mode numbers."""
        raise NotImplementedError

    def mode_numbers(self) -> NDArray:
        """The (signed) mode number of each coefficient."""
        raise NotImplementedError

    def forward(self, values: NDArray, axis: int) -> NDArray:
        raise NotImplementedError

    def backward(self, coeffs: NDArray, axis: int) -> NDArray:
        raise NotImplementedError

    def amplitudes(self, coeffs: NDArray, axis: int) -> NDArray:
        """Magnitude per mode number 0, ..., num_modes - 1 along *axis*."""
        raise NotImplementedError

    def resize(self, coeffs: NDArray, target: _Basis, axis: int) -> NDArray:
        """Coefficients of the same function in a basis of another size (zero-padded or truncated)."""
        raise NotImplementedError

    def size_for(self, num_modes: int) -> int:
        """Number of nodes of the smallest basis of this kind with *num_modes* mode numbers."""
        raise NotImplementedError


class _ChebyshevBasis(_Basis):
    """Chebyshev series on nodes given by the matrix ``T_k(xi_j)`` and its inverse."""

    kind = "chebyshev"

    def __init__(self, size: int, to_values: NDArray | None = None, to_coeffs: NDArray | None = None) -> None:
        super().__init__(size)
        self._to_values = to_values
        self._to_coeffs = to_coeffs

    @property
    def num_modes(self) -> int:
        return self.size

    def mode_numbers(self) -> NDArray:
        return np.arange(self.size)

    def forward(self, values: NDArray, axis: int) -> NDArray:
        return _apply(self._to_coeffs, values, axis)

    def backward(self, coeffs: NDArray, axis: int) -> NDArray:
        return _apply(self._to_values, coeffs, axis)

    def amplitudes(self, coeffs: NDArray, axis: int) -> NDArray:
        return np.abs(coeffs)

    def resize(self, coeffs: NDArray, target: _Basis, axis: int) -> NDArray:
        n, m = self.size, target.size
        if m <= n:
            return coeffs[_index(coeffs.ndim, axis, slice(0, m))]
        pad = [(0, 0)] * coeffs.ndim
        pad[axis] = (0, m - n)
        return np.pad(coeffs, pad)

    def size_for(self, num_modes: int) -> int:
        return max(2, num_modes)


def _dct1(a: NDArray, axis: int) -> NDArray:
    # Real and imaginary parts separately: this works with every SciPy version.
    if np.iscomplexobj(a):
        return scipy.fft.dct(a.real, type=1, axis=axis) + 1j * scipy.fft.dct(a.imag, type=1, axis=axis)
    return scipy.fft.dct(a, type=1, axis=axis)


class _LobattoBasis(_ChebyshevBasis):
    r"""Chebyshev series on the nodes :math:`\xi_j = -\cos(j\pi/N)`, with the fast cosine transform."""

    def __init__(self, size: int) -> None:
        super().__init__(size)
        # T_k(xi_j) = (-1)^k cos(j k pi / N): the discrete cosine transform
        # of type 1, up to the signs and half weights for k = 0 and k = N.
        self._signs = (-1.0) ** np.arange(size)
        self._half = np.ones(size)
        self._half[[0, -1]] = 0.5

    def forward(self, values: NDArray, axis: int) -> NDArray:
        factors = self._signs * self._half / (self.size - 1)
        return _dct1(values, axis) * _along(factors, values.ndim, axis)

    def backward(self, coeffs: NDArray, axis: int) -> NDArray:
        factors = self._signs * np.where(self._half < 1, 1.0, 0.5)
        return _dct1(coeffs * _along(factors, coeffs.ndim, axis), axis)


def _gauss_basis(axis: InfiniteAxis) -> _ChebyshevBasis:
    r"""Chebyshev series on the Gauss points :math:`\xi_j = \sin\varphi_j` of an infinite axis."""
    n = len(axis)
    k = np.arange(n)
    V = np.cos(np.outer(np.pi / 2 - axis._angles(), k))            # T_k(xi_j)
    # Discrete orthogonality (Gauss--Chebyshev quadrature) inverts V without solving a system.
    norm = np.full(n, 2.0 / n)
    norm[0] = 1.0 / n
    return _ChebyshevBasis(n, V, norm[:, None] * V.T)


def _radau_basis(axis: SemiInfiniteAxis) -> _ChebyshevBasis:
    r"""Chebyshev series on the Gauss--Radau points :math:`\xi_j = -\cos\theta_j` of a semi-infinite axis."""
    n = len(axis)
    k = np.arange(n)
    V = (-1.0) ** k * np.cos(np.outer(axis._angles(), k))          # T_k(xi_j)
    # Gauss--Radau--Chebyshev quadrature: the weight of the node xi = -1 is halved.
    weights = np.full(n, 2.0)
    weights[0] = 1.0
    norm = np.full(n, 2.0 / (2 * n - 1))
    norm[0] = 1.0 / (2 * n - 1)
    return _ChebyshevBasis(n, V, norm[:, None] * V.T * weights[None, :])


class _FourierBasis(_Basis):
    r"""Fourier series :math:`\sum_k c_k e^{i\kappa_k x}` on equidistant nodes :math:`x_j = x_0 + jP/n`."""

    kind = "fourier"

    def __init__(self, size: int, low: float, period: float) -> None:
        super().__init__(size)
        self.low = low
        self.period = period

    @property
    def num_modes(self) -> int:
        return self.size // 2 + 1

    def mode_numbers(self) -> NDArray:
        return np.rint(np.fft.fftfreq(self.size, 1.0 / self.size)).astype(int)

    def _phase(self, k) -> NDArray:
        r""":math:`e^{i\kappa_k x_0}`, the value of mode *k* at the first node."""
        return np.exp(2j * np.pi * np.asarray(k) * (self.low / self.period))

    def _phases(self) -> NDArray | None:
        if self.low == 0:
            return None
        phases = self._phase(self.mode_numbers())
        if self.size % 2 == 0:
            phases[self.size // 2] = 1.0            # the Nyquist mode is cos(kappa (x - x_0))
        return phases

    def forward(self, values: NDArray, axis: int) -> NDArray:
        c = np.fft.fft(values, axis=axis) / self.size
        phases = self._phases()
        return c if phases is None else c * _along(np.conj(phases), c.ndim, axis)

    def backward(self, coeffs: NDArray, axis: int) -> NDArray:
        phases = self._phases()
        if phases is not None:
            coeffs = coeffs * _along(phases, coeffs.ndim, axis)
        return np.fft.ifft(coeffs, axis=axis) * self.size

    def amplitudes(self, coeffs: NDArray, axis: int) -> NDArray:
        # A real mode A cos(kx + phi) has |c_k| = |c_-k| = A / 2: add both.
        n, ndim = self.size, coeffs.ndim
        mag = np.abs(coeffs)
        amp = mag[_index(ndim, axis, slice(0, self.num_modes))].copy()
        paired = (n - 1) // 2
        if paired:
            amp[_index(ndim, axis, slice(1, paired + 1))] += mag[_index(ndim, axis, slice(n - 1, n - paired - 1, -1))]
        return amp

    def _to_centered(self, coeffs: NDArray, axis: int) -> NDArray:
        """Coefficients of the modes -K, ..., K with K = size // 2."""
        n, ndim = self.size, coeffs.ndim
        shifted = np.fft.fftshift(coeffs, axes=axis)
        if n % 2:
            return shifted
        # cos(kappa (x - x_0)) is half the mode +K and half the mode -K.
        nyquist = shifted[_index(ndim, axis, slice(0, 1))] / 2
        phase = self._phase(n // 2)
        return np.concatenate(
            [nyquist * phase, shifted[_index(ndim, axis, slice(1, None))], nyquist * np.conj(phase)], axis=axis
        )

    def _from_centered(self, centered: NDArray, axis: int) -> NDArray:
        n, ndim = self.size, centered.ndim
        if n % 2:
            return np.fft.ifftshift(centered, axes=axis)
        # On the nodes the modes +K and -K are both +-1 times their value at x_0.
        phase = self._phase(n // 2)
        nyquist = (
            centered[_index(ndim, axis, slice(-1, None))] * phase
            + centered[_index(ndim, axis, slice(0, 1))] * np.conj(phase)
        )
        shifted = np.concatenate([nyquist, centered[_index(ndim, axis, slice(1, -1))]], axis=axis)
        return np.fft.ifftshift(shifted, axes=axis)

    def resize(self, coeffs: NDArray, target: _Basis, axis: int) -> NDArray:
        centered = self._to_centered(coeffs, axis)
        K, K_new = self.size // 2, target.size // 2
        if K_new <= K:
            centered = centered[_index(centered.ndim, axis, slice(K - K_new, K + K_new + 1))]
        else:
            pad = [(0, 0)] * centered.ndim
            pad[axis] = (K_new - K, K_new - K)
            centered = np.pad(centered, pad)
        return target._from_centered(centered, axis)

    def size_for(self, num_modes: int) -> int:
        # An even number of points, as the azimuthal axes of curvilinear grids require.
        return max(2, 2 * (num_modes - 1))


def _basis(axis: Axis) -> _Basis | None:
    """The expansion that belongs to an axis, or None for finite-difference axes."""
    n = len(axis)
    if axis.periodic:
        return _FourierBasis(n, axis.low, axis.period)
    if isinstance(axis, ChebyshevAxis):
        return _LobattoBasis(n)
    if isinstance(axis, RadialAxis):
        return _LobattoBasis(2 * n)
    if isinstance(axis, ColatitudeAxis):
        return _FourierBasis(2 * n, np.pi / (2 * n), 2 * np.pi)
    if isinstance(axis, SemiInfiniteAxis):
        return _radau_basis(axis)
    if isinstance(axis, InfiniteAxis):
        return _gauss_basis(axis)
    return None


def _label(grid: Grid, i: int) -> str:
    """Name (or index) and type of an axis, for error messages."""
    axis = grid.axes[i]
    name = repr(axis.name) if axis.name is not None else str(i)
    kind = type(axis).__name__ + (", not periodic" if isinstance(axis, EquidistantAxis) and not axis.periodic else "")
    return f"{name} ({kind})"


def _require_basis(grid: Grid, i: int) -> _Basis:
    basis = _basis(grid.axes[i])
    if basis is None:
        raise ValueError(
            f"Axis {_label(grid, i)} is discretized with finite differences and has no spectral expansion. "
            "Spectral coefficients exist on periodic, Chebyshev, radial, colatitude and unbounded axes; "
            "select those with the `axis` argument."
        )
    return basis


def _resolve_axes(grid: Grid, axis: AxisSpec) -> list[int]:
    if axis is None:
        return list(range(grid.ndims))
    if isinstance(axis, (int, np.integer, str)):
        return [grid.axis_index(axis)]
    indices = [grid.axis_index(a) for a in axis]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Axes must not be repeated, got {axis!r}.")
    return sorted(indices)


def _spectral_axes(grid: Grid, axis: AxisSpec) -> dict[int, _Basis]:
    """The bases of the selected axes; with ``axis=None`` those of all spectral axes."""
    if axis is None:
        bases = {i: _basis(ax) for i, ax in enumerate(grid.axes)}
        bases = {i: basis for i, basis in bases.items() if basis is not None}
        if not bases:
            raise ValueError(
                "The grid has no spectral axis (periodic, Chebyshev, radial, colatitude or unbounded): "
                "finite differences have no spectral expansion."
            )
        return bases
    return {i: _require_basis(grid, i) for i in _resolve_axes(grid, axis)}


def _extend(grid: Grid, f: NDArray, axes) -> NDArray:
    """Continue a field across the origin and the poles along the folded axes among *axes*.

    Radial axes become the whole diameter (ascending from -R to R),
    colatitude axes the whole great circle; the added halves hold the
    values at the antipodal points. Radial axes come first: their
    antipodal map needs the other axes in their original layout.
    """
    data = f
    for i in axes:
        if isinstance(grid.axes[i], RadialAxis):
            data = np.concatenate([np.flip(grid._antipodal_map(i)(data), axis=i), data], axis=i)
    for i in axes:
        if isinstance(grid.axes[i], ColatitudeAxis):
            data = np.concatenate([data, np.flip(grid._antipodal_map(i)(data), axis=i)], axis=i)
    return data


def _restrict(grid: Grid, data: NDArray, axes) -> NDArray:
    """Inverse of :func:`_extend`: keep the half of each folded axis that belongs to the grid."""
    for i in axes:
        n = len(grid.axes[i])
        if isinstance(grid.axes[i], RadialAxis):
            data = data[_index(data.ndim, i, slice(n, None))]
        elif isinstance(grid.axes[i], ColatitudeAxis):
            data = data[_index(data.ndim, i, slice(0, n))]
    return data


def _as_field(grid: Grid, f) -> NDArray:
    f = grid._check_field(f)
    if not np.issubdtype(f.dtype, np.inexact):
        f = f.astype(float)
    return f


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def mode_numbers(grid: Grid, axis: int | str = 0) -> NDArray:
    r"""The mode number of each coefficient along an axis.

    Parameters
    ----------
    grid : Grid
        The grid.
    axis : int or str
        Axis index or name.

    Returns
    -------
    NDArray
        Integer array with one entry per coefficient: the degree :math:`k`
        of :math:`T_k` for Chebyshev series, and the signed mode number
        :math:`k` of :math:`e^{2\pi i k x / P}` in the order of
        :func:`numpy.fft.fftfreq` for Fourier series (multiply by
        :math:`2\pi/P` to get wavenumbers).

    Examples
    --------
    >>> from numgrids import Grid, chebyshev, periodic
    >>> from numgrids.spectral import mode_numbers
    >>> grid = Grid(x=chebyshev(5, 0, 1), y=periodic(6))
    >>> mode_numbers(grid, "x")
    array([0, 1, 2, 3, 4])
    >>> mode_numbers(grid, "y")
    array([ 0,  1,  2, -3, -2, -1])
    """
    return _require_basis(grid, grid.axis_index(axis)).mode_numbers()


def coefficients(grid: Grid, f: NDArray, axis: AxisSpec = None) -> NDArray:
    r"""Expansion coefficients of a meshed function.

    Transforms along one axis, several axes or (by default) all axes; in
    the last case the result holds the coefficients of the tensor-product
    expansion. See the :mod:`module documentation <numgrids.spectral>` for
    the expansion that belongs to each axis type and for the ordering and
    normalization of the coefficients.

    Parameters
    ----------
    grid : Grid
        The grid.
    f : NDArray
        Meshed function of shape ``grid.shape`` (real or complex).
    axis : int, str, sequence of those, or None
        The axis or axes to transform (default: all). The other axes keep
        the values at their grid points.

    Returns
    -------
    NDArray
        The coefficients. The array has the shape of the grid, except
        along radial and colatitude axes, which have twice as many
        coefficients as points. It is complex if *f* is complex or a
        Fourier series is involved.

    Raises
    ------
    ValueError
        If one of the axes is discretized with finite differences
        (equidistant and not periodic, or logarithmic).

    See Also
    --------
    from_coefficients : the inverse transform.

    Examples
    --------
    Chebyshev coefficients: :math:`2x^2 - 1 = T_2(x)` on :math:`[-1, 1]`.

    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev, periodic
    >>> grid = Grid(x=chebyshev(6, -1, 1))
    >>> x = grid.coords
    >>> grid.coefficients(2 * x ** 2 - 1 + 0.5 * x).round(12) + 0
    array([0. , 0.5, 1. , 0. , 0. , 0. ])

    Fourier coefficients: :math:`3\cos 2x` has :math:`c_{\pm 2} = 3/2`.

    >>> grid = Grid(x=periodic(8))
    >>> c = grid.coefficients(1 + 3 * np.cos(2 * grid.coords))
    >>> c.real.round(12) + 0
    array([1. , 0. , 1.5, 0. , 0. , 0. , 1.5, 0. ])
    """
    f = _as_field(grid, f)
    bases = {i: _require_basis(grid, i) for i in _resolve_axes(grid, axis)}
    c = _extend(grid, f, bases)
    for i, basis in bases.items():
        c = basis.forward(c, i)
    return c


def from_coefficients(grid: Grid, c: NDArray, axis: AxisSpec = None) -> NDArray:
    """Values on the grid of a function given by its expansion coefficients.

    The inverse of :func:`coefficients`: ``grid.from_coefficients(
    grid.coefficients(f, axis), axis)`` returns *f* up to round-off.

    Parameters
    ----------
    grid : Grid
        The grid.
    c : NDArray
        The coefficients, as returned by :func:`coefficients` for the same
        *axis* argument.
    axis : int, str, sequence of those, or None
        The axis or axes along which *c* holds coefficients (default: all).

    Returns
    -------
    NDArray
        Meshed function of shape ``grid.shape``. Along periodic and
        colatitude axes the result is complex; take the real part if the
        coefficients are those of a real function
        (:math:`c_{-k} = \\bar c_k`).

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev, periodic
    >>> grid = Grid(x=chebyshev(12, 0, 1), y=periodic(16))
    >>> X, Y = grid.meshed_coords
    >>> f = np.exp(X) * np.sin(Y)
    >>> c = grid.coefficients(f)
    >>> bool(np.allclose(grid.from_coefficients(c), f, atol=1e-13))
    True

    Differentiation with respect to *y* in coefficient space:

    >>> from numgrids.spectral import mode_numbers
    >>> dc = 1j * mode_numbers(grid, "y") * c
    >>> bool(np.allclose(grid.from_coefficients(dc).real, np.exp(X) * np.cos(Y), atol=1e-12))
    True
    """
    bases = {i: _require_basis(grid, i) for i in _resolve_axes(grid, axis)}
    c = np.asarray(c)
    if not np.issubdtype(c.dtype, np.inexact):
        c = c.astype(float)
    expected = tuple(bases[i].size if i in bases else n for i, n in enumerate(grid.shape))
    if c.shape != expected:
        raise ValueError(
            f"Coefficient array has shape {c.shape}, expected {expected} "
            "(radial and colatitude axes have twice as many coefficients as points)."
        )
    for i, basis in bases.items():
        c = basis.backward(c, i)
    return _restrict(grid, c, bases)


def _target_grid(grid: Grid, target) -> Grid:
    if isinstance(target, Grid):
        if target.ndims != grid.ndims:
            raise ValueError(f"Target grid has {target.ndims} dimensions, expected {grid.ndims}.")
        for i, (old, new) in enumerate(zip(grid.axes, target.axes)):
            if old.resized(len(new)).renamed(new.name) != new:
                raise ValueError(
                    f"Axis {i} of the target grid differs in more than the number of points "
                    f"({old!r} vs. {new!r}). Use `grid.interpolate` or an Interpolator to change the domain."
                )
        return target
    shape = (target,) if isinstance(target, (int, np.integer)) else tuple(target)
    if len(shape) != grid.ndims:
        raise ValueError(f"Expected a grid or {grid.ndims} numbers of points, got {target!r}.")
    return grid._new_like([ax if len(ax) == n else ax.resized(n) for ax, n in zip(grid.axes, shape)])


def resample(grid: Grid, f: NDArray, target: Grid | int | Sequence[int]) -> NDArray:
    """Transfer a meshed function to a grid with other numbers of points.

    The change of resolution is done in coefficient space: for more points
    the coefficients are padded with zeros, which is the same as spectral
    interpolation; for fewer points the series is truncated. A function
    that is resolved on both grids is transferred without loss.

    Parameters
    ----------
    grid : Grid
        The grid on which *f* is given.
    f : NDArray
        Meshed function of shape ``grid.shape``.
    target : Grid, int or sequence of int
        The new numbers of points per axis, or a grid that differs from
        *grid* in the numbers of points only.

    Returns
    -------
    NDArray
        The function on the new grid.

    Raises
    ------
    ValueError
        If the number of points of a finite-difference axis changes; use
        :meth:`Grid.interpolate <numgrids.grids.Grid.interpolate>` for those.

    See Also
    --------
    Resolution.suggested_grid : a grid on which the function is resolved.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev
    >>> coarse, fine = Grid(x=chebyshev(16, 0, 1)), Grid(x=chebyshev(40, 0, 1))
    >>> f = coarse.resample(np.exp(coarse.coords), fine)
    >>> bool(np.allclose(f, np.exp(fine.coords), atol=1e-13))
    True
    >>> coarse.resample(np.exp(coarse.coords), 8).shape
    (8,)
    """
    f = _as_field(grid, f)
    new_grid = _target_grid(grid, target)
    changed = [i for i in range(grid.ndims) if len(grid.axes[i]) != len(new_grid.axes[i])]
    for i in changed:
        if _basis(grid.axes[i]) is None:
            raise ValueError(
                f"Axis {_label(grid, i)} is discretized with finite differences and cannot be resampled "
                "in coefficient space. Use `grid.interpolate(f, new_grid)` instead."
            )
    data = _extend(grid, f, changed)
    for i in changed:
        source, destination = _basis(grid.axes[i]), _basis(new_grid.axes[i])
        data = destination.backward(source.resize(source.forward(data, i), destination, i), i)
    data = _restrict(new_grid, data, changed)
    if np.iscomplexobj(data) and not np.iscomplexobj(f):
        data = data.real
    return data.copy() if data is f else data


# ---------------------------------------------------------------------------
# Resolution diagnostics
# ---------------------------------------------------------------------------

def _tail_width(num_modes: int) -> int:
    """Number of highest modes that must be negligible: a tenth, at least two.

    Two, because every second coefficient of an even or odd function
    vanishes, whatever the resolution.
    """
    return max(2, -(-num_modes // 10))


def _modes_needed(significant: int) -> int:
    """Smallest number of modes for which *significant* leading modes leave a negligible tail."""
    m = significant + 2
    while m - _tail_width(m) < significant:
        m += 1
    return m


def _fit_decay(magnitudes: NDArray, modes: NDArray) -> tuple[str, float, float]:
    """Fit ``exp(-c k)`` and ``k**(-p)`` to the magnitudes of some modes; return the better model, c and p.

    The rate *c* is that of the upper half of the modes (the decay often
    sets in gradually), the order *p* that of all of them. A geometric
    decay is the normal case. The decay only counts as algebraic if the
    power law fits clearly better and the geometric rate drops markedly
    from the lower to the upper half, as it does for a power law.
    """
    if len(modes) < 4:
        return "", float("nan"), float("nan")
    k = modes.astype(float)
    y = np.log(magnitudes[modes])

    def fit(t: NDArray, y: NDArray) -> tuple[float, float]:
        slope, intercept = np.polyfit(t, y, 1)
        return -float(slope), float(np.sum((y - slope * t - intercept) ** 2))

    order, misfit_algebraic = fit(np.log(k), y)
    _, misfit_geometric = fit(k, y)
    half = max(4, len(modes) - len(modes) // 2)
    rate = fit(k[-half:], y[-half:])[0]
    algebraic = (
        len(modes) >= 8
        and misfit_algebraic < 0.25 * misfit_geometric          # squared residuals: half the rms misfit
        and rate < 0.6 * fit(k[:-half], y[:-half])[0]
    )
    return "algebraic" if algebraic else "geometric", rate, order


def _assess(spectrum: NDArray, tol: float) -> dict:
    """Judge the resolution from the magnitudes of the modes 0, ..., M - 1 (see :func:`resolution`)."""
    M = len(spectrum)
    scale = float(spectrum.max())
    if scale == 0:
        return {"resolved": True, "modes": 2, "tail": 0.0, "decay": "", "rate": float("nan"), "order": float("nan")}
    relative = spectrum / scale
    envelope = np.maximum.accumulate(relative[::-1])[::-1]       # largest magnitude from each mode on
    remainder = np.cumsum(relative[::-1])[::-1]                  # sum of the magnitudes from each mode on
    w = _tail_width(M)
    tail = float(remainder[M - w])
    resolved = tail <= tol
    result = {"resolved": resolved, "tail": tail}

    # The decay is fitted to the modes that are larger than all higher ones (zeros
    # and oscillations in between say nothing about it), from the point where the
    # spectrum has left its plateau of large coefficients.
    start = max(1, int(np.count_nonzero(envelope >= 0.5)))
    peaks = np.flatnonzero((relative == envelope) & (relative > 0))
    if resolved:
        significant = int(np.count_nonzero(remainder > tol))
        decay, rate, order = _fit_decay(relative, peaks[(peaks >= start) & (peaks < significant)])
        return {**result, "modes": _modes_needed(significant), "decay": decay, "rate": rate, "order": order}

    # The largest of the highest modes is the last one to trust: the modes behind
    # it may vanish by symmetry.
    last = M - w + int(np.argmax(relative[M - w:]))
    decay, rate, order = _fit_decay(relative, peaks[(peaks >= start) & (peaks <= last)])
    middle = M // 2
    flat = last - middle < 2 or envelope[middle] < 2.5 * relative[last]
    if flat or not decay or (decay == "geometric" and rate <= 0) or (decay == "algebraic" and order <= 1):
        return {**result, "modes": None, "decay": "none", "rate": rate, "order": order}
    # Extrapolate: the mode from which on the sum of the magnitudes is below the tolerance.
    with np.errstate(over="ignore"):
        if decay == "geometric":
            reach = last + np.log(relative[last] / (-np.expm1(-rate) * tol)) / rate
        else:
            reach = (relative[last] * last ** order / ((order - 1) * tol)) ** (1 / (order - 1))
    reach = int(min(np.ceil(reach), 10 ** 9))
    # An extrapolation from few modes is uncertain: suggest a clear step at least.
    modes = max(_modes_needed(reach + 1), round(1.25 * M) + 5)
    return {**result, "modes": modes, "decay": decay, "rate": rate, "order": order}


@dataclass(eq=False)
class AxisResolution:
    """Resolution of a function along one axis; part of a :class:`Resolution`.

    Attributes
    ----------
    axis : int
        Index of the axis.
    name : str or None
        Name of the axis.
    kind : str
        ``"fourier"`` or ``"chebyshev"``; empty for finite-difference
        axes, which cannot be assessed.
    num_points : int
        Number of grid points of the axis.
    resolved : bool or None
        Whether the highest modes are below the tolerance; ``None`` if the
        axis cannot be assessed.
    needed : int or None
        Estimated number of points needed for the tolerance, including
        the modes that have to be negligible at the end. For a resolved
        function it is at most :attr:`num_points`; otherwise it is
        extrapolated from the decay of the spectrum, or ``None`` if the
        spectrum does not decay (or the axis cannot be assessed).
    tail : float
        Sum of the magnitudes of the highest modes (a tenth of all modes,
        at least two), relative to the largest magnitude of all modes.
    decay : str
        ``"geometric"`` (the normal case) or ``"algebraic"`` (if a power
        law describes the decay of the spectrum clearly better: the
        function is not smooth); ``"none"`` if an unresolved spectrum
        does not decay; empty if there are too few modes to tell.
    rate : float
        Fitted rate *c* of a geometric decay ``exp(-c k)`` with the mode
        number *k* (from the upper half of the decaying modes).
    order : float
        Fitted order *p* of an algebraic decay ``k**(-p)``.
    spectrum : NDArray
        Magnitude of each mode number 0, 1, 2, ..., maximized over the
        other axes. On Fourier axes mode *k* collects the coefficients of
        :math:`e^{\\pm i\\kappa_k x}`, so that the magnitude is the
        amplitude of a real oscillation. On radial and colatitude axes
        the modes are those of the whole diameter or great circle.
    tol : float
        The tolerance of the assessment.
    """

    axis: int
    name: str | None
    kind: str
    num_points: int
    resolved: bool | None
    needed: int | None
    tail: float
    decay: str
    rate: float
    order: float
    spectrum: NDArray = field(repr=False)
    tol: float = 1e-8

    def __str__(self) -> str:
        label = f"axis {self.name}" if self.name is not None else f"axis {self.axis}"
        if self.resolved is None:
            return f"{label}: not assessed (finite differences have no spectral expansion)"
        if self.resolved:
            return f"{label}: resolved, {self.needed} of {self.num_points} points needed"
        text = f"{label}: NOT resolved, tail at {self.tail:.0e}"
        if self.needed is None:
            return text + f" and not decaying ({self.num_points} points: too few, or the data is noisy or not smooth)"
        number = str(self.needed) if self.needed < 100_000 else f"{self.needed:.0e}"
        text += f"; about {number} points needed"
        if self.decay == "algebraic":
            text += f" (algebraic decay of order {self.order:.1f}: the function is not smooth)"
        return text


@dataclass(eq=False)
class Resolution:
    """Result of :func:`resolution`: is a meshed function resolved on its grid?

    Printing the object gives a summary with one line per axis.

    Attributes
    ----------
    resolved : bool
        ``True`` if the function is resolved along all axes that can be
        assessed.
    tol : float
        The tolerance of the assessment.
    axes : tuple of AxisResolution
        The details for each axis. ``result[axis]`` returns the entry of
        an axis given by index or name.
    grid : Grid
        The grid.
    """

    resolved: bool
    tol: float
    axes: tuple[AxisResolution, ...]
    grid: Grid = field(repr=False, compare=False)

    def __getitem__(self, axis: int | str) -> AxisResolution:
        return self.axes[self.grid.axis_index(axis)]

    @property
    def needed(self) -> tuple[int | None, ...]:
        """The estimated number of points needed along each axis (``None`` where unknown)."""
        return tuple(a.needed for a in self.axes)

    @property
    def suggested_shape(self) -> tuple[int, ...]:
        """Numbers of points to try next.

        The estimate :attr:`AxisResolution.needed` where there is one, but
        at most four times the present number, because an estimate for an
        unresolved function is an extrapolation; twice the present number
        where the spectrum does not decay; the present number for axes
        that cannot be assessed.
        """
        shape = []
        for a in self.axes:
            if a.resolved is None:
                shape.append(a.num_points)
            elif a.needed is None:
                shape.append(2 * a.num_points)
            else:
                shape.append(min(a.needed, 4 * a.num_points))
        return tuple(shape)

    @property
    def suggested_grid(self) -> Grid:
        """A grid of the same type with :attr:`suggested_shape` points."""
        axes = [ax if len(ax) == n else ax.resized(n) for ax, n in zip(self.grid.axes, self.suggested_shape)]
        return self.grid._new_like(axes)

    def summary(self) -> str:
        """The verdict and one line per axis."""
        verdict = "resolved" if self.resolved else "NOT resolved"
        lines = [f"Resolution at tol={self.tol:g}: {verdict}"]
        lines += [f"  {a}" for a in self.axes]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    __repr__ = __str__

    def plot(self, ax=None, **kwargs):
        """Plot the spectrum of each axis on a logarithmic scale (requires matplotlib).

        The magnitudes are relative to the largest one of each axis; the
        dashed line marks the tolerance. The modes are drawn as dots
        without a connecting line: the coefficients of even and odd
        functions alternate between their true size and round-off.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created if omitted.
        **kwargs
            Passed on to ``ax.semilogy``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from numgrids.plots import require_matplotlib
        plt = require_matplotlib()
        if ax is None:
            _, ax = plt.subplots()
        kwargs.setdefault("marker", ".")
        if "ls" not in kwargs:
            kwargs.setdefault("linestyle", "none")
        for a in self.axes:
            if a.resolved is None or not a.spectrum.max() > 0:
                continue
            label = a.name if a.name is not None else f"axis {a.axis}"
            status = "resolved" if a.resolved else "not resolved"
            # Zeros (e.g. every second coefficient of an even function) cannot be drawn.
            relative = np.where(a.spectrum > 0, a.spectrum / a.spectrum.max(), np.nan)
            ax.semilogy(np.arange(len(relative)), relative, label=f"{label} ({a.num_points} points, {status})",
                        **kwargs)
        ax.axhline(self.tol, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("mode number")
        ax.set_ylabel("coefficient magnitude (relative)")
        ax.legend()
        return ax


def resolution(grid: Grid, f: NDArray, tol: float = 1e-8) -> Resolution:
    r"""Check whether a meshed function is resolved on its grid.

    The check looks at the expansion coefficients of the function along
    each axis (see :func:`coefficients`); it needs neither a finer grid
    nor another evaluation of the function. For each axis, the magnitude
    of each mode is maximized over the other axes and divided by the
    largest one. The function is *resolved* along the axis if the
    magnitudes of the highest modes (a tenth of all, at least two) add up
    to less than *tol*. The sum of the magnitudes of all modes that a
    truncated series lacks bounds its error, so adding points then changes
    the function by no more than about *tol* times its size.

    The number of points needed follows from the same rule. A resolved
    series can be chopped where the sum of the dropped modes falls below
    *tol*; the number of points is the number of modes kept plus the
    negligible ones required at the end. For a function that is not
    resolved, the decay of the spectrum is extrapolated to that point. The
    extrapolation assumes a geometric decay :math:`e^{-ck}`, as for
    smooth functions, unless a power law :math:`k^{-p}` describes the
    spectrum clearly better; this is the mark of a kink or another
    singularity, and the number of points becomes very large. The
    estimate is only a guide: repeat the check on the finer grid. If the
    highest modes do not decay at all, no estimate is possible: the
    number of points is far too small, or the data contains noise or
    jumps.

    Parameters
    ----------
    grid : Grid
        The grid.
    f : NDArray
        Meshed function of shape ``grid.shape`` (real or complex).
    tol : float
        Tolerance for the highest modes, relative to the largest mode
        (default ``1e-8``). It cannot be smaller than the round-off level
        of the coefficients, which is taken as ten times the machine
        epsilon per mode.

    Returns
    -------
    Resolution
        The verdict, with the details for each axis. Axes that are
        discretized with finite differences are not assessed.

    Raises
    ------
    ValueError
        If the grid has no spectral axis at all.

    Notes
    -----
    The rule follows the idea with which Chebfun chops Chebyshev series,
    in a simpler form and for a tolerance chosen by the user. Like every
    check that only sees the values on the grid, it can be fooled: a
    function that oscillates so fast that the grid takes it for a slow
    one (aliasing) passes as resolved.

    Reference: Aurentz & Trefethen, *Chopping a Chebyshev series*, ACM
    Trans. Math. Softw. 43 (2017).

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev, periodic
    >>> grid = Grid(x=chebyshev(40, -1, 1), y=periodic(16))
    >>> X, Y = grid.meshed_coords
    >>> grid.resolution(np.exp(X) * np.sin(Y))
    Resolution at tol=1e-08: resolved
      axis x: resolved, 11 of 40 points needed
      axis y: resolved, 6 of 16 points needed

    A function with a steep front in *x* and short waves in *y* is not
    resolved on this grid. The result suggests a finer one:

    >>> func = lambda x, y: np.tanh(8 * x) * np.exp(np.sin(3 * y))
    >>> result = grid.resolution(grid.evaluate(func))
    >>> result.resolved, result["x"].resolved, result["y"].resolved
    (False, False, False)
    >>> print(result["y"])
    axis y: NOT resolved, tail at 4e-02 and not decaying (16 points: too few, or the data is noisy or not smooth)
    >>> finer = result.suggested_grid
    >>> bool(finer.shape[0] > 100), finer.shape[1]
    (True, 32)
    >>> finer.resolution(finer.evaluate(func))["x"].resolved
    True
    """
    f = _as_field(grid, f)
    if not np.all(np.isfinite(f)):
        raise ValueError("The array contains non-finite values (nan or inf).")
    if not 0 < tol < 1:
        raise ValueError(f"tol must be between 0 and 1, got {tol}")
    tol = float(tol)

    results = []
    for i, axis in enumerate(grid.axes):
        basis = _basis(axis)
        if basis is None:
            results.append(AxisResolution(
                axis=i, name=axis.name, kind="", num_points=len(axis), resolved=None, needed=None,
                tail=float("nan"), decay="", rate=float("nan"), order=float("nan"), spectrum=np.empty(0), tol=tol,
            ))
            continue
        amplitudes = basis.amplitudes(basis.forward(_extend(grid, f, [i]), i), i)
        spectrum = amplitudes.max(axis=tuple(j for j in range(grid.ndims) if j != i))
        verdict = _assess(spectrum, max(tol, _ROUNDOFF * basis.num_modes))
        needed = verdict.pop("modes")
        if needed is not None:
            # Folded axes hold half of the nodes of the diameter or great circle.
            size = basis.size_for(needed)
            needed = max(2, -(-size // 2) if isinstance(axis, _FOLDED) else size)
        results.append(AxisResolution(
            axis=i, name=axis.name, kind=basis.kind, num_points=len(axis), needed=needed,
            spectrum=spectrum, tol=tol, **verdict,
        ))
    assessed = [a for a in results if a.resolved is not None]
    if not assessed:
        raise ValueError(
            "The grid has no spectral axis (periodic, Chebyshev, radial, colatitude or unbounded), so the "
            "resolution cannot be judged from expansion coefficients. Use `estimate_error` or "
            "`convergence_study` for finite-difference grids."
        )
    return Resolution(resolved=all(a.resolved for a in assessed), tol=tol, axes=tuple(results), grid=grid)


def is_resolved(grid: Grid, f: NDArray, tol: float = 1e-8) -> bool:
    """Whether a meshed function is resolved on its grid to the given tolerance.

    Short for ``resolution(grid, f, tol).resolved``; see :func:`resolution`.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev
    >>> f = lambda x: 1 / (1 + 25 * x ** 2)
    >>> [Grid(chebyshev(n, -1, 1)).is_resolved(f(Grid(chebyshev(n, -1, 1)).coords)) for n in (40, 80, 160)]
    [False, False, True]
    """
    return resolution(grid, f, tol).resolved


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def spectral_filter(
    grid: Grid,
    f: NDArray,
    method: str = "lowpass",
    *,
    keep: float | int = 2 / 3,
    alpha: float = 36.0,
    order: float = 8,
    axis: AxisSpec = None,
) -> NDArray:
    r"""Damp or remove the high modes of a meshed function.

    Available as :meth:`Grid.filter <numgrids.grids.Grid.filter>`. The
    coefficient of mode number :math:`k` is multiplied by a factor
    :math:`\sigma(k)`:

    * ``"lowpass"`` -- sharp truncation: :math:`\sigma = 1` for the
      lowest modes, 0 for the others. The default keeps two thirds of the
      modes. This is the 2/3 rule for dealiasing: the product of two
      functions filtered in this way has no aliasing errors in the modes
      that are kept, so filtering the product again gives its exact
      truncation.
    * ``"exponential"`` -- the smooth filter
      :math:`\sigma = \exp\bigl(-\alpha\,(k/k_{\max})^p\bigr)`. With the
      defaults :math:`\alpha = 36`, :math:`p = 8` the lowest modes are
      practically unchanged and the highest one is damped to round-off
      level. This stabilizes time stepping of nonlinear equations with
      much less loss of accuracy than a sharp truncation.

    On periodic axes :math:`k` is the absolute mode number, so a real
    function stays real.

    Filtering changes the function everywhere, in general also at the
    boundary points of non-periodic axes: impose boundary conditions
    after filtering, not before.

    Parameters
    ----------
    grid : Grid
        The grid.
    f : NDArray
        Meshed function of shape ``grid.shape`` (real or complex).
    method : str
        ``"lowpass"`` (default) or ``"exponential"``.
    keep : float or int
        For ``"lowpass"``: a float in ``(0, 1]`` is the fraction of the
        mode numbers that is kept on each axis (default 2/3); an int is
        the number of mode numbers that is kept, i.e. the modes
        :math:`k = 0, \ldots, \mathtt{keep} - 1`.
    alpha : float
        For ``"exponential"``: the strength :math:`\alpha \ge 0`.
    order : float
        For ``"exponential"``: the exponent :math:`p > 0`. The larger it
        is, the more modes are left untouched.
    axis : int, str, sequence of those, or None
        The axis or axes to filter. By default all spectral axes;
        finite-difference axes are left alone.

    Returns
    -------
    NDArray
        The filtered function.

    Raises
    ------
    ValueError
        If a selected axis is discretized with finite differences, or if
        the grid has no spectral axis.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, periodic
    >>> grid = Grid(x=periodic(16))
    >>> x = grid.coords
    >>> f = np.cos(2 * x) + 0.1 * np.cos(7 * x)
    >>> bool(np.allclose(grid.filter(f), np.cos(2 * x)))           # modes 0 to 5 are kept
    True
    >>> smooth = grid.filter(f, "exponential")
    >>> abs(grid.coefficients(smooth)[[2, 7]]).round(4)             # 0.5 and 0.05 before
    array([0.4997, 0.    ])
    """
    f = _as_field(grid, f)
    if method not in ("lowpass", "exponential"):
        raise ValueError(f"Unknown filter {method!r}. Use 'lowpass' or 'exponential'.")
    if method == "lowpass":
        if isinstance(keep, bool) or not isinstance(keep, (int, float, np.integer, np.floating)):
            raise TypeError(f"keep must be a fraction or a number of modes, got {keep!r}")
        is_count = isinstance(keep, (int, np.integer))
        if (is_count and keep < 1) or (not is_count and not 0 < keep <= 1):
            raise ValueError(f"keep must be a fraction in (0, 1] or a positive number of modes, got {keep!r}")
    elif alpha < 0 or order <= 0:
        raise ValueError(f"alpha must not be negative and order must be positive, got alpha={alpha}, order={order}")

    result = f
    for i, basis in _spectral_axes(grid, axis).items():
        k = np.abs(basis.mode_numbers())
        if method == "lowpass":
            # The tiny offset guards fractions like 2/3 of 33 modes against rounding down.
            kept = int(keep) if is_count else max(1, int(np.floor(keep * basis.num_modes + 1e-9)))
            sigma = (k < kept).astype(float)
        else:
            sigma = np.exp(-alpha * (k / (basis.num_modes - 1)) ** order)
        c = basis.forward(_extend(grid, result, [i]), i)
        result = _restrict(grid, basis.backward(c * _along(sigma, c.ndim, i), i), [i])
    if np.iscomplexobj(result) and not np.iscomplexobj(f):
        result = result.real
    return result

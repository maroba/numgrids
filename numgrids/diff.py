"""Differentiation strategies for the different axis types.

Each axis type selects the most suitable method to compute partial
derivatives along that axis:

* :class:`FiniteDifferenceDiff` -- equidistant, non-periodic axes,
* :class:`FFTDiff` -- equidistant, periodic axes (spectral accuracy),
* :class:`ChebyshevDiff` -- Chebyshev axes (spectral accuracy),
* :class:`LogDiff` -- logarithmic axes,
* :class:`RadialDiff`, :class:`ColatitudeDiff` -- axes through the origin
  and the poles of curvilinear grids (spectral accuracy),
* :class:`MappedChebyshevDiff` -- axes on unbounded domains (spectral accuracy).

Users normally do not instantiate these classes directly but use
:class:`numgrids.Diff`, :func:`numgrids.diff` or :meth:`numgrids.Grid.diff`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.sparse
from numpy.fft import fft, fftfreq, ifft
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from numgrids.axes import ColatitudeAxis, EquidistantAxis, LogAxis, RadialAxis, UnboundedAxis
from numgrids.utils import embed_1d_operator

if TYPE_CHECKING:
    from numgrids.grids import Grid


# ---------------------------------------------------------------------------
# findiff compatibility
# ---------------------------------------------------------------------------

def _make_findiff_operator(axis_index: int, spacing: float, order: int, acc: int):
    """Create a findiff finite-difference operator.

    Uses the modern ``findiff.Diff`` API (findiff >= 0.11) when available
    and falls back to the legacy ``FinDiff`` class otherwise.
    """
    try:
        from findiff import Diff as _FDDiff
        if "grid" not in inspect.signature(_FDDiff.__init__).parameters:
            raise ImportError
    except ImportError:  # pragma: no cover - legacy findiff
        from findiff import FinDiff
        return FinDiff(axis_index, spacing, order, acc=acc)
    return _FDDiff(axis_index, spacing, acc=acc) ** order


def _check_stencil_fits(num_points: int, order: int, acc: int) -> None:
    """Raise a helpful error if a finite-difference stencil is wider than the axis."""
    needed = order + acc + 1
    if num_points < needed:
        raise ValueError(
            f"Too few grid points ({num_points}) for a derivative of order {order} "
            f"with accuracy order {acc}; need at least {needed}. Use more points "
            f"or a lower `acc`."
        )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GridDiff:
    """Base class for grid differentiation operators.

    A ``GridDiff`` encapsulates the logic for computing a partial derivative
    of a given order along one axis of a :class:`~numgrids.grids.Grid`.

    Subclasses must set ``self.operator`` to a callable ``(NDArray) -> NDArray``
    and may override :meth:`as_matrix` to provide a sparse-matrix
    representation.
    """

    def __init__(self, grid: Grid, order: int, axis_index: int) -> None:
        """Initialize the differentiation operator.

        Parameters
        ----------
        grid : Grid
            The grid on which differentiation is performed.
        order : int
            Derivative order (e.g. 1 for first derivative).
        axis_index : int
            Index of the axis along which to differentiate.
        """
        self.axis = grid.get_axis(axis_index)
        self.order = order
        self.grid = grid
        self.axis_index = axis_index

    def __call__(self, f: NDArray) -> NDArray:
        """Apply the differentiation operator to a meshed function.

        Parameters
        ----------
        f : NDArray
            Array of shape ``grid.shape``.

        Returns
        -------
        NDArray
            The derivative, same shape as *f*.
        """
        f = np.asarray(f)
        if f.shape != self.grid.shape:
            raise ValueError(
                f"Array of shape {f.shape} does not match grid shape {self.grid.shape}."
            )
        return self.operator(f)

    def as_matrix(self) -> spmatrix:
        """Return a sparse-matrix representation of the operator.

        Returns
        -------
        spmatrix
            Scipy sparse matrix of shape ``(grid.size, grid.size)``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------

class FiniteDifferenceDiff(GridDiff):
    """Finite-difference partial derivative.

    Wraps the ``findiff`` library to apply finite-difference stencils on
    equidistant, non-periodic grids.  The accuracy order (stencil width) is
    configurable via the *acc* parameter.
    """

    def __init__(self, grid: Grid, order: int, axis_index: int, acc: int = 4) -> None:
        super().__init__(grid, order, axis_index)
        if not isinstance(self.axis, EquidistantAxis):
            raise TypeError(f"Axis must be of type EquidistantAxis. Got: {type(self.axis)}")

        self.acc = acc
        _check_stencil_fits(len(self.axis), order, acc)
        self.operator = _make_findiff_operator(axis_index, self.axis.spacing, order, acc)

    def as_matrix(self) -> spmatrix:
        return self.operator.matrix(self.grid.shape)


# ---------------------------------------------------------------------------
# FFT spectral
# ---------------------------------------------------------------------------

class FFTDiff(GridDiff):
    r"""FFT spectral partial derivative for periodic grids.

    Computes derivatives via the discrete Fourier transform:

    .. math::
        \widehat{f^{(n)}}(k) = (i k)^n \, \hat{f}(k)

    The (dense, circulant) spectral differentiation matrix is only built
    when :meth:`as_matrix` is called.

    Only valid for :class:`~numgrids.axes.EquidistantAxis` with
    ``periodic=True``.
    """

    def __init__(self, grid: Grid, order: int, axis_index: int, acc: int = 6) -> None:
        super().__init__(grid, order, axis_index)

        if not isinstance(self.axis, EquidistantAxis):
            raise TypeError("Spectral FFT differentiation requires equidistant axis.")
        if not self.axis.periodic:
            raise TypeError("Spectral FFT differentiation requires periodic boundary conditions.")

        self.acc = acc
        n = len(self.axis)
        period = n * self.axis.spacing
        scale = 2 * np.pi / period

        k_values = fftfreq(n) * n
        self._W_1d = (1j * k_values * scale) ** order
        if n % 2 == 0 and order % 2 == 1:
            self._W_1d[n // 2] = 0

        self.operator = self._setup_operator(grid)
        self._D: spmatrix | None = None

    def as_matrix(self) -> spmatrix:
        if self._D is None:
            self._D = self._build_spectral_matrix()
        return self._D

    def _setup_operator(self, grid: Grid) -> Callable[[NDArray], NDArray]:
        shape = [1] * grid.ndims
        shape[self.axis_index] = len(self.axis)
        W = self._W_1d.reshape(shape)

        def operator(f: NDArray) -> NDArray:
            F = fft(f, axis=self.axis_index)
            dF = W * F
            result = ifft(dF, axis=self.axis_index)
            if np.iscomplexobj(f):
                return result
            return np.real(result)

        return operator

    def _build_spectral_matrix(self) -> spmatrix:
        """Build the spectral differentiation matrix.

        The matrix ``F^{-1} diag(W) F`` is circulant, so it is fully
        determined by its first column ``ifft(W)``.  For multi-dimensional
        grids it is embedded via Kronecker products with identity matrices.
        """
        column = np.real(ifft(self._W_1d))
        D = scipy.sparse.csc_matrix(scipy.linalg.circulant(column))
        return embed_1d_operator(D, self.grid.shape, self.axis_index)


# ---------------------------------------------------------------------------
# Chebyshev spectral
# ---------------------------------------------------------------------------

def chebyshev_diff_matrix(num_points: int) -> NDArray:
    """First-order Chebyshev differentiation matrix on ``[-1, 1]``.

    The nodes are ordered *ascending*, ``x_k = -cos(k pi / N)``, which is the
    ordering used by :class:`~numgrids.axes.ChebyshevAxis` for its external
    coordinates.  The diagonal is computed with the "negative sum trick"
    for improved round-off behavior.

    Reference: Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.
    """
    N = num_points - 1
    k = np.arange(N + 1)
    x = np.sin(np.pi * (N - 2 * k) / (2 * N))
    c = np.ones(N + 1)
    c[0] = c[-1] = 2
    c *= (-1.0) ** k
    dX = x[:, None] - x[None, :]
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    # Reverse the node ordering (descending -> ascending):
    return D[::-1, ::-1].copy()


class ChebyshevDiff(GridDiff):
    """Chebyshev spectral partial derivative.

    Constructs the Chebyshev differentiation matrix on the canonical
    interval ``[-1, 1]``, rescales it to the user domain and raises it to
    the requested power.  For multi-dimensional grids the 1-D matrix is
    applied along the respective axis; the full sparse matrix (Kronecker
    product with identities) is only built when :meth:`as_matrix` is called.

    Reference: Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.
    """

    def __init__(self, grid: Grid, order: int, axis_index: int) -> None:
        super().__init__(grid, order, axis_index)
        scale = self.axis[-1] - self.axis[0]
        D1 = chebyshev_diff_matrix(len(self.axis)) * (2 / scale)
        self._matrix_1d = np.linalg.matrix_power(D1, order)
        self._diff_matrix: spmatrix | None = None

        D = self._matrix_1d

        def operator(f: NDArray) -> NDArray:
            df = np.tensordot(D, f, axes=([1], [axis_index]))
            return np.moveaxis(df, 0, axis_index)

        self.operator = operator

    def as_matrix(self) -> spmatrix:
        if self._diff_matrix is None:
            self._diff_matrix = embed_1d_operator(
                scipy.sparse.csc_matrix(self._matrix_1d), self.grid.shape, self.axis_index
            )
        return self._diff_matrix


# ---------------------------------------------------------------------------
# Logarithmic axes
# ---------------------------------------------------------------------------

def _falling_factorial_coefficients(order: int) -> NDArray:
    r"""Coefficients :math:`c_m` of :math:`\prod_{k=0}^{n-1} (\theta - k) = \sum_m c_m \theta^m`.

    These are the signed Stirling numbers of the first kind.
    """
    coeffs = np.array([1.0])
    for k in range(order):
        shifted = np.concatenate([[0.0], coeffs])      # theta * p(theta)
        scaled = np.concatenate([coeffs, [0.0]]) * k   # k * p(theta)
        coeffs = shifted - scaled
    return coeffs


class LogDiff(GridDiff):
    r"""Partial derivative on a logarithmic axis.

    With :math:`u = \ln x` and :math:`\theta = d/du`, the identity

    .. math::
        x^n \frac{d^n f}{dx^n} = \theta(\theta - 1)\cdots(\theta - n + 1)\, f

    expresses the *n*-th derivative in terms of derivatives on the
    equidistant log-scale coordinate, which are computed with finite
    differences.  For ``n = 1`` this reduces to the familiar chain rule
    :math:`df/dx = (1/x)\, df/du`.
    """

    def __init__(self, grid: Grid, order: int, axis_index: int, acc: int = 6) -> None:
        super().__init__(grid, order, axis_index)
        if not isinstance(self.axis, LogAxis):
            raise TypeError(f"Axis must be of type LogAxis. Got: {type(self.axis)}")

        self.acc = acc
        _check_stencil_fits(len(self.axis), order, acc)
        u = self.axis.coords_internal
        du = u[1] - u[0]
        self._coeffs = _falling_factorial_coefficients(order)
        self._fds = {
            m: _make_findiff_operator(axis_index, du, m, acc)
            for m in range(1, order + 1)
            if self._coeffs[m] != 0
        }

        shape = [1] * grid.ndims
        shape[axis_index] = len(self.axis)
        self._x_pow = (self.axis.coords ** (-order)).reshape(shape)

        def operator(f: NDArray) -> NDArray:
            result = np.zeros(f.shape, dtype=np.result_type(f, float))
            for m, fd in self._fds.items():
                result = result + self._coeffs[m] * fd(f)
            return self._x_pow * result

        self.operator = operator

    def as_matrix(self) -> spmatrix:
        """Return sparse-matrix representation of the log-scale derivative.

        Constructs the matrix as ``diag(x^-n) @ sum_m c_m D_u^m``, where
        ``D_u^m`` are finite-difference matrices on the equidistant
        log-scale coordinates.
        """
        shape = self.grid.shape
        D = None
        for m, fd in self._fds.items():
            term = self._coeffs[m] * fd.matrix(shape)
            D = term if D is None else D + term
        x_pow = np.broadcast_to(self._x_pow, shape).ravel()
        return scipy.sparse.diags(x_pow) @ D


# ---------------------------------------------------------------------------
# Radial axes through the origin
# ---------------------------------------------------------------------------

def _check_parity(parity: int) -> int:
    if parity not in (1, -1):
        raise ValueError(f"parity must be +1 or -1, got {parity!r}.")
    return int(parity)


class RadialDiff(GridDiff):
    r"""Chebyshev derivative along a :class:`~numgrids.axes.RadialAxis`.

    The radial axis stores the positive half of the Chebyshev nodes of the
    diameter :math:`[-R, R]`. The values on the negative half are those at
    the antipodal points :math:`(r, \varphi + \pi)` (and :math:`\pi - \theta`
    on spherical grids), which the grid supplies. With the full Chebyshev
    differentiation matrix :math:`D` split into the blocks acting on the
    positive (:math:`A`) and negative (:math:`B`) half,

    .. math::
        \partial_r^n f = A\, f + B\, (\mathcal{P} f),

    where :math:`\mathcal{P}` is the antipodal map. This is spectrally
    accurate for smooth (single-valued) fields and has no singularity at the
    origin.

    Scalar fields are *even* across the origin in this sense. Quantities that
    change sign when continued across the origin (e.g. the radial component
    of a vector field, whose unit vector flips) are *odd*:
    :math:`\partial_r^n f = A\, f - B\, (\mathcal{P} f)`; pass ``parity=-1``.

    Reference: Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000, ch. 11.
    """

    def __init__(self, grid: Grid, order: int, axis_index: int, parity: int = 1) -> None:
        super().__init__(grid, order, axis_index)
        if not isinstance(self.axis, RadialAxis):
            raise TypeError(f"Axis must be of type RadialAxis. Got: {type(self.axis)}")
        self._antipodal = grid._antipodal_map(axis_index)
        self._parity = _check_parity(parity)

        n = len(self.axis)
        D = chebyshev_diff_matrix(2 * n) / self.axis.high      # ascending diameter nodes
        D = np.linalg.matrix_power(D, order)
        # Diameter node q (ascending) <-> radial index: q >= n -> q - n,
        # q < n -> n - 1 - q at the antipodal point.
        self._A = D[n:, n:].copy()
        self._B = self._parity * D[n:, :n][:, ::-1]
        self._matrix: spmatrix | None = None

        A, B, antipodal = self._A, self._B, self._antipodal

        def apply(M: NDArray, f: NDArray) -> NDArray:
            return np.moveaxis(np.tensordot(M, f, axes=([1], [axis_index])), 0, axis_index)

        def operator(f: NDArray) -> NDArray:
            return apply(A, f) + apply(B, antipodal(f))

        self.operator = operator

    def as_matrix(self) -> spmatrix:
        if self._matrix is None:
            shape = self.grid.shape
            size = int(np.prod(shape))
            perm = self._antipodal(np.arange(size).reshape(shape)).ravel()
            P = scipy.sparse.csr_matrix((np.ones(size), (np.arange(size), perm)), shape=(size, size))
            A = embed_1d_operator(scipy.sparse.csc_matrix(self._A), shape, self.axis_index)
            B = embed_1d_operator(scipy.sparse.csc_matrix(self._B), shape, self.axis_index)
            self._matrix = scipy.sparse.csc_matrix(A + B @ P)
        return self._matrix


# ---------------------------------------------------------------------------
# Colatitude axes through the poles
# ---------------------------------------------------------------------------

def fourier_diff_matrix(num_points: int, order: int, period: float = 2 * np.pi) -> NDArray:
    """Dense Fourier spectral differentiation matrix of the given order.

    For an even number of points and odd orders the Nyquist mode is set to
    zero (as in :class:`FFTDiff`); even orders keep it, so that e.g. the
    second derivative has no spurious null space.
    """
    k = fftfreq(num_points) * num_points
    W = (1j * k * 2 * np.pi / period) ** order
    if num_points % 2 == 0 and order % 2 == 1:
        W[num_points // 2] = 0
    return np.real(scipy.linalg.circulant(np.real(ifft(W))))


class ColatitudeDiff(GridDiff):
    r"""Fourier derivative along a :class:`~numgrids.axes.ColatitudeAxis`.

    The axis stores the half :math:`0 < \theta < \pi` of the great circle
    :math:`[0, 2\pi)`. The values on the far half are those at
    :math:`(\theta, \varphi + \pi)` (supplied by the grid), so the derivative
    is computed with the Fourier differentiation matrix :math:`D` of the full
    circle, split into the blocks acting on the near (:math:`A`) and far
    (:math:`B`) half:

    .. math::
        \partial_\theta^n f = A\, f + B\, (\mathcal{P} f).

    As for :class:`RadialDiff`, ``parity=-1`` differentiates quantities that
    change sign when continued across a pole.
    """

    def __init__(self, grid: Grid, order: int, axis_index: int, parity: int = 1) -> None:
        super().__init__(grid, order, axis_index)
        if not isinstance(self.axis, ColatitudeAxis):
            raise TypeError(f"Axis must be of type ColatitudeAxis. Got: {type(self.axis)}")
        self._antipodal = grid._antipodal_map(axis_index)
        self._parity = _check_parity(parity)

        n = len(self.axis)
        D = fourier_diff_matrix(2 * n, order)
        # Great-circle node j >= n is the point (2 pi - theta_j, phi) = (theta_{2n-1-j}, phi + pi).
        self._A = D[:n, :n].copy()
        self._B = self._parity * D[:n, n:][:, ::-1]
        self._matrix: spmatrix | None = None

        A, B, antipodal = self._A, self._B, self._antipodal

        def apply(M: NDArray, f: NDArray) -> NDArray:
            return np.moveaxis(np.tensordot(M, f, axes=([1], [axis_index])), 0, axis_index)

        def operator(f: NDArray) -> NDArray:
            return apply(A, f) + apply(B, antipodal(f))

        self.operator = operator

    as_matrix = RadialDiff.as_matrix


# ---------------------------------------------------------------------------
# Unbounded axes (mapped Chebyshev)
# ---------------------------------------------------------------------------

def barycentric_diff_matrix(differences: NDArray, weights: NDArray) -> NDArray:
    r"""First-order polynomial differentiation matrix on arbitrary nodes.

    For nodes :math:`\xi_j` with barycentric weights :math:`w_j`,

    .. math::
        D_{ij} = \frac{w_j / w_i}{\xi_i - \xi_j} \quad (i \ne j), \qquad
        D_{ii} = -\sum_{j \ne i} D_{ij}.

    The diagonal is computed with the "negative sum trick", so that
    constants are differentiated to zero exactly (up to round-off).

    Parameters
    ----------
    differences : NDArray
        The node differences :math:`\xi_i - \xi_j`, shape ``(n, n)``. They
        are passed in (instead of the nodes) because the caller can often
        compute them without cancellation.
    weights : NDArray
        The barycentric weights, shape ``(n,)`` (a common factor is irrelevant).

    Reference: Berrut & Trefethen, *Barycentric Lagrange Interpolation*,
    SIAM Review 46 (2004).
    """
    n = len(weights)
    D = (weights[None, :] / weights[:, None]) / (differences + np.eye(n))
    D[np.diag_indices(n)] = 0.0
    D[np.diag_indices(n)] = -D.sum(axis=1)
    return D


class MappedChebyshevDiff(GridDiff):
    r"""Spectral partial derivative on an unbounded axis.

    Used for :class:`~numgrids.axes.SemiInfiniteAxis` and
    :class:`~numgrids.axes.InfiniteAxis`, whose points are the images
    :math:`x(\xi_j)` of Chebyshev-type nodes on :math:`[-1, 1]`. The first
    derivative is

    .. math::
        \frac{d}{dx} = \frac{d\xi}{dx}\,\frac{d}{d\xi},

    with the polynomial differentiation matrix on the nodes
    (:func:`barycentric_diff_matrix`). Higher derivatives are expanded with
    the chain rule,

    .. math::
        \frac{d^n}{dx^n} = \sum_{k=1}^{n} c_{n,k}(\xi)\,\frac{d^k}{d\xi^k},

    rather than computed as powers of the first-derivative matrix: a
    derivative of a polynomial in :math:`\xi` need not be one (on the whole
    line it has a factor :math:`(1 - \xi^2)^{3/2}`), and differentiating it
    again as if it were would spoil the spectral accuracy for functions
    that decay algebraically. The coefficients vanish at infinity, so no
    condition has to be imposed there.

    Reference: Boyd, *Chebyshev and Fourier Spectral Methods*, 2nd ed.,
    Dover, 2001, ch. 17.
    """

    def __init__(self, grid: Grid, order: int, axis_index: int) -> None:
        super().__init__(grid, order, axis_index)
        if not isinstance(self.axis, UnboundedAxis):
            raise TypeError(f"Axis must be a SemiInfiniteAxis or InfiniteAxis. Got: {type(self.axis)}")
        axis = self.axis
        D_xi = barycentric_diff_matrix(axis._internal_differences(axis.coords), axis.barycentric_weights)
        D_xi_k = np.eye(len(axis))
        self._matrix_1d = np.zeros((len(axis), len(axis)))
        for c in axis._chain_rule_coefficients(order):
            D_xi_k = D_xi @ D_xi_k
            self._matrix_1d += c[:, None] * D_xi_k
        self._diff_matrix: spmatrix | None = None

        D = self._matrix_1d

        def operator(f: NDArray) -> NDArray:
            df = np.tensordot(D, f, axes=([1], [axis_index]))
            return np.moveaxis(df, 0, axis_index)

        self.operator = operator

    as_matrix = ChebyshevDiff.as_matrix

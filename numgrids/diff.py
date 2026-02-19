from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse
from numpy.fft import fft, ifft, fftfreq
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from findiff import FinDiff

from numgrids.axes import EquidistantAxis, LogAxis

if __name__ != '__main__':
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from numgrids.grids import Grid


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
        return self.operator(f)

    def as_matrix(self) -> spmatrix:
        """Return a sparse-matrix representation of the operator.

        Returns
        -------
        spmatrix
            Scipy sparse matrix of shape ``(grid.size, grid.size)``.
        """
        raise NotImplementedError


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
        self.operator = FinDiff(axis_index, self.axis.spacing, order, acc=acc)

    def as_matrix(self) -> spmatrix:
        return self.operator.matrix(self.grid.shape)


class FFTDiff(GridDiff):
    r"""FFT spectral partial derivative for periodic grids.

    Computes derivatives via the discrete Fourier transform:

    .. math::
        \widehat{f^{(n)}}(k) = (i k)^n \, \hat{f}(k)

    A sparse spectral differentiation matrix is also built so that
    :meth:`as_matrix` returns the exact spectral operator.

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
        self._D = self._build_spectral_matrix()

    def as_matrix(self) -> spmatrix:
        return self._D

    def _setup_operator(self, grid: Grid) -> Callable[[NDArray], NDArray]:
        shape = [1] * grid.ndims
        shape[self.axis_index] = len(self.axis)
        W = self._W_1d.reshape(shape)

        def operator(f: NDArray) -> NDArray:
            F = fft(f, axis=self.axis_index)
            dF = W * F
            return np.real(ifft(dF, axis=self.axis_index))

        return operator

    def _build_spectral_matrix(self) -> spmatrix:
        """Build the spectral differentiation matrix via DFT.

        Constructs D = F_inv @ diag(W) @ F, where F is the DFT matrix.
        For multi-dimensional grids, uses Kronecker products with identity
        matrices for the other axes.
        """
        n = len(self.axis)
        j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        F = np.exp(-2j * np.pi * j_idx * k_idx / n)
        F_inv = np.exp(2j * np.pi * j_idx * k_idx / n) / n

        D = np.real(F_inv @ np.diag(self._W_1d) @ F)
        D_sparse = scipy.sparse.csc_matrix(D)

        if self.grid.ndims == 1:
            return D_sparse

        result = None
        for i in range(self.grid.ndims):
            if i == self.axis_index:
                mat = D_sparse
            else:
                mat = scipy.sparse.identity(len(self.grid.axes[i]))
            result = mat if result is None else scipy.sparse.kron(result, mat)

        return result


class ChebyshevDiff(GridDiff):
    """Chebyshev spectral partial derivative.

    Constructs the Chebyshev differentiation matrix on the canonical
    interval ``[-1, 1]`` and rescales it to the user domain.  Higher-order
    derivatives are obtained by repeated multiplication of the first-order
    matrix.  For multi-dimensional grids the 1-D matrix is embedded via
    Kronecker products with identity matrices on the remaining axes.

    Reference: Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.
    """

    def __init__(self, grid: Grid, order: int, axis_index: int) -> None:
        super().__init__(grid, order, axis_index)
        self._scale = (self.axis[-1] - self.axis[0])
        self._diff_matrix = self._setup_diff_matrix()
        for _ in range(self.order - 1):
            self._diff_matrix *= self._diff_matrix
        self._diff_matrix *= (2 / self._scale) ** self.order

        def operator(f):
            f = f.reshape(-1)
            df = self._diff_matrix * f
            return df.reshape(self.grid.shape)

        self.operator = operator

    def as_matrix(self) -> spmatrix:
        return self._diff_matrix

    def _setup_diff_matrix(self) -> spmatrix:
        N = len(self.axis) - 1
        x = self.axis.coords_internal
        D = np.zeros((N + 1, N + 1))
        D[0, 0] = (2 * N ** 2 + 1) / 6
        D[N, N] = -D[0, 0]

        for j in range(1, N):
            D[j, j] = - 0.5 * x[j] / (1 - x[j] ** 2)

        for i in range(0, N + 1):
            c_i = 2 if i == 0 or i == N else 1
            for j in range(0, N + 1):
                if i == j:
                    continue
                c_j = 2 if j == 0 or j == N else 1
                D[i, j] = c_i / c_j * (-1) ** (i + j) / (x[i] - x[j])

        D = scipy.sparse.csc_matrix(-D)
        if self.grid.ndims == 1:
            return D

        # In multiple dimensions, we need a Kronecker product of
        # the 1D diff matrix and identity matrices:
        for i in range(self.grid.ndims):
            if i == self.axis_index:
                D_i = D
            else:
                D_i = scipy.sparse.identity(len(self.grid.axes[i]))

            if i == 0:
                D_mult = D_i
            else:
                D_mult = scipy.sparse.kron(D_mult, D_i)

        return D_mult


class LogDiff(GridDiff):
    r"""Partial derivative on a logarithmic axis.

    Uses finite differences on the equidistant log-scale coordinates and
    applies the chain rule:

    .. math::
        \frac{df}{dx} = \frac{1}{x}\,\frac{df}{d(\ln x)}
    """

    def __init__(self, grid: Grid, order: int, axis_index: int, acc: int = 6) -> None:
        super().__init__(grid, order, axis_index)
        if not isinstance(self.axis, LogAxis):
            raise TypeError(f"Axis must be of type LogAxis. Got: {type(self.axis)}")

        self.acc = acc
        x = self.axis.coords_internal
        self._fd = FinDiff(axis_index, x[1] - x[0], order, acc=acc)

        def operator(f: NDArray) -> NDArray:
            return self._fd(f) / self.grid.meshed_coords[axis_index]

        self.operator = operator

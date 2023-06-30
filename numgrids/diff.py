import numpy as np
import scipy.sparse
from numpy.fft import fft, ifft
from findiff import FinDiff

from numgrids.axes import EquidistantAxis, LogAxis


class GridDiff:
    """Base class for add grid differentiators.

        Child classes must implement a callable member "operator".
    """

    def __init__(self, grid, order, axis_index):
        self.axis = grid.get_axis(axis_index)
        self.order = order
        self.grid = grid
        self.axis_index = axis_index

    def __call__(self, f):
        return self.operator(f)


class FiniteDifferenceDiff(GridDiff):
    """Partial derivative based on finite difference approximations.

        Used for equidistant, non-periodic grids.
    """

    def __init__(self, grid, order, axis_index):
        super(FiniteDifferenceDiff, self).__init__(grid, order, axis_index)
        if not isinstance(self.axis, EquidistantAxis):
            raise TypeError("Axis must be of type EquidistantAxis. Got: {}".format(type(self.axis)))

        # TODO make the accuracy order flexible:
        self.operator = FinDiff(axis_index, self.axis.spacing, order, acc=4)

    def as_matrix(self):
        return self.operator.matrix(self.grid.shape)


class FFTDiff(GridDiff):
    """Partial Derivative based on FFT spectral method.

        Used for equidistant, periodic grids.
    """

    def __init__(self, grid, order, axis_index):
        super(FFTDiff, self).__init__(grid, order, axis_index)

        if not isinstance(self.axis, EquidistantAxis):
            raise TypeError("Spectral FFT differentiation requires equidistant axis.")
        if not self.axis.periodic:
            raise TypeError("Spectral FFT differentiation requires periodic boundary conditions.")

        self.operator = self._setup_operator(grid)

        # since we have no matrix representation of a FFTDiff, use finite differences
        # until we implement something better:
        self._D = FinDiff(0, self.axis.spacing, 1, acc=6).matrix((len(self.axis),))

    def as_matrix(self):
        return self._D

    def _setup_operator(self, grid):
        n = len(self.axis)
        W = 1j * np.hstack((
            np.arange(n // 2),
            np.array([0]),
            np.arange(-n // 2 + 1, 0)
        ))
        if self.order % 2:
            W[n // 2] = 0
        if self.order > 1:
            W **= self.order
        W = np.swapaxes(np.ones(grid.shape) * W, self.axis_index, -1)

        def operator(f):
            F = fft(f, axis=self.axis_index)
            dF = W * F
            return np.real(ifft(dF, axis=self.axis_index))

        return operator


class ChebyshevDiff(GridDiff):
    """Partial derivative based on Chebyshev spectral method.

        Used for grids with non-equidistant Chebyshev axes.
    """

    def __init__(self, grid, order, axis_index):
        super(ChebyshevDiff, self).__init__(grid, order, axis_index)
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

    def as_matrix(self):
        return self._diff_matrix

    def _setup_diff_matrix(self):
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
    """Partial derivative for a LogAxis.

        Based on finite differences on the (equidistant) log-scale of coordinates.
    """

    def __init__(self, grid, order, axis_index):
        super(LogDiff, self).__init__(grid, order, axis_index)
        if not isinstance(self.axis, LogAxis):
            raise TypeError("Axis must be of type LogAxis. Got: {}".format(type(self.axis)))

        def operator(f):
            x = self.axis.coords_internal
            fd = FinDiff(axis_index, x[1] - x[0], order, acc=6)
            return fd(f) / self.grid.meshed_coords[axis_index]

        self.operator = operator

import numpy as np
from numpy.fft import fft, ifft
from findiff import FinDiff

from numgrids.axes import EquidistantAxis


class GridDiff:

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
            raise TypeError("Axis must be of type EquidistantAxis. Got: {}".format(type(axis)))

        # TODO make the accuracy order flexible:
        self.operator = FinDiff(axis_index, self.axis.spacing, order, acc=4)


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


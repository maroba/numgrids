import numpy as np
from numpy.fft import fft, ifft, fftfreq
from findiff import FinDiff

# Differentiators go here

# For equidistant, periodic axes, use FFT
# For Chebyshev grids, etc.
from numgrids.axes import EquidistantAxis


class FiniteDifferenceDiff:

    def __init__(self, grid, order, axis_index):
        self.order = order
        self.axis = axis_index

        axis = grid.get_axis(axis_index)
        if not isinstance(axis, EquidistantAxis):
            raise TypeError("Axis must be of type EquidistantAxis. Got: {}".format(type(axis)))

        self.operator = FinDiff(axis_index, axis.spacing, order, acc=4)

    def __call__(self, f):
        return self.operator(f)


class FFTDiff:

    def __init__(self, grid, order, axis_index):
        axis = grid.get_axis(axis_index)
        if not isinstance(axis, EquidistantAxis):
            raise TypeError("Spectral FFT differentiation requires equidistant axis.")
        if not axis.periodic:
            raise TypeError("Spectral FFT differentiation requires periodic boundary conditions.")
        self.order = order
        self.grid = grid
        self.axis_index = axis_index
        self.axis = grid.get_axis(axis_index)

        self._W = self._setup_kvec(grid)

    def __call__(self, f):
        F = fft(f, axis=self.axis_index)
        dF = self._W * F
        return np.real(ifft(dF, axis=self.axis_index))

    def _setup_kvec(self, grid):
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
        return np.swapaxes(np.ones(grid.shape) * W, self.axis_index, -1)


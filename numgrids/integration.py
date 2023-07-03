import numpy as np

from numgrids import Diff, Grid
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix, identity, kron

from numgrids.axes import EquidistantAxis
from numgrids.utils import multi_kron


class Integral:
    """
    The integration operator for integrating on a grid.

        .. math::
            \int_V ... dV

    Integration always runs over the whole grid.
    """

    def __init__(self, grid):
        """Constructor


        Parameters
        ----------
        grid: Grid
            The grid on which to integrate.
        """
        self.grid = grid

        self.D_invs = []
        for i, axis in enumerate(self.grid.axes):
            internal_grid = Grid(axis)
            D = Diff(internal_grid, 1, 0).as_matrix()
            D_inv = inv(csc_matrix(D[1:, 1:])) # inv wants a csc_matrix, but findiff returns csr_matrix
            self.D_invs.append(D_inv)

        self.eyes = [identity(len(axis)) for axis in self.grid.axes]

    def __call__(self, f):
        """
        Apply the integration to a meshed function.

        Parameters
        ----------
        f: numpy.ndarray
            The meshed function to integrate over. The shape must be compatible with
            the shape of the grid.

        Returns
        -------
        The result of the integration (float).
        """
        f_ = f
        for i, axis in enumerate(self.grid.axes):
            if type(axis) == EquidistantAxis and axis.periodic:
                # FFT-spectral integration is much easier and still spectral accuracy!
                f_ = axis.spacing * np.sum(f_, axis=0)
            else:
                f_ = f_[1:, ...]
                shape = f_.shape
                f_ = f_.reshape(-1)
                matrices = [identity(len(axis)) for axis in self.grid.axes]
                matrices[i] = self.D_invs[i]
                D_inv_big = multi_kron(*matrices[i:])
                f_ = (D_inv_big * f_).reshape(shape)[-1, ...]

        return f_
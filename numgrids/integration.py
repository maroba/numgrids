from numgrids import Diff, Grid
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix, identity, kron

from numgrids.utils import multi_kron


class Integral:

    def __init__(self, grid):
        self.grid = grid

        self.D_invs = []
        for i, axis in enumerate(self.grid.axes):
            internal_grid = Grid(axis)
            D = Diff(internal_grid, 1, 0).as_matrix()
            D_inv = inv(csc_matrix(D[1:, 1:])) # inv wants a csc_matrix, but findiff returns csr_matrix
            self.D_invs.append(D_inv)

        self.eyes = [identity(len(axis)) for axis in self.grid.axes]

    def __call__(self, f):

        f_ = f
        for i, axis in enumerate(self.grid.axes):
            f_ = f_[1:, ...]
            shape = f_.shape
            f_ = f_.reshape(-1)
            matrices = [identity(len(axis)) for axis in self.grid.axes]
            matrices[i] = self.D_invs[i]
            D_inv_big = multi_kron(*matrices[i:])
            f_ = (D_inv_big * f_).reshape(shape)[-1, ...]

        return f_
"""Direct solvers for the linear systems of operators on grids.

The matrices of spectral methods are sparse only in the sense of having a
tensor-product structure: along each axis they are dense.  Sparse LU
factorizations fill in heavily on them, so two other strategies are often
much faster:

* **dense LU** for moderate numbers of unknowns, and
* **Fourier decoupling**: if the system does not change under shifts along
  one or more periodic axes (e.g. the azimuthal angle of a disk, cylinder
  or ball, as long as the coefficients do not depend on it), a discrete
  Fourier transform along these axes splits the system into small,
  independent systems, one per Fourier mode.

:class:`LinearSolver` chooses between these and sparse LU, and keeps the
factorization for solving with many right-hand sides.

Singular systems, such as a Poisson problem with Neumann conditions on the
whole boundary (which determines the solution only up to a constant), are
solved in the least-squares sense: if the system is consistent, the result
is one of its solutions.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numgrids.grids import Grid
    from numgrids.operators import Operator

_METHODS = ("auto", "fourier", "dense", "sparse")

# Systems up to this size are solved with dense LU, unless they are banded
# along some axes (finite differences), where sparse LU has little fill-in.
_DENSE_LIMIT = 8000
# Up to this many points along the banded axes, a system still counts as dense.
_BANDED_MIN_POINTS = 8
# A dense matrix whose smallest LU pivot is below this fraction of the largest is singular.
_SINGULAR_PIVOT = 1e-13
# Fourier decoupling: bounds on the memory (bytes) and work (flops) of the dense mode blocks.
_FOURIER_MAX_BYTES = 1.5e9
_FOURIER_MAX_WORK = 2e11


def shift_invariant_axes(A, grid: Grid) -> list[int]:
    """Indices of the periodic axes along which the matrix *A* is shift invariant.

    *A* acts on flattened fields of the grid.  It is shift invariant along
    an axis if it commutes with the cyclic shift by one grid point along
    that axis, i.e. if it is block circulant.
    """
    A = scipy.sparse.csr_matrix(A)
    if A.nnz == 0:
        return []
    scale = np.max(np.abs(A.data))
    idx = np.arange(grid.size).reshape(grid.shape)
    axes = []
    for p, axis in enumerate(grid.axes):
        if not axis.periodic or len(axis) < 2:
            continue
        perm = np.roll(idx, -1, axis=p).ravel()
        diff = A[perm][:, perm] - A
        if diff.nnz == 0 or np.max(np.abs(diff.data)) <= 1e-10 * scale:
            axes.append(p)
    return axes


def banded_points(A, grid: Grid, axes=None) -> int:
    """Number of grid points along the axes on which the matrix *A* is banded.

    Along an axis with a spectral discretization every point is coupled to
    all others on the same grid line, while finite differences only couple
    neighbors.  The product of the lengths of the banded axes (among *axes*,
    default all) tells how much a sparse factorization can gain: 1 means
    that the matrix is dense in every direction.
    """
    coo = scipy.sparse.coo_matrix(A)
    rows = np.unravel_index(coo.row, grid.shape)
    cols = np.unravel_index(coo.col, grid.shape)
    points = 1
    for p in (range(grid.ndims) if axes is None else axes):
        n = grid.shape[p]
        offset = np.abs(rows[p] - cols[p])
        if grid.axes[p].periodic:
            offset = np.minimum(offset, n - offset)
        if offset.size and offset.max() < max(n // 2, 1):
            points *= n
    return points


class FourierBlocks:
    r"""Block diagonalization of shift-invariant matrices by a discrete Fourier transform.

    A field on the grid is indexed by ``(a, j)``, where *j* runs over the
    points of the shift-invariant (periodic) axes and *a* over the points
    of the remaining axes.  A shift-invariant matrix has the entries
    :math:`A_{(a,i),(b,j)} = C_{ab}(j - i)`.  It maps the Fourier mode
    :math:`\hat u_b\, e^{2\pi i m j / N}` to
    :math:`(\hat A_m \hat u)_a\, e^{2\pi i m i / N}` with the small matrices
    :math:`\hat A_m = \sum_d C(d)\, e^{2\pi i m d / N}`.
    """

    def __init__(self, grid: Grid, axes: list[int]) -> None:
        shape = grid.shape
        self.axes = list(axes)
        rest = [i for i in range(len(shape)) if i not in self.axes]
        self.mode_shape = tuple(shape[a] for a in self.axes)
        self.n_modes = int(np.prod(self.mode_shape))
        self.n_rest = grid.size // self.n_modes
        self.size = grid.size
        #: flat index of the grid point (a, j), shape (n_rest, n_modes)
        self.index = (np.arange(grid.size).reshape(shape)
                      .transpose(rest + self.axes).reshape(self.n_rest, self.n_modes))
        self.rest_axes = rest
        self._fft_axes = tuple(range(1, 1 + len(self.axes)))

    def work(self) -> float:
        """Rough number of floating point operations of factorizing all blocks."""
        return self.n_modes * float(self.n_rest) ** 3

    def nbytes(self) -> float:
        """Memory needed for the dense blocks (complex)."""
        return 16.0 * self.n_rest * self.size

    def blocks(self, A) -> NDArray:
        """The blocks :math:`\\hat A_m` of a shift-invariant matrix, shape ``(n_modes, n_rest, n_rest)``."""
        A = scipy.sparse.csr_matrix(A)
        C = A[self.index[:, 0]].toarray()[:, self.index.ravel()]
        C = C.reshape((self.n_rest, self.n_rest) + self.mode_shape)
        axes = tuple(range(2, 2 + len(self.axes)))
        hat = np.fft.ifftn(C, axes=axes) * self.n_modes
        return np.moveaxis(hat.reshape(self.n_rest, self.n_rest, self.n_modes), -1, 0)

    def sparse_blocks(self, A) -> list:
        """The blocks :math:`\\hat A_m` as sparse (CSC) matrices, for banded blocks."""
        ref = scipy.sparse.coo_matrix(scipy.sparse.csr_matrix(A)[self.index[:, 0]])
        rest_of = np.empty(self.size, dtype=np.intp)
        shift_of = np.empty(self.size, dtype=np.intp)
        rest_of[self.index] = np.arange(self.n_rest)[:, None]
        shift_of[self.index] = np.arange(self.n_modes)[None, :]
        b, d = rest_of[ref.col], shift_of[ref.col]
        shape = (self.n_rest, self.n_rest)
        return [scipy.sparse.csc_matrix((ref.data * self.mode(m)[d], (ref.row, b)), shape=shape)
                for m in range(self.n_modes)]

    def forward(self, b: NDArray) -> NDArray:
        """Fourier coefficients of a flat field, shape ``(n_rest, n_modes)``."""
        f = b[self.index].reshape((self.n_rest,) + self.mode_shape)
        return np.fft.fftn(f, axes=self._fft_axes).reshape(self.n_rest, self.n_modes)

    def backward(self, coeffs: NDArray) -> NDArray:
        """The flat field with the given Fourier coefficients."""
        f = np.fft.ifftn(coeffs.reshape((self.n_rest,) + self.mode_shape), axes=self._fft_axes)
        out = np.empty(self.size, dtype=complex)
        out[self.index] = f.reshape(self.n_rest, self.n_modes)
        return out

    def mode(self, m: int) -> NDArray:
        r"""The values :math:`e^{2\pi i m j / N}` of Fourier mode *m* on the points *j*."""
        coeffs = np.zeros(self.n_modes, dtype=complex)
        coeffs[m] = self.n_modes
        return np.fft.ifftn(coeffs.reshape(self.mode_shape)).ravel()

    def conjugate_mode(self, m: int) -> int:
        """Index of the mode with the opposite wave numbers."""
        multi = np.unravel_index(m, self.mode_shape)
        return int(np.ravel_multi_index([(-k) % n for k, n in zip(multi, self.mode_shape)], self.mode_shape))


def _dense_solver(A: NDArray):
    """A function solving ``A x = b`` for a dense matrix (least squares if *A* is singular)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
        lu, piv = scipy.linalg.lu_factor(A, check_finite=False)
    pivots = np.abs(np.diag(lu))
    if not np.all(np.isfinite(pivots)) or pivots.min() <= _SINGULAR_PIVOT * pivots.max():
        pinv = scipy.linalg.pinv(A)
        return lambda b: pinv @ b
    return lambda b: scipy.linalg.lu_solve((lu, piv), b, check_finite=False)


def _sparse_solver(A):
    """A function solving ``A x = b`` for a sparse matrix."""
    A = scipy.sparse.csc_matrix(A)
    try:
        lu = scipy.sparse.linalg.splu(A)
    except RuntimeError as err:  # exactly singular
        if A.shape[0] > _DENSE_LIMIT:
            raise np.linalg.LinAlgError(
                "The system is singular. If the solution is only determined up to a constant "
                "(e.g. Neumann conditions on the whole boundary), prescribe its value somewhere."
            ) from err
        return _dense_solver(A.toarray())
    if np.iscomplexobj(A):
        return lambda b: lu.solve(np.asarray(b, dtype=complex))

    def solve(b):
        if np.iscomplexobj(b):  # real factorization, complex right-hand side
            return lu.solve(b.real) + 1j * lu.solve(b.imag)
        return lu.solve(b)

    return solve


class LinearSolver:
    """A factorized linear boundary value problem ``L u = rhs``.

    Created by :meth:`Operator.factorize <numgrids.operators.Operator.factorize>`.
    Calling the solver with a right-hand side returns the solution; the
    (expensive) factorization is done only once, so solving for many
    right-hand sides is cheap, e.g. in implicit time stepping.

    Parameters
    ----------
    operator : Operator
        The operator *L*.
    bcs : optional
        Boundary conditions, in any form accepted by
        :meth:`Operator.solve <numgrids.operators.Operator.solve>`.  Their
        values are fixed at construction.
    method : str
        ``"auto"`` (default), ``"fourier"``, ``"dense"`` or ``"sparse"``;
        see :meth:`Operator.solve <numgrids.operators.Operator.solve>`.

    Attributes
    ----------
    method : str
        The method actually used.

    Examples
    --------
    Implicit Euler steps for the heat equation, with one factorization:

    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev
    >>> grid = Grid(x=chebyshev(30, 0, 1))
    >>> dt = 0.01
    >>> step = (1 - dt * grid.laplacian_operator()).factorize(bcs=0)
    >>> u = np.sin(np.pi * grid.coords)
    >>> for _ in range(10):
    ...     u = step(u)
    >>> bool(np.allclose(u, (1 + dt * np.pi ** 2) ** -10 * np.sin(np.pi * grid.coords)))
    True
    """

    def __init__(self, operator: Operator, bcs=None, method: str = "auto") -> None:
        from numgrids.boundary import resolve_bcs
        from numgrids.operators import _count_zero_rows

        if method not in _METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {_METHODS}.")
        grid = operator.grid
        self.grid = grid
        L, b = operator.with_bcs(0.0, bcs)
        self._bc_rows = np.zeros(grid.size, dtype=bool)
        for bc in resolve_bcs(grid, bcs):
            self._bc_rows[bc.face.flat_indices] = True
        self._bc_values = b[self._bc_rows]
        n_zero = _count_zero_rows(L)
        if n_zero:
            raise ValueError(
                f"The system has {n_zero} empty row(s) not covered by boundary conditions, "
                "so it is singular. This typically happens when grid points lie on a "
                "coordinate singularity (e.g. r = 0); start the axis slightly away from it "
                "or impose a condition there."
            )
        self._real = not np.iscomplexobj(L)
        self._fourier: FourierBlocks | None = None
        self._sparse_blocks = False
        self.method = self._choose(L, method)
        if self.method == "fourier" and self._sparse_blocks:
            self._solve = [_sparse_solver(block) for block in self._fourier.sparse_blocks(L)]
        elif self.method == "fourier":
            self._solve = [_dense_solver(block) for block in self._fourier.blocks(L)]
        elif self.method == "dense":
            self._solve = _dense_solver(L.toarray())
        else:
            self._solve = _sparse_solver(L)

    def _choose(self, L, method: str) -> str:
        n = L.shape[0]
        if method in ("auto", "fourier"):
            axes = shift_invariant_axes(L, self.grid)
            if axes:
                blocks = FourierBlocks(self.grid, axes)
                # Blocks that are banded (finite differences on the other axes) stay sparse.
                self._sparse_blocks = banded_points(L, self.grid, blocks.rest_axes) > _BANDED_MIN_POINTS
                if (method == "fourier" or self._sparse_blocks
                        or (blocks.nbytes() <= _FOURIER_MAX_BYTES and blocks.work() <= _FOURIER_MAX_WORK)):
                    self._fourier = blocks
                    return "fourier"
            elif method == "fourier":
                raise ValueError(
                    "The system is not shift invariant along any periodic axis "
                    "(its coefficients or boundary values depend on all of them), "
                    "so it cannot be decoupled by a Fourier transform."
                )
        if method == "dense" or (method == "auto" and n <= _DENSE_LIMIT
                                 and banded_points(L, self.grid) <= _BANDED_MIN_POINTS):
            return "dense"
        return "sparse"

    def __call__(self, rhs=0.0) -> NDArray:
        """Solve for a right-hand side (scalar, array of grid shape, or callable)."""
        grid = self.grid
        if callable(rhs):
            rhs = grid.evaluate(rhs)
        b = np.broadcast_to(np.asarray(rhs), grid.shape).ravel()
        b = np.array(b, dtype=np.result_type(b, self._bc_values, float), copy=True)
        b[self._bc_rows] = self._bc_values
        real = self._real and not np.iscomplexobj(b)
        if self.method == "fourier":
            coeffs = self._fourier.forward(b)
            for m, solve in enumerate(self._solve):
                coeffs[:, m] = solve(coeffs[:, m])
            u = self._fourier.backward(coeffs)
        else:
            u = self._solve(b)
        if real:
            u = np.real(u)
        return np.asarray(u).reshape(grid.shape)

    solve = __call__

    def __repr__(self) -> str:
        return f"LinearSolver(grid={type(self.grid).__name__}{self.grid.shape}, method={self.method!r})"

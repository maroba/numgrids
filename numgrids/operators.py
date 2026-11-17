"""Linear operators on grids with an algebra and built-in solvers.

An :class:`Operator` wraps a sparse matrix acting on functions meshed on a
:class:`~numgrids.grids.Grid`.  Operators can be combined with ``+``,
``-``, ``*``, ``@``, ``/`` and ``**``:

* ``op + c`` with a scalar *c* means ``op + c * I``,
* ``V * op`` with an array *V* of grid shape means "apply *op*, then
  multiply by *V*",
* ``op * V`` means "multiply by *V*, then apply *op*",
* ``op1 * op2`` and ``op1 @ op2`` mean composition.

Linear boundary value problems and eigenvalue problems can be solved
directly with :meth:`Operator.solve` and :meth:`Operator.eig`.

Examples
--------
>>> import numpy as np
>>> from numgrids import Grid, chebyshev
>>> grid = Grid(x=chebyshev(30, 0, np.pi))
>>> L = grid.diff_operator(2)                     # d^2/dx^2
>>> u = L.solve(-np.sin(grid.coords), bcs=0)      # u'' = -sin(x), u(0) = u(pi) = 0
>>> bool(np.allclose(u, np.sin(grid.coords)))
True
>>> vals, vecs = (-L).eig(k=3, bcs=0)             # -u'' = lambda u
>>> np.round(vals, 6)
array([1., 4., 9.])
"""

from __future__ import annotations

import numbers
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray
from scipy.sparse import spmatrix

if TYPE_CHECKING:
    from numgrids.grids import Grid


class Operator:
    """A linear operator acting on meshed functions of a grid.

    Parameters
    ----------
    grid : Grid
        The grid on which the operator acts.
    matrix : sparse matrix or array
        Square matrix of shape ``(grid.size, grid.size)``.
    """

    __array_ufunc__ = None  # let numpy defer to our reflected operators
    __array_priority__ = 1000

    def __init__(self, grid: Grid, matrix) -> None:
        matrix = scipy.sparse.csr_matrix(matrix)
        if matrix.shape != (grid.size, grid.size):
            raise ValueError(
                f"Matrix shape {matrix.shape} does not match grid size {grid.size}."
            )
        self.grid = grid
        self.matrix = matrix

    # ── construction ────────────────────────────────────────────

    @classmethod
    def identity(cls, grid: Grid) -> Operator:
        """The identity operator on *grid*."""
        return cls(grid, scipy.sparse.identity(grid.size, format="csr"))

    @classmethod
    def multiplication(cls, grid: Grid, values) -> Operator:
        """The operator "multiply by *values*" (a diagonal matrix)."""
        values = np.broadcast_to(np.asarray(values), grid.shape)
        return cls(grid, scipy.sparse.diags(values.ravel()))

    # ── application ─────────────────────────────────────────────

    def __call__(self, f: NDArray) -> NDArray:
        """Apply the operator to a meshed function of shape ``grid.shape``."""
        f = np.asarray(f)
        if f.shape != self.grid.shape:
            raise ValueError(
                f"Array of shape {f.shape} does not match grid shape {self.grid.shape}."
            )
        return (self.matrix @ f.ravel()).reshape(self.grid.shape)

    def as_matrix(self) -> spmatrix:
        """The sparse matrix representation (CSR format)."""
        return self.matrix

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def __repr__(self) -> str:
        return (
            f"Operator(grid={type(self.grid).__name__}{self.grid.shape}, "
            f"nnz={self.matrix.nnz})"
        )

    # ── algebra ─────────────────────────────────────────────────

    def _coerce(self, other: Any) -> spmatrix | None:
        """Convert *other* to a sparse matrix, or ``None`` if impossible."""
        if isinstance(other, Operator):
            self._check_same_grid(other.grid)
            return other.matrix
        if hasattr(other, "as_matrix") and hasattr(other, "grid"):
            self._check_same_grid(other.grid)
            return scipy.sparse.csr_matrix(other.as_matrix())
        if isinstance(other, numbers.Number):
            return other * scipy.sparse.identity(self.grid.size, format="csr")
        if isinstance(other, np.ndarray):
            if other.ndim == 0:
                return other.item() * scipy.sparse.identity(self.grid.size, format="csr")
            if other.shape != self.grid.shape:
                raise ValueError(
                    f"Array of shape {other.shape} does not match grid shape {self.grid.shape}."
                )
            return scipy.sparse.diags(other.ravel())
        return None

    def _check_same_grid(self, grid: Grid) -> None:
        if grid is not self.grid and grid.shape != self.grid.shape:
            raise ValueError("Operators are defined on incompatible grids.")

    def __add__(self, other) -> Operator:
        M = self._coerce(other)
        if M is None:
            return NotImplemented
        return Operator(self.grid, self.matrix + M)

    __radd__ = __add__

    def __sub__(self, other) -> Operator:
        M = self._coerce(other)
        if M is None:
            return NotImplemented
        return Operator(self.grid, self.matrix - M)

    def __rsub__(self, other) -> Operator:
        M = self._coerce(other)
        if M is None:
            return NotImplemented
        return Operator(self.grid, M - self.matrix)

    def __neg__(self) -> Operator:
        return Operator(self.grid, -self.matrix)

    def __pos__(self) -> Operator:
        return self

    def __mul__(self, other) -> Operator:
        # op * other: other acts first
        M = self._coerce(other)
        if M is None:
            return NotImplemented
        if isinstance(other, numbers.Number):
            return Operator(self.grid, self.matrix * other)
        return Operator(self.grid, self.matrix @ M)

    def __rmul__(self, other) -> Operator:
        # other * op: op acts first
        M = self._coerce(other)
        if M is None:
            return NotImplemented
        if isinstance(other, numbers.Number):
            return Operator(self.grid, other * self.matrix)
        return Operator(self.grid, M @ self.matrix)

    def __matmul__(self, other) -> Operator:
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            return NotImplemented
        M = self._coerce(other)
        if M is None:
            return NotImplemented
        return Operator(self.grid, self.matrix @ M)

    def __rmatmul__(self, other) -> Operator:
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            return NotImplemented
        M = self._coerce(other)
        if M is None:
            return NotImplemented
        return Operator(self.grid, M @ self.matrix)

    def __truediv__(self, other) -> Operator:
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return Operator(self.grid, self.matrix / other)

    def __pow__(self, n: int) -> Operator:
        if not isinstance(n, numbers.Integral) or n < 0:
            raise ValueError("Operators can only be raised to non-negative integer powers.")
        result = scipy.sparse.identity(self.grid.size, format="csr")
        for _ in range(n):
            result = result @ self.matrix
        return Operator(self.grid, result)

    # ── solvers ─────────────────────────────────────────────────

    def with_bcs(self, rhs=0.0, bcs=None) -> tuple[spmatrix, NDArray]:
        """Return the linear system ``(L, b)`` with boundary conditions applied.

        Parameters
        ----------
        rhs : float, NDArray or callable
            Right-hand side.  Scalars are broadcast; callables are evaluated
            on the grid via :meth:`~numgrids.grids.Grid.evaluate`.
        bcs : see :func:`~numgrids.boundary.resolve_bcs`
            Boundary conditions.

        Returns
        -------
        tuple of (spmatrix, NDArray)
            System matrix (CSC) and flattened right-hand side.
        """
        from numgrids.boundary import apply_bcs, resolve_bcs

        if callable(rhs):
            rhs = self.grid.evaluate(rhs)
        rhs = np.broadcast_to(np.asarray(rhs), self.grid.shape).ravel().copy()
        if not np.iscomplexobj(rhs):
            rhs = rhs.astype(float)
        L, b = apply_bcs(resolve_bcs(self.grid, bcs), self.matrix, rhs)
        return scipy.sparse.csc_matrix(L), b

    def solve(self, rhs=0.0, bcs=None) -> NDArray:
        """Solve the linear problem ``L u = rhs`` subject to boundary conditions.

        Parameters
        ----------
        rhs : float, NDArray or callable
            Right-hand side.  Scalars are broadcast; callables are called
            with the meshed coordinates (``rhs(*grid.meshed_coords)``).
        bcs : optional
            Boundary conditions.  Accepted forms (see
            :func:`~numgrids.boundary.resolve_bcs`):

            * ``None`` -- no boundary conditions (e.g. fully periodic grids),
            * a scalar or callable -- Dirichlet condition on all faces,
            * a dict mapping face keys (``"0_low"``, ``"x_high"``, ...
              or ``"all"``) to values or to :class:`~numgrids.Dirichlet`,
              :class:`~numgrids.Neumann`, :class:`~numgrids.Robin` specs,
            * a list of bound boundary conditions (``DirichletBC`` etc.).

        Returns
        -------
        NDArray
            The solution, of shape ``grid.shape``.

        Examples
        --------
        >>> import numpy as np
        >>> from numgrids import Grid, Neumann, chebyshev
        >>> grid = Grid(x=chebyshev(20, 0, 1))
        >>> L = grid.diff_operator(2)
        >>> u = L.solve(2, bcs={"x_low": 0, "x_high": Neumann(2)})   # u = x^2
        >>> bool(np.allclose(u, grid.coords ** 2))
        True
        """
        L, b = self.with_bcs(rhs, bcs)
        n_zero = _count_zero_rows(L)
        if n_zero:
            raise ValueError(
                f"The system has {n_zero} empty row(s) not covered by boundary conditions, "
                "so it is singular. This typically happens when grid points lie on a "
                "coordinate singularity (e.g. r = 0); start the axis slightly away from it "
                "or impose a condition there."
            )
        u = scipy.sparse.linalg.spsolve(L, b)
        return np.asarray(u).reshape(self.grid.shape)

    def eig(
        self,
        k: int = 6,
        bcs=None,
        sigma: float | complex | None = None,
        which: str = "SM",
        dense_threshold: int = 2000,
    ) -> tuple[NDArray, NDArray]:
        r"""Solve the eigenvalue problem :math:`L u = \lambda u`.

        Boundary conditions are treated as *homogeneous* (their values are
        ignored).  Dirichlet points are eliminated from the problem, and
        Neumann/Robin rows turn it into a generalized eigenproblem
        :math:`L u = \lambda B u` -- so no spurious eigenvalues appear.

        Parameters
        ----------
        k : int
            Number of eigenpairs to compute (default 6).
        bcs : optional
            Boundary conditions, in any form accepted by :meth:`solve`.
        sigma : float or complex, optional
            If given, return the *k* eigenvalues closest to *sigma*.
        which : str
            Which eigenvalues to return if *sigma* is not given:
            ``"SM"`` smallest magnitude (default), ``"LM"`` largest
            magnitude, ``"SR"`` smallest real part, ``"LR"`` largest
            real part.
        dense_threshold : int
            Problems with at most this many unknowns are solved with dense
            linear algebra (robust); larger ones with sparse shift-invert
            Arnoldi iterations.

        Returns
        -------
        eigenvalues : NDArray
            Shape ``(k,)``, sorted according to *which* / *sigma*.  Real if
            all imaginary parts are negligible.
        eigenvectors : NDArray
            Shape ``(k, *grid.shape)``; ``eigenvectors[i]`` belongs to
            ``eigenvalues[i]``.  Each eigenvector has unit Euclidean norm
            (as a flat vector) and its largest component is positive.
        """
        from numgrids.boundary import DirichletBC, resolve_bcs

        if which not in ("SM", "LM", "SR", "LR"):
            raise ValueError(f"Unknown value for which: {which!r}")

        size = self.grid.size
        bc_list = resolve_bcs(self.grid, bcs)

        A = scipy.sparse.csr_matrix(self.matrix, dtype=np.result_type(self.matrix.dtype, float))
        bc_rows = np.zeros(size, dtype=bool)
        dirichlet = np.zeros(size, dtype=bool)
        for bc in bc_list:
            idx = bc.face.flat_indices
            A, _ = bc.apply_to_system(A, np.zeros(size, dtype=A.dtype))
            A = scipy.sparse.csr_matrix(A)
            bc_rows[idx] = True
            dirichlet[idx] = isinstance(bc, DirichletBC)

        n_zero = _count_zero_rows(A)
        if n_zero:
            warnings.warn(
                f"The operator has {n_zero} empty row(s) not covered by boundary conditions "
                "(e.g. at a coordinate singularity such as r = 0). They produce spurious zero "
                "eigenvalues; start the axis slightly away from the singularity or impose "
                "boundary conditions there.",
                RuntimeWarning,
                stacklevel=2,
            )

        keep = ~dirichlet
        A = A[keep][:, keep]
        B_diag = (~bc_rows[keep]).astype(float)
        generalized = not np.all(B_diag == 1.0)
        n = A.shape[0]
        k = min(k, n - int(np.sum(B_diag == 0)))

        if n <= dense_threshold:
            Ad = A.toarray()
            if generalized:
                vals, vecs = scipy.linalg.eig(Ad, np.diag(B_diag))
                finite = np.isfinite(vals)
                vals, vecs = vals[finite], vecs[:, finite]
            else:
                vals, vecs = scipy.linalg.eig(Ad)
        else:
            M = scipy.sparse.diags(B_diag) if generalized else None
            if sigma is not None or which == "SM":
                shift = 0.0 if sigma is None else sigma
                I_n = M if M is not None else scipy.sparse.identity(n)
                lu = scipy.sparse.linalg.splu(scipy.sparse.csc_matrix(A - shift * I_n))
                dtype = np.result_type(A.dtype, type(shift))
                if M is None:
                    matvec = lu.solve
                else:
                    def matvec(x):
                        return lu.solve(np.asarray(M @ x, dtype=dtype))
                opinv = scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec, dtype=dtype)
                mu, vecs = scipy.sparse.linalg.eigs(opinv, k=k, which="LM")
                vals = shift + 1.0 / mu
            elif generalized:
                raise NotImplementedError(
                    "For large problems with Neumann/Robin conditions, pass `sigma` "
                    "to select the eigenvalues of interest."
                )
            else:
                vals, vecs = scipy.sparse.linalg.eigs(A, k=k, which=which)

        order = self._sort_eigenvalues(vals, which, sigma)[:k]
        vals, vecs = vals[order], vecs[:, order]

        full = np.zeros((size, len(vals)), dtype=vecs.dtype)
        full[keep, :] = vecs
        # Normalize: unit Euclidean norm, largest component real and positive.
        for j in range(full.shape[1]):
            v = full[:, j]
            pivot = v[np.argmax(np.abs(v))]
            full[:, j] = v / np.linalg.norm(v) * (np.abs(pivot) / pivot)
        if np.all(np.abs(vals.imag) <= 1e-10 * np.maximum(1.0, np.abs(vals.real))):
            vals = vals.real
            if np.all(np.abs(full.imag) <= 1e-8):
                full = full.real
        vecs_out = np.moveaxis(full, 1, 0).reshape((len(vals),) + self.grid.shape)
        return vals, vecs_out

    @staticmethod
    def _sort_eigenvalues(vals: NDArray, which: str, sigma) -> NDArray:
        if sigma is not None:
            return np.argsort(np.abs(vals - sigma))
        if which == "SM":
            return np.lexsort((vals.real, np.abs(vals)))
        if which == "LM":
            return np.argsort(-np.abs(vals))
        if which == "SR":
            return np.argsort(vals.real)
        return np.argsort(-vals.real)


def _count_zero_rows(M) -> int:
    """Number of rows of a sparse matrix without nonzero entries."""
    M = scipy.sparse.csr_matrix(M, copy=True)
    M.eliminate_zeros()
    return int(np.sum(np.diff(M.indptr) == 0))

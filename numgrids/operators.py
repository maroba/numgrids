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
directly with :meth:`Operator.solve` and :meth:`Operator.eig`.  Functions
of operators -- fractional powers, square roots, exponentials, inverses --
come from :meth:`Operator.function`, :meth:`Operator.power`,
:meth:`Operator.sqrt`, :meth:`Operator.exp` and :meth:`Operator.inverse`.

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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray
from scipy.sparse import spmatrix

if TYPE_CHECKING:
    from numgrids.grids import Grid
    from numgrids.linalg import LinearSolver


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

    def toarray(self) -> NDArray:
        """The matrix representation as a dense array."""
        return self.matrix.toarray()

    @property
    def T(self) -> Operator:
        """The transposed operator."""
        return Operator(self.grid, self.matrix.T)

    @property
    def H(self) -> Operator:
        """The conjugate transposed operator (adjoint for the Euclidean inner product)."""
        return Operator(self.grid, self.matrix.conj().T)

    def adjoint(self) -> Operator:
        r"""The adjoint with respect to the :math:`L^2` inner product of the grid.

        :math:`\langle f, L g\rangle = \langle L^\dagger f, g\rangle` with
        :math:`\langle f, g\rangle = \int \bar f g\, dV` evaluated by the
        quadrature of the grid (:meth:`Grid.inner <numgrids.grids.Grid.inner>`),
        i.e. :math:`L^\dagger = W^{-1} L^H W` with the diagonal matrix *W*
        of the quadrature weights.  Boundary terms are part of the adjoint:
        operators such as the Laplacian are only self-adjoint on functions
        that satisfy suitable boundary conditions.

        Examples
        --------
        The derivative on a periodic grid is anti-self-adjoint:

        >>> import numpy as np
        >>> from numgrids import Grid, periodic
        >>> D = Grid(x=periodic(16)).diff_operator()
        >>> bool(np.allclose(D.adjoint().toarray(), -D.toarray()))
        True
        """
        w = self.grid.quadrature_weights.ravel()
        W = scipy.sparse.diags(w)
        W_inv = scipy.sparse.diags(1.0 / w)
        return Operator(self.grid, W_inv @ self.matrix.conj().T @ W)

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

    def solve(self, rhs=0.0, bcs=None, method: str = "auto") -> NDArray:
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
        method : str
            How to solve the linear system:

            * ``"fourier"`` -- decouple the system by a Fourier transform
              along the periodic axes on which neither the operator nor the
              boundary conditions depend (e.g. the azimuthal angle of a
              disk or ball), and solve one small system per Fourier mode.
              By far the fastest method where it applies.
            * ``"dense"`` -- dense LU factorization; faster than sparse LU
              for spectral discretizations of moderate size.
            * ``"sparse"`` -- sparse LU factorization; best for finite
              differences.
            * ``"auto"`` (default) -- the first of these that applies.

            To solve with many right-hand sides, use :meth:`factorize`.

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
        return self.factorize(bcs, method=method)(rhs)

    def factorize(self, bcs=None, method: str = "auto") -> LinearSolver:
        """Factorize the linear problem ``L u = rhs`` for repeated solves.

        Returns a :class:`~numgrids.linalg.LinearSolver`: call it with a
        right-hand side to get the solution.  The factorization is computed
        once, so each further solve is cheap -- useful for implicit time
        stepping, iterations and parameter studies.

        Parameters
        ----------
        bcs, method
            As for :meth:`solve`.  The boundary values are fixed.

        Examples
        --------
        >>> import numpy as np
        >>> from numgrids import Grid, chebyshev
        >>> grid = Grid(x=chebyshev(20, 0, np.pi))
        >>> solver = grid.diff_operator(2).factorize(bcs=0)
        >>> x = grid.coords
        >>> bool(np.allclose(solver(-np.sin(x)), np.sin(x)))
        True
        >>> bool(np.allclose(solver(-4 * np.sin(2 * x)), np.sin(2 * x)))
        True
        """
        from numgrids.linalg import LinearSolver
        return LinearSolver(self, bcs, method=method)

    def eig(
        self,
        k: int = 6,
        bcs=None,
        sigma: float | complex | None = None,
        which: str = "SM",
        dense_threshold: int = 2000,
        M=None,
        normalize: str = "euclidean",
        method: str = "auto",
    ) -> tuple[NDArray, NDArray]:
        r"""Solve the eigenvalue problem :math:`L u = \lambda u` or :math:`L u = \lambda M u`.

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
        M : Operator, array, callable or scalar, optional
            Right-hand side operator of the *generalized* eigenvalue problem
            :math:`L u = \lambda M u`, e.g. a mass density or a weight
            function.  An array of grid shape (or a callable evaluated on
            the grid) means multiplication by that function; an
            :class:`Operator` or sparse matrix is used as is.  Defaults to
            the identity.
        normalize : str
            Normalization of the eigenvectors: ``"euclidean"`` (default) --
            unit Euclidean norm of the array of values; ``"l2"`` -- unit
            :math:`L^2` norm on the grid, :math:`\int |u|^2 dV = 1` (see
            :meth:`Grid.norm <numgrids.grids.Grid.norm>`), as for wave
            functions; ``"max"`` -- largest absolute value 1.
        method : str
            ``"auto"`` (default) or ``"direct"``.  With ``"auto"``, problems
            that do not depend on one or more periodic axes (e.g. the
            azimuthal angle of a disk or ball) are decoupled by a Fourier
            transform along these axes into one small eigenproblem per
            Fourier mode, which is much faster and gives all eigenvalues
            to full accuracy.  ``"direct"`` always treats the problem as a
            whole (dense or sparse, see *dense_threshold*).

        Returns
        -------
        eigenvalues : NDArray
            Shape ``(k,)``, sorted according to *which* / *sigma*.  Real if
            all imaginary parts are negligible.
        eigenvectors : NDArray
            Shape ``(k, *grid.shape)``; ``eigenvectors[i]`` belongs to
            ``eigenvalues[i]``.  Each eigenvector is normalized (see
            *normalize*) and its largest component is positive.  For a
            real operator, eigenvectors of real eigenvalues are real, also
            for degenerate eigenvalues (e.g. symmetry-related modes on a
            disk or sphere): a degenerate eigenspace is returned as a real
            orthonormal basis.

        Examples
        --------
        A string with density :math:`\rho = 1/x^2` on :math:`[1, e]`,
        :math:`-u'' = \lambda \rho u`, has the eigenvalues
        :math:`1/4 + n^2 \pi^2`:

        >>> import numpy as np
        >>> from numgrids import Grid, chebyshev
        >>> grid = Grid(x=chebyshev(40, 1, np.e))
        >>> vals, _ = (-grid.diff_operator(2)).eig(k=2, bcs=0, M=1 / grid.coords ** 2)
        >>> bool(np.allclose(vals, 0.25 + np.pi ** 2 * np.array([1, 4])))
        True
        """
        from numgrids.boundary import DirichletBC, resolve_bcs

        if which not in ("SM", "LM", "SR", "LR"):
            raise ValueError(f"Unknown value for which: {which!r}")
        if normalize not in ("euclidean", "l2", "max"):
            raise ValueError(f"Unknown normalization: {normalize!r}. Use 'euclidean', 'l2' or 'max'.")
        if method not in ("auto", "direct"):
            raise ValueError(f"Unknown method: {method!r}. Use 'auto' or 'direct'.")

        size = self.grid.size
        bc_list = resolve_bcs(self.grid, bcs)

        A = scipy.sparse.csr_matrix(self.matrix, dtype=np.result_type(self.matrix.dtype, float))
        B = self._mass_matrix(M)
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
        generalized = M is not None or bool(np.any(bc_rows[keep]))
        blocks = None
        if method == "auto":
            blocks = self._fourier_blocks(A, B)
        if blocks is not None:
            # Right-hand side matrix on the full grid, as below.
            B_full = scipy.sparse.diags((~bc_rows).astype(float))
            if B is not None:
                B_full = B_full @ B
            A_hat = blocks.blocks(A)
            B_hat = blocks.blocks(B_full) if generalized else None
        A = A[keep][:, keep]
        # Right-hand side matrix: M (or the identity), with the rows of the
        # Neumann/Robin conditions set to zero (they hold for every lambda).
        row_mask = scipy.sparse.diags((~bc_rows[keep]).astype(float))
        if B is None:
            B = row_mask
        else:
            B = scipy.sparse.csr_matrix(row_mask @ B[keep][:, keep])
        n = A.shape[0]
        k = min(k, n - int(np.sum(bc_rows[keep])))

        if blocks is not None:
            real = not np.iscomplexobj(A) and not np.iscomplexobj(B)
            vals, vecs = self._eig_fourier(blocks, A_hat, B_hat, keep, k, which, sigma, real)
        elif n <= dense_threshold:
            Ad = A.toarray()
            if generalized:
                vals, vecs = scipy.linalg.eig(Ad, B.toarray())
                finite = np.isfinite(vals)
                vals, vecs = vals[finite], vecs[:, finite]
            else:
                vals, vecs = scipy.linalg.eig(Ad)
        else:
            Bs = scipy.sparse.csr_matrix(B) if generalized else None
            # Compute more eigenpairs than requested: Arnoldi iterations can miss
            # members of (nearly) degenerate eigenvalue clusters at the edge of
            # the requested set. The surplus is discarded after sorting.
            k_arnoldi = min(n - 2, max(2 * k, k + 10))
            # A fixed (pseudo-random) start vector makes the results reproducible.
            v0 = np.random.default_rng(20231117).standard_normal(n)
            if sigma is not None or which == "SM":
                if sigma is None:
                    # Shift slightly off zero: an exact zero eigenvalue (e.g. constants
                    # on a closed surface) would make the shifted matrix singular.
                    shift = -1e-8 * scipy.sparse.linalg.norm(A, 1)
                else:
                    shift = sigma
                I_n = Bs if Bs is not None else scipy.sparse.identity(n)
                lu = scipy.sparse.linalg.splu(scipy.sparse.csc_matrix(A - shift * I_n))
                dtype = np.result_type(A.dtype, type(shift))
                if Bs is None:
                    matvec = lu.solve
                else:
                    def matvec(x):
                        return lu.solve(np.asarray(Bs @ x, dtype=dtype))
                opinv = scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec, dtype=dtype)
                mu, vecs = scipy.sparse.linalg.eigs(opinv, k=k_arnoldi, which="LM", v0=v0)
                vals = shift + 1.0 / mu
            elif generalized:
                raise NotImplementedError(
                    "For large generalized problems (with M or Neumann/Robin conditions), "
                    "pass `sigma` to select the eigenvalues of interest."
                )
            else:
                vals, vecs = scipy.sparse.linalg.eigs(A, k=k_arnoldi, which=which, v0=v0)

        order = self._sort_eigenvalues(vals, which, sigma)[:k]
        vals, vecs = vals[order], vecs[:, order]
        if not np.iscomplexobj(A) and not np.iscomplexobj(B):
            vals, vecs = _real_eigenbasis(vals, vecs)

        full = np.zeros((size, len(vals)), dtype=vecs.dtype)
        full[keep, :] = vecs
        # Normalize, and make the largest component real and positive.
        weights = self.grid.quadrature_weights.ravel() if normalize == "l2" else None
        for j in range(full.shape[1]):
            v = full[:, j]
            pivot = v[np.argmax(np.abs(v))]
            if normalize == "l2":
                norm = np.sqrt(abs(np.sum(weights * np.abs(v) ** 2)))
            elif normalize == "max":
                norm = np.abs(pivot)
            else:
                norm = np.linalg.norm(v)
            full[:, j] = v / norm * (np.abs(pivot) / pivot)
        if np.all(np.abs(vals.imag) <= 1e-10 * np.maximum(1.0, np.abs(vals.real))):
            vals = vals.real
            if np.all(np.abs(full.imag) <= 1e-8):
                full = full.real
        vecs_out = np.moveaxis(full, 1, 0).reshape((len(vals),) + self.grid.shape)
        return vals, vecs_out

    def _fourier_blocks(self, A, B):
        """Fourier decoupling of the eigenproblem, or ``None`` if it does not apply or pay off."""
        from numgrids.linalg import _FOURIER_MAX_BYTES, _FOURIER_MAX_WORK, FourierBlocks, shift_invariant_axes

        axes = shift_invariant_axes(A, self.grid)
        if B is not None and axes:
            invariant_B = shift_invariant_axes(B, self.grid)
            axes = [a for a in axes if a in invariant_B]
        if not axes:
            return None
        blocks = FourierBlocks(self.grid, axes)
        # Dense eigensolvers need about 25 n^3 operations.
        if blocks.nbytes() > _FOURIER_MAX_BYTES or 25 * blocks.work() > _FOURIER_MAX_WORK:
            return None
        return blocks

    def _eig_fourier(self, blocks, A_hat, B_hat, keep, k, which, sigma, real):
        """Eigenpairs from one small dense eigenproblem per Fourier mode."""
        keep_rest = keep[blocks.index[:, 0]]
        all_vals, all_vecs, all_modes = [], [], []
        done = {}
        for m in range(blocks.n_modes):
            conj = blocks.conjugate_mode(m)
            if real and conj in done:
                # A real problem has complex conjugate blocks for opposite wave numbers.
                vals, vecs = done[conj]
                vals, vecs = vals.conj(), vecs.conj()
            else:
                Am = A_hat[m][keep_rest][:, keep_rest]
                if B_hat is not None:
                    vals, vecs = scipy.linalg.eig(Am, B_hat[m][keep_rest][:, keep_rest])
                    finite = np.isfinite(vals)
                    vals, vecs = vals[finite], vecs[:, finite]
                else:
                    vals, vecs = scipy.linalg.eig(Am)
                done[m] = (vals, vecs)
            all_vals.append(vals)
            all_vecs.append(vecs)
            all_modes.append(np.full(len(vals), m))
        vals = np.concatenate(all_vals)
        modes = np.concatenate(all_modes)
        offsets = np.concatenate([[0], np.cumsum([len(v) for v in all_vals])])
        order = self._sort_eigenvalues(vals, which, sigma)[:k]
        index = blocks.index[keep_rest]
        # Position of the kept grid points in the reduced (Dirichlet points removed) system.
        position = np.cumsum(keep) - 1
        vecs = np.zeros((int(np.sum(keep)), len(order)), dtype=complex)
        for col, i in enumerate(order):
            m = modes[i]
            v = all_vecs[m][:, i - offsets[m]]
            vecs[position[index], col] = np.outer(v, blocks.mode(m))
        return vals[order], vecs

    def _mass_matrix(self, M):
        """Sparse matrix of the right-hand side operator of a generalized eigenproblem."""
        if M is None:
            return None
        if callable(M) and not isinstance(M, Operator) and not hasattr(M, "as_matrix"):
            M = self.grid.evaluate(M)
        if scipy.sparse.issparse(M):
            matrix = scipy.sparse.csr_matrix(M)
            if matrix.shape != self.matrix.shape:
                raise ValueError(f"M has shape {matrix.shape}, expected {self.matrix.shape}.")
        elif isinstance(M, np.ndarray) and M.ndim == 2 and M.shape == self.matrix.shape \
                and M.shape != self.grid.shape:
            matrix = scipy.sparse.csr_matrix(M)
        else:
            matrix = Operator.identity(self.grid)._coerce(M)
            if matrix is None:
                raise TypeError(f"Cannot use an object of type {type(M).__name__} as M.")
        return scipy.sparse.csr_matrix(matrix, dtype=np.result_type(matrix.dtype, float))

    # ── functions of the operator ───────────────────────────────

    def function(self, f: Callable[[NDArray], NDArray], bcs=None, *, max_size: int | None = None) -> Operator:
        r"""The operator :math:`f(L)` for a scalar function *f*.

        :math:`f(L)` has the eigenfunctions of :math:`L` and the
        eigenvalues :math:`f(\lambda_n)`: with the eigenpairs
        :math:`L \varphi_n = \lambda_n \varphi_n` subject to the boundary
        conditions, it maps :math:`u = \sum_n c_n \varphi_n` to
        :math:`\sum_n f(\lambda_n)\, c_n \varphi_n`.  See also
        :meth:`power`, :meth:`sqrt`, :meth:`exp` and :meth:`inverse`.

        Parameters
        ----------
        f : callable
            The function.  It is called with a one-dimensional array of
            eigenvalues (a real array if all eigenvalues are real, a
            complex one otherwise) and returns the array of its values,
            which may be complex.
        bcs : optional
            Boundary conditions, in any form accepted by :meth:`solve`.
            They are treated as *homogeneous* (their values are ignored),
            as in :meth:`eig`.  Not needed on periodic grids.
        max_size : int, optional
            Largest size of the dense matrices to work with (default 4000);
            larger problems raise an error.

        Returns
        -------
        Operator
            :math:`f(L)` on the functions that satisfy the boundary
            conditions.  Applied to such a function, it gives
            :math:`f(L) u` at the points without a condition and sets the
            boundary values such that the result satisfies the conditions:
            to zero for Dirichlet conditions, and for Neumann and Robin
            conditions to the values that follow from the condition.  The
            values of the argument on the boundary points are not used
            (the matrix has zero columns there), since for a function that
            satisfies the conditions they follow from the other ones.

            So ``L.function(lambda z: z, bcs)`` agrees with *L* on such
            functions except on the boundary points, where :math:`L u`
            does not satisfy the conditions in general, and
            ``L.function(np.ones_like, bcs)`` is the projection that
            overwrites the boundary values such that the conditions hold.
            The results obey the functional calculus,
            ``L.function(f, bcs) @ L.function(g, bcs)`` is
            ``L.function(lambda z: f(z) * g(z), bcs)``.  To use the result
            in :meth:`solve`, :meth:`eig` or :func:`~numgrids.evolve`,
            pass the same boundary conditions there.

        Notes
        -----
        The computation needs dense matrices.  If the operator and the
        boundary conditions are shift invariant along periodic axes --
        operators with constant coefficients on periodic grids, or the
        angles of disks, cylinders, spheres and balls -- a Fourier
        transform along these axes leaves one small matrix per Fourier
        mode, and the result is a :class:`~numgrids.funm.FourierOperator`
        that is applied with fast Fourier transforms.  Otherwise the
        matrix has the size of the grid.

        The function of a matrix is evaluated with its eigendecomposition
        as long as the eigenvectors are well conditioned (1-norm condition
        number up to :math:`10^5`), as for Laplacians, diffusion and
        Schrödinger operators.  Operators further from normal, such as
        convection-dominated ones, are handled by the Schur-Parlett method
        :func:`scipy.linalg.funm`, which is less accurate for close
        eigenvalues; a warning tells if the estimated error is large.  The
        named functions (:meth:`power` etc.) use dedicated algorithms then.

        Examples
        --------
        A low-pass filter on a periodic grid, keeping the wave numbers
        :math:`|k| \le 4` (the eigenvalues of :math:`-d^2/dx^2` are
        :math:`k^2`):

        >>> import numpy as np
        >>> from numgrids import Grid, periodic
        >>> grid = Grid(x=periodic(64))
        >>> x = grid.coords
        >>> lowpass = (-grid.laplacian_operator()).function(lambda lam: lam <= 16)
        >>> bool(np.allclose(lowpass(np.sin(2 * x) + np.sin(9 * x)), np.sin(2 * x)))
        True
        """
        from numgrids import funm
        return funm._function(self, f, bcs, max_size)

    def power(self, s: float, bcs=None, *, max_size: int | None = None) -> Operator:
        r"""The power :math:`L^s` for a real exponent *s*.

        ``(-grid.laplacian_operator()).power(s, bcs=0)`` is the (spectral)
        fractional Laplacian :math:`(-\Delta)^s` with Dirichlet conditions.
        Negative exponents give powers of the inverse; where the operator
        has zero eigenvalues, such as the Laplacian on a periodic grid or
        with Neumann conditions on the whole boundary, the pseudo-inverse
        takes its place: the zero modes are mapped to zero, so that e.g.
        ``(-L).power(-1)`` solves :math:`-\Delta u = f` for the part of
        :math:`f` with zero mean and returns the solution with zero mean.

        Parameters
        ----------
        s : float
            The exponent.  For a non-integer exponent the operator must not
            have negative eigenvalues: take the power of :math:`-L` for a
            negative operator such as the Laplacian.
        bcs, max_size
            As for :meth:`function`, which also describes the result and
            the numerical methods.  If the eigenvectors are ill
            conditioned, :func:`scipy.linalg.fractional_matrix_power` is
            used.

        Examples
        --------
        The fractional Poisson problem :math:`(-d^2/dx^2)^{1/2} u = f`
        on :math:`[0, \pi]` with :math:`u = 0` at both ends:

        >>> import numpy as np
        >>> from numgrids import Grid, chebyshev
        >>> grid = Grid(x=chebyshev(30, 0, np.pi))
        >>> x = grid.coords
        >>> L = -grid.laplacian_operator()
        >>> half = L.power(0.5, bcs=0)
        >>> bool(np.allclose(half(np.sin(3 * x)), 3 * np.sin(3 * x)))
        True
        >>> u = L.power(-0.5, bcs=0)(np.sin(3 * x))        # solves the problem ...
        >>> bool(np.allclose(u, np.sin(3 * x) / 3))
        True
        >>> u = half.solve(np.sin(3 * x), bcs=0)           # ... and so does this
        >>> bool(np.allclose(u, np.sin(3 * x) / 3))
        True
        """
        from numgrids import funm
        return funm._power(self, s, bcs, max_size)

    def sqrt(self, bcs=None, *, max_size: int | None = None) -> Operator:
        r"""The square root :math:`L^{1/2}`: ``L.sqrt(bcs) @ L.sqrt(bcs)`` acts as *L*.

        Same as ``power(0.5, bcs)``; :func:`scipy.linalg.sqrtm` is used if
        the eigenvectors are ill conditioned.

        Examples
        --------
        On a periodic grid, :math:`(-d^2/dx^2)^{1/2}` multiplies the
        Fourier mode :math:`e^{ikx}` by :math:`|k|`:

        >>> import numpy as np
        >>> from numgrids import Grid, periodic
        >>> grid = Grid(x=periodic(32))
        >>> x = grid.coords
        >>> root = (-grid.laplacian_operator()).sqrt()
        >>> bool(np.allclose(root(np.cos(5 * x)), 5 * np.cos(5 * x)))
        True
        """
        from numgrids import funm
        return funm._power(self, 0.5, bcs, max_size)

    def exp(self, t: float | complex = 1.0, bcs=None, *, max_size: int | None = None) -> Operator:
        r"""The exponential :math:`e^{tL}`.

        :math:`u(t) = e^{tL} u_0` solves :math:`\partial_t u = L u` with
        :math:`u(0) = u_0` and homogeneous boundary conditions, so the
        result is the propagator over the time *t*.  A complex *t* gives
        e.g. the propagator :math:`e^{-itH}` of the Schrödinger equation
        as ``H.exp(-1j * t, bcs)``.

        Parameters
        ----------
        t : float or complex
            The factor in the exponent (default 1).
        bcs, max_size
            As for :meth:`function`, which also describes the result and
            the numerical methods.  If the eigenvectors are ill
            conditioned, :func:`scipy.linalg.expm` is used.

        Examples
        --------
        Heat conduction in a rod with fixed ends:

        >>> import numpy as np
        >>> from numgrids import Grid, chebyshev
        >>> grid = Grid(x=chebyshev(30, 0, 1))
        >>> x = grid.coords
        >>> propagator = grid.laplacian_operator().exp(0.1, bcs=0)
        >>> u = propagator(np.sin(np.pi * x))
        >>> bool(np.allclose(u, np.exp(-0.1 * np.pi ** 2) * np.sin(np.pi * x)))
        True
        """
        from numgrids import funm
        return funm._exp(self, t, bcs, max_size)

    def inverse(self, bcs=None, *, max_size: int | None = None) -> Operator:
        r"""The inverse :math:`L^{-1}` with homogeneous boundary conditions (Green's operator).

        ``L.inverse(bcs)(f)`` is the solution of :math:`L u = f` that
        satisfies the homogeneous conditions, like ``L.solve(f, bcs)`` with
        zero boundary values.  Unlike a factorization
        (:meth:`factorize`), the inverse is an operator: it can be
        combined with others, and its matrix holds the discrete Green's
        function.  Same as ``power(-1, bcs)``, including the treatment of
        zero eigenvalues.

        Examples
        --------
        >>> import numpy as np
        >>> from numgrids import Grid, chebyshev
        >>> grid = Grid(x=chebyshev(30, 0, 1))
        >>> x = grid.coords
        >>> G = grid.laplacian_operator().inverse(bcs=0)
        >>> bool(np.allclose(G(np.ones(grid.shape)), x * (x - 1) / 2))     # u'' = 1
        True
        """
        from numgrids import funm
        return funm._power(self, -1, bcs, max_size)

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


def _real_eigenbasis(vals: NDArray, vecs: NDArray, rtol: float = 1e-8) -> tuple[NDArray, NDArray]:
    r"""Replace complex eigenvectors of a real problem by real ones.

    For a real matrix, the eigenspace of a real eigenvalue has a real basis,
    but eigensolvers may return complex vectors for it.  This happens when
    the eigenvalue is degenerate: any complex combination of the basis is an
    eigenvector, and a nonsymmetric discretization may even split the
    eigenvalue into a pair :math:`\lambda \pm i\epsilon` with complex
    conjugate eigenvectors.  For each cluster of (numerically) equal, real
    eigenvalues, the real and imaginary parts of its eigenvectors span the
    real eigenspace; an orthonormal basis of that span replaces the cluster's
    eigenvectors (also if they were real but not orthogonal).  Genuinely
    complex eigenvalues are left unchanged.
    """
    vals = np.array(vals, dtype=complex)
    vecs = np.array(vecs, dtype=complex)
    scale = rtol * np.maximum(1.0, np.abs(vals))
    real_valued = np.abs(vals.imag) <= scale
    unassigned = set(np.flatnonzero(real_valued))
    while unassigned:
        i = min(unassigned)
        cluster = [j for j in sorted(unassigned)
                   if abs(vals[j].real - vals[i].real) <= max(scale[i], scale[j])]
        unassigned.difference_update(cluster)
        V = vecs[:, cluster]
        m = len(cluster)
        vals[cluster] = vals[cluster].real
        if m == 1 and not np.any(V.imag):
            continue
        U, s, _ = np.linalg.svd(np.hstack([V.real, V.imag]), full_matrices=False)
        if s[m - 1] <= 1e-6 * s[0]:
            # The vectors do not span an m-dimensional space; leave them alone.
            continue
        vecs[:, cluster] = U[:, :m]
    return vals, vecs


def _count_zero_rows(M) -> int:
    """Number of rows of a sparse matrix without nonzero entries."""
    M = scipy.sparse.csr_matrix(M, copy=True)
    M.eliminate_zeros()
    return int(np.sum(np.diff(M.indptr) == 0))

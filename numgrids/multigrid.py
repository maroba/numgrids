r"""Geometric multigrid solver for linear boundary value problems.

:class:`MultiGridSolver` solves :math:`L u = f` with boundary conditions on
the finest grid of a :class:`~numgrids.grids.MultiGrid`.  Memory and run
time grow only linearly with the number of grid points, so that large
finite-difference problems -- in particular in 3D, where the sparse LU
factorization behind :meth:`Operator.solve
<numgrids.operators.Operator.solve>` runs out of memory -- stay feasible.

The operator is given as a function that builds it on a grid
(*rediscretization*): it is called for every level of the hierarchy.

Examples
--------
Poisson's equation :math:`\nabla^2 u = f` on the unit square with
:math:`u = 0` on the boundary and the exact solution
:math:`u = \sin(\pi x) \sin(\pi y)`:

>>> import numpy as np
>>> from numgrids import MultiGrid, equidistant
>>> mg = MultiGrid(x=equidistant(65, 0, 1), y=equidistant(65, 0, 1))
>>> f = lambda x, y: -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
>>> u = mg.solve(lambda grid: grid.laplacian_operator(), f, bcs=0)
>>> X, Y = mg.levels[0].meshed_coords
>>> bool(np.abs(u - np.sin(np.pi * X) * np.sin(np.pi * Y)).max() < 1e-6)
True

For several right-hand sides, set the solver up once and reuse it:

>>> solver = mg.solver(lambda grid: grid.laplacian_operator(), bcs=0)
>>> u, info = solver.solve(f, return_info=True)
>>> info.converged, info.iterations < 30
(True, True)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from numgrids.axes import Axis, EquidistantAxis, LogAxis
from numgrids.grids import Grid

if TYPE_CHECKING:
    from numgrids.boundary import BoundaryCondition
    from numgrids.grids import MultiGrid
    from numgrids.operators import Operator

# Accuracy order of the finite-difference stencils in the low-order hierarchy.
_LOW_ORDER = 2
# Levels with fewer points along some axis are not used: the stencils no longer fit
# (second order / the default fourth order of the library).
_MIN_POINTS_LOW_ORDER = 5
_MIN_POINTS = 8
# Relaxation weight of the Jacobi smoother; close to the optimum in 1D, 2D and 3D.
_OMEGA = 0.8
# Extra smoothing next to faces with Neumann or Robin conditions: number of
# sweeps per regular sweep, and how many grid points from the face they reach.
_BOUNDARY_SWEEPS = 3
_BOUNDARY_WIDTH = 4
# GMRES restarts after at most this many iterations, earlier if the Krylov
# vectors would need more than the given amount of memory.
_MAX_RESTART = 50
_MIN_RESTART = 10
_KRYLOV_BYTES = 2 ** 30

CYCLES = ("V", "W")
METHODS = ("gmres", "plain")


class MultiGridError(RuntimeError):
    """Raised when :class:`MultiGridSolver` does not converge.

    Attributes
    ----------
    u : NDArray
        The last iterate.
    residuals : list of float
        Relative residual norm after each iteration (see :class:`MultiGridInfo`).
    """

    def __init__(self, message: str, u: NDArray, residuals: list[float]) -> None:
        super().__init__(message)
        self.u = u
        self.residuals = residuals


@dataclass(frozen=True)
class MultiGridInfo:
    r"""Convergence record of a multigrid solve.

    Attributes
    ----------
    iterations : int
        Number of iterations; each one costs one multigrid cycle.
    residuals : list of float
        Relative residual norm :math:`\|b - A u\|_2 / \|b\|_2` of the linear
        system after each iteration; ``residuals[0]`` belongs to the initial
        guess.  Here :math:`A u = b` is the system for the interior values:
        known boundary values count as part of the right-hand side :math:`b`.
    converged : bool
        Whether the tolerance was reached.
    shapes : tuple of tuple of int
        Grid shapes of the levels used, finest first.
    """

    iterations: int
    residuals: list[float]
    converged: bool
    shapes: tuple[tuple[int, ...], ...]

    @property
    def convergence_factor(self) -> float:
        """Average factor by which one iteration reduces the residual."""
        if self.iterations == 0 or self.residuals[0] == 0:
            return 0.0
        return float((self.residuals[-1] / self.residuals[0]) ** (1.0 / self.iterations))


class MultiGridSolver:
    r"""Geometric multigrid solver for :math:`L u = f` with boundary conditions.

    The solver is set up once for an operator and a set of boundary
    conditions and can then solve for any number of right-hand sides with
    :meth:`solve`.  For a single solve, :meth:`MultiGrid.solve
    <numgrids.grids.MultiGrid.solve>` is more convenient.

    One multigrid cycle smooths the error with damped Jacobi sweeps, restricts
    the residual to the next coarser grid (full weighting), solves there
    recursively -- by sparse LU on the coarsest grid --, interpolates the
    correction back (linearly) and smooths again.  By default the cycle is
    not iterated on its own but preconditions GMRES.

    Parameters
    ----------
    multigrid : MultiGrid
        The grid hierarchy.  The problem is solved on its finest grid; coarser
        levels are used as long as every axis has at least 5 points (8 for
        ``low_order=False``); a larger ``min_size`` of the
        :class:`~numgrids.grids.MultiGrid` stops the coarsening earlier.
        All axes must be equidistant (non-periodic) or logarithmic, because
        only finite differences give sparse operators.
    operator : callable
        ``operator(grid)`` returns the operator :math:`L` on the given grid as
        an :class:`~numgrids.operators.Operator`, e.g. ``lambda grid:
        grid.laplacian_operator()``.  It is called once per level (for
        ``low_order=True`` once more, see below), so coefficients must be
        built from the grid passed in, e.g. with
        :meth:`grid.evaluate <numgrids.grids.Grid.evaluate>`.
    bcs : optional
        Boundary conditions in any form accepted by
        :func:`~numgrids.boundary.resolve_bcs`.  Every boundary face needs a
        condition.  Values may be scalars, callables or arrays on the faces
        of the finest grid (coarser levels only need the type of the
        conditions).
    pre, post : int
        Number of smoothing sweeps before and after the coarse-grid
        correction (default 2 each).
    cycle : str
        ``"V"`` (default) or ``"W"``: one or two coarse-grid corrections per
        level.
    low_order : bool
        If true (default), the multigrid cycle is built from second-order
        finite-difference stencils, whatever accuracy order the operator
        asks for; the cycle then only *preconditions* the iteration for the
        operator as given, whose solution is the one returned.  The compact
        second-order stencils are what a point smoother needs; the one-sided
        stencils that higher orders use next to the boundary (already for
        the default ``acc=4``) make it diverge.  To this end, *operator* is
        also called with grids whose finite differences are limited to
        second order.  With ``low_order=False`` the cycle uses the operator
        exactly as given on all levels -- sensible only if it is of second
        order anyway.

    Raises
    ------
    TypeError
        If an axis is spectral (Chebyshev, periodic, ...) or *operator* is
        not a function of the grid.
    ValueError
        If a boundary face has no condition, the operator cannot be built
        on a level, or the problem is singular (e.g. only Neumann
        conditions).

    Notes
    -----
    The number of iterations is independent of the grid size for elliptic
    operators with comparable grid spacing in all directions: Laplacians,
    variable coefficients and lower-order terms included.  Operators of
    second order (``acc=2``) need about 10 iterations; the default
    fourth-order operators need 20 to 50, because the second-order cycle
    approximates their one-sided stencils next to the boundary less well.

    Point smoothers do not cope with strongly anisotropic problems (grid
    spacings or coefficients that differ by more than a factor of about 4
    between directions), with indefinite problems (Helmholtz) or with
    second derivatives written as products of first derivatives; then many
    more iterations are needed or the solver fails with a
    :class:`MultiGridError`.

    Examples
    --------
    A variable-coefficient problem :math:`-\nabla^2 u + (1 + x^2) u = 1`
    with mixed boundary conditions:

    >>> from numgrids import MultiGrid, Neumann, equidistant
    >>> from numgrids.multigrid import MultiGridSolver
    >>> mg = MultiGrid(x=equidistant(33, 0, 1), y=equidistant(33, 0, 1))
    >>> def operator(grid):
    ...     c = grid.evaluate(lambda x, y: 1 + x ** 2)
    ...     return -grid.laplacian_operator() + c
    >>> solver = MultiGridSolver(mg, operator, bcs={"all": 0, "x_high": Neumann(0)})
    >>> u = solver.solve(1.0)
    >>> u.shape
    (33, 33)
    """

    def __init__(
        self,
        multigrid: MultiGrid,
        operator: Callable[[Grid], Operator],
        bcs=None,
        *,
        pre: int = 2,
        post: int = 2,
        cycle: str = "V",
        low_order: bool = True,
    ) -> None:
        from numgrids.boundary import apply_bcs, resolve_bcs

        if not callable(operator) or hasattr(operator, "as_matrix"):
            raise TypeError(
                "`operator` must be a function that builds the operator on a given grid, "
                "e.g. `lambda grid: grid.laplacian_operator()`, so that it can be built on every level."
            )
        if cycle not in CYCLES:
            raise ValueError(f"Unknown cycle {cycle!r}. Use one of {CYCLES}.")
        if pre < 0 or post < 0 or pre + post == 0:
            raise ValueError("`pre` and `post` must be non-negative and not both zero.")

        min_points = _MIN_POINTS_LOW_ORDER if low_order else _MIN_POINTS
        grids = [g for g in multigrid.levels if min(g.shape) >= min_points] or multigrid.levels[:1]
        fine = grids[0]
        _check_axes(fine)
        self.grid = fine
        self.shapes = tuple(g.shape for g in grids)
        self._pre, self._post = int(pre), int(post)
        self._gamma = 1 if cycle == "V" else 2

        # The system to solve: the operator as given, on the finest grid.
        bc_list = resolve_bcs(fine, bcs)
        for bc in bc_list:
            if bc.face.grid.shape != fine.shape:
                raise ValueError(
                    f"A boundary condition is bound to a grid of shape {bc.face.grid.shape}, "
                    f"but the finest grid has shape {fine.shape}."
                )
        bc_rows = _bc_mask(fine, bc_list)
        if not np.array_equal(bc_rows, fine.boundary.ravel()):
            raise ValueError(
                "Every boundary face needs a boundary condition; "
                f"got conditions for {sorted({bc.face.key for bc in bc_list})} "
                f"out of {list(fine.faces)}."
            )
        M = _matrix(operator(fine), fine)
        # Boundary rows R and values g, as apply_bcs would put them into the system.
        R, g = apply_bcs(bc_list, scipy.sparse.csr_matrix((fine.size, fine.size)), np.zeros(fine.size))
        A = _with_rows(M, bc_rows, R)
        del M, R
        # Internally the unknowns are ordered [interior, boundary] on all levels.
        self._perm = _interior_first(bc_rows)
        self._n_free = fine.size - int(bc_rows.sum())
        self._matrix = A[self._perm][:, self._perm]
        self._boundary = _BoundaryRows(self._matrix, self._n_free)
        self._bc_values = np.asarray(g)[bc_rows]

        # The hierarchy for the multigrid cycle.
        self._levels: list[_Level] = []
        if len(grids) > 1:
            for k, grid in enumerate(grids):
                if low_order:
                    A = _level_matrix(operator, _low_order_grid(grid), bc_list)
                elif k > 0:
                    A = _level_matrix(operator, grid, bc_list)
                self._levels.append(_Level(A, grid.boundary.ravel(), _near_coupled_faces(grid, bc_list),
                                           coarsest=k == len(grids) - 1))
            for k in range(len(grids) - 1):
                self._levels[k].set_transfer(_prolongation(grids[k + 1], grids[k]), self._levels[k + 1])
        del A
        self._direct = None

    def __repr__(self) -> str:
        return f"MultiGridSolver(levels={list(self.shapes)})"

    # ── solving ─────────────────────────────────────────────────

    def solve(
        self,
        rhs=0.0,
        *,
        u0=None,
        tol: float = 1e-8,
        maxiter: int = 100,
        method: str = "gmres",
        return_info: bool = False,
    ) -> NDArray | tuple[NDArray, MultiGridInfo]:
        r"""Solve :math:`L u = \text{rhs}` subject to the boundary conditions.

        Parameters
        ----------
        rhs : float, NDArray or callable
            Right-hand side on the finest grid.  Scalars are broadcast;
            callables are called with the meshed coordinates.
        u0 : float, NDArray or callable, optional
            Initial guess (default zero).  Its boundary values are replaced
            by those that satisfy the boundary conditions.
        tol : float
            Stop when the residual norm of the linear system falls below
            *tol* times the norm of its right-hand side.  Known boundary
            values count as part of the right-hand side (see
            :class:`MultiGridInfo`), so that the tolerance means the same
            for problems driven by boundary values as for those driven by
            *rhs*.
        maxiter : int
            Maximum number of iterations (multigrid cycles).
        method : str
            ``"gmres"`` (default): GMRES with one multigrid cycle as
            preconditioner -- robust, also for higher-order operators and
            Neumann or Robin conditions.  ``"plain"``: repeat the multigrid
            cycle on its own; about as fast for second-order operators with
            Dirichlet conditions, otherwise much slower or divergent.
        return_info : bool
            Also return a :class:`MultiGridInfo` with the number of
            iterations and the residual history.

        Returns
        -------
        NDArray or tuple of (NDArray, MultiGridInfo)
            The solution, of shape ``grid.shape``.

        Raises
        ------
        MultiGridError
            If the tolerance is not reached within *maxiter* iterations.
            The exception holds the last iterate and the residual history.
        """
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}. Use one of {METHODS}.")
        grid, n = self.grid, self._n_free
        b = self._field(rhs, "rhs")
        dtype = np.result_type(b, self._bc_values, self._matrix.dtype)
        b = b.astype(dtype)
        b[n:] = self._bc_values

        # The right-hand side of the equations for the interior values: what
        # remains of b when the boundary conditions are satisfied and the
        # interior is zero.  Its norm is the reference for the tolerance.
        lifted = np.zeros(grid.size, dtype=dtype)
        self._boundary.constrain(lifted, b)
        reference = float(np.linalg.norm(b - self._matrix @ lifted))
        if u0 is None:
            x = lifted
        else:
            x = self._field(u0, "u0").astype(dtype)
            self._boundary.constrain(x, b)

        if reference == 0:
            # Nothing drives the interior: zero with the boundary values is the solution.
            x, residuals, converged = lifted, [0.0], True
        elif not self._levels:
            x, residuals, converged = self._solve_direct(b, x, reference)
        elif method == "gmres":
            restart = int(np.clip(_KRYLOV_BYTES // b.nbytes, _MIN_RESTART, _MAX_RESTART))
            x, residuals, converged = _gmres(self._apply, self._precondition, b, x, reference, tol, maxiter, restart)
        else:
            x, residuals, converged = _iterate(self._apply, self._precondition, b, x, reference, tol, maxiter)

        u = np.empty_like(x)
        u[self._perm] = x
        u = u.reshape(grid.shape)
        if not converged:
            raise MultiGridError(
                f"Multigrid did not converge in {len(residuals) - 1} iterations "
                f"(relative residual {residuals[-1]:.3e}, tolerance {tol:.1e}). "
                + ("The residual stagnates at a small value: the tolerance is probably below what "
                   "rounding errors permit for this problem."
                   if residuals[-1] < 1e-6 else
                   "The problem may be strongly anisotropic, indefinite or singular; "
                   "see the notes in the documentation of MultiGridSolver."),
                u,
                residuals,
            )
        if return_info:
            return u, MultiGridInfo(len(residuals) - 1, residuals, True, self.shapes)
        return u

    def as_preconditioner(self) -> scipy.sparse.linalg.LinearOperator:
        """One multigrid cycle as a preconditioner for SciPy's iterative solvers.

        The returned :class:`~scipy.sparse.linalg.LinearOperator` maps a
        flattened residual to an approximate solution of the system
        ``operator(grid).with_bcs(rhs, bcs)`` and can be passed as ``M`` to
        e.g. :func:`scipy.sparse.linalg.bicgstab`.
        """
        if not self._levels:
            self._factorize()
        apply = self._precondition if self._levels else self._direct
        perm = self._perm

        def matvec(r: NDArray) -> NDArray:
            z = np.empty_like(r)
            z[perm] = apply(r[perm])
            return z

        n = self.grid.size
        return scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec, dtype=self._matrix.dtype)

    # ── internals ───────────────────────────────────────────────

    def _field(self, values, name: str) -> NDArray:
        """A scalar, array or callable as a vector in the internal ordering."""
        grid = self.grid
        if callable(values):
            values = grid.evaluate(values)
        values = np.asarray(values)
        if values.ndim and values.shape != grid.shape:
            raise ValueError(
                f"Array '{name}' has shape {values.shape}, which does not match "
                f"the shape {grid.shape} of the finest grid."
            )
        values = np.broadcast_to(values, grid.shape).ravel()[self._perm]
        return np.array(values, dtype=np.result_type(values, float))

    def _apply(self, x: NDArray) -> NDArray:
        return self._matrix @ x

    def _precondition(self, r: NDArray) -> NDArray:
        """One multigrid cycle for the residual *r*, starting from zero."""
        return self._cycle(0, r)

    def _cycle(self, k: int, f: NDArray) -> NDArray:
        level = self._levels[k]
        if level.coarsest:
            return level.solve(f)
        coarse = self._levels[k + 1]
        u = np.zeros_like(f)
        level.boundary.constrain(u, f)
        level.smooth(u, f, self._pre)
        for _ in range(1 if coarse.coarsest else self._gamma):
            # The boundary rows are satisfied exactly, so only interior residuals
            # are restricted; the coarse correction has homogeneous conditions.
            fc = np.zeros(coarse.size, dtype=f.dtype)
            fc[:coarse.n_free] = level.restrict @ level.residual(u, f)
            u[:level.n_free] += level.prolong @ self._cycle(k + 1, fc)
            level.boundary.update(u, f)
        level.smooth(u, f, self._post)
        return u

    def _factorize(self) -> None:
        if self._direct is None:
            self._direct = _factorized(self._matrix)

    def _solve_direct(self, b: NDArray, x: NDArray, reference: float):
        """Too small to coarsen: solve directly."""
        self._factorize()
        r0 = float(np.linalg.norm(b - self._matrix @ x) / reference)
        x = self._direct(b)
        r1 = float(np.linalg.norm(b - self._matrix @ x) / reference)
        # A direct solve is as good as it gets, whatever the tolerance.
        return x, [r0, r1], bool(np.isfinite(r1))


# ---------------------------------------------------------------------------
# One level of the hierarchy
# ---------------------------------------------------------------------------

def _interior_first(bc_rows: NDArray) -> NDArray:
    """Permutation that orders the unknowns ``[interior, boundary]``."""
    return np.concatenate([np.flatnonzero(~bc_rows), np.flatnonzero(bc_rows)])


class _BoundaryRows:
    """The boundary rows of a system whose unknowns are ordered ``[interior, boundary]``.

    These rows hold the boundary conditions.  They are no differential
    equations and are not smoothed; instead the boundary values are computed
    from them exactly, given the interior values (which is trivial for
    Dirichlet conditions).
    """

    def __init__(self, A, n_free: int) -> None:
        rows = A[n_free:]
        self.n_free = n_free
        self.coupling = scipy.sparse.csr_matrix(rows[:, :n_free])
        block = scipy.sparse.csc_matrix(rows[:, n_free:])
        self.diagonal = block.diagonal()
        # Dirichlet rows do not involve interior values.
        self.coupled = self.coupling.nnz > 0
        identity = scipy.sparse.identity(block.shape[0], format="csc")
        self._solve = None if (block != identity).nnz == 0 else _factorized(block)

    def constrain(self, u: NDArray, f: NDArray) -> None:
        """Set the boundary values of *u* such that the boundary rows hold."""
        n = self.n_free
        g = f[n:] - self.coupling @ u[:n] if self.coupled else f[n:]
        u[n:] = g if self._solve is None else self._solve(g)

    def update(self, u: NDArray, f: NDArray) -> None:
        """Restore the boundary rows after the interior values have changed."""
        if self.coupled:
            self.constrain(u, f)


class _Level:
    """Matrices of one level of the hierarchy, in the ordering ``[interior, boundary]``."""

    def __init__(self, A, bc_rows: NDArray, relax_more: NDArray, coarsest: bool) -> None:
        self.size = A.shape[0]
        self.coarsest = coarsest
        self.perm = _interior_first(bc_rows)
        self.n_free = n = self.size - int(bc_rows.sum())
        A = scipy.sparse.csr_matrix(A)[self.perm][:, self.perm]
        if coarsest:
            self.solve = _factorized(A)
            return
        self._interior = A[:n]                                 # interior rows, all columns
        self.boundary = _BoundaryRows(A, n)

        # Jacobi uses the diagonal the interior equations have once the boundary
        # values are eliminated; for Neumann and Robin conditions it differs
        # from the diagonal of the matrix next to the boundary.
        diag = self._interior.diagonal()
        if self.boundary.coupled:
            d = self.boundary.diagonal
            scale = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
            T = scipy.sparse.csr_matrix((scipy.sparse.diags(scale) @ self.boundary.coupling).T)
            diag = diag - np.asarray(self._interior[:, n:].multiply(T).sum(axis=1)).ravel()
        if np.any(diag == 0):
            raise ValueError(
                "The operator has zero diagonal entries, so that it cannot be smoothed. "
                "Multigrid needs an elliptic operator built from second derivatives "
                "(e.g. grid.laplacian_operator() or grid.diff_operator(2, axis))."
            )
        self._weights = _OMEGA / diag

        # Interior points that get extra sweeps (see _near_coupled_faces).
        self._strip = np.flatnonzero(relax_more[self.perm[:n]])
        self._strip_matrix = self._interior[self._strip]

    def set_transfer(self, P, coarse: _Level) -> None:
        """Store interpolation from, and restriction to, the next coarser level."""
        n = self.n_free
        P = scipy.sparse.csr_matrix(P)[self.perm][:, coarse.perm]
        self.prolong = scipy.sparse.csr_matrix(P[:n])
        # Full weighting: the transpose of the interpolation between interior
        # points, scaled to an average.
        R = scipy.sparse.csr_matrix(P[:n, :coarse.n_free].T)
        sums = np.asarray(R.sum(axis=1)).ravel()
        sums[sums == 0] = 1.0
        self.restrict = scipy.sparse.csr_matrix(scipy.sparse.diags(1.0 / sums) @ R)

    def residual(self, u: NDArray, f: NDArray) -> NDArray:
        """Residual of the interior equations."""
        return f[:self.n_free] - self._interior @ u

    def smooth(self, u: NDArray, f: NDArray, sweeps: int) -> None:
        """Damped Jacobi sweeps over the interior, each followed by the boundary rows."""
        strip = self._strip
        for _ in range(sweeps):
            u[:self.n_free] += self._weights * self.residual(u, f)
            self.boundary.update(u, f)
            for _ in range(_BOUNDARY_SWEEPS if len(strip) else 0):
                u[strip] += self._weights[strip] * (f[strip] - self._strip_matrix @ u)
                self.boundary.update(u, f)


# ---------------------------------------------------------------------------
# Setting up the levels
# ---------------------------------------------------------------------------

def _check_axes(grid: Grid) -> None:
    for i, axis in enumerate(grid.axes):
        if type(axis) not in (EquidistantAxis, LogAxis) or axis.periodic:
            kind = "periodic axis" if axis.periodic else type(axis).__name__
            raise TypeError(
                f"Axis {i} is a {kind}, whose spectral differentiation matrices are dense; "
                "multigrid needs the sparse finite-difference operators of equidistant "
                "(non-periodic) or logarithmic axes. Use Operator.solve() for spectral grids."
            )


@functools.cache
def _low_order_class(cls: type[Axis]) -> type[Axis]:
    """Subclass of an axis class whose finite differences are limited to second order."""

    def create_diff_operator(self, grid, order, axis_index, acc=_LOW_ORDER):
        return cls.create_diff_operator(self, grid, order, axis_index, acc=min(acc, _LOW_ORDER))

    return type(cls.__name__, (cls,), {"create_diff_operator": create_diff_operator, "__module__": __name__})


def _low_order_grid(grid: Grid) -> Grid:
    """A copy of *grid* on which all finite differences are of second order."""
    axes = []
    for axis in grid.axes:
        kwargs = {} if axis.name is None else {"name": axis.name}
        axes.append(_low_order_class(type(axis))(len(axis), axis.low, axis.high, **kwargs))
    return Grid(*axes)


def _matrix(op, grid: Grid):
    """The matrix of what the user's function returned for *grid*."""
    if hasattr(op, "to_operator"):
        op = op.to_operator()
    if scipy.sparse.issparse(op) or isinstance(op, np.ndarray):
        matrix = op
    elif hasattr(op, "matrix") and hasattr(op, "grid"):
        if op.grid.shape != grid.shape:
            raise ValueError(
                f"`operator` returned an operator on a grid of shape {op.grid.shape} when called "
                f"with a grid of shape {grid.shape}. Build the operator from the grid passed in."
            )
        matrix = op.matrix
    else:
        raise TypeError(f"`operator` must return an Operator, got {type(op).__name__}.")
    if matrix.shape != (grid.size, grid.size):
        raise ValueError(
            f"`operator` returned a matrix of shape {matrix.shape} for a grid of size {grid.size}."
        )
    return scipy.sparse.csr_matrix(matrix)


def _rebind(bcs: list[BoundaryCondition], grid: Grid) -> list[BoundaryCondition]:
    """The same kinds of conditions on another grid, with zero values.

    Corrections computed on the coarser levels satisfy homogeneous
    conditions, so the boundary values are only needed on the finest grid.
    """
    from numgrids.boundary import DirichletBC, NeumannBC, RobinBC

    result = []
    for bc in bcs:
        face = grid.faces[bc.face.key]
        if isinstance(bc, RobinBC):
            if np.ndim(bc.a) or np.ndim(bc.b):
                raise ValueError("Multigrid needs scalar Robin coefficients a and b.")
            result.append(RobinBC(face, bc.a, bc.b))
        elif isinstance(bc, NeumannBC):
            result.append(NeumannBC(face))
        elif isinstance(bc, DirichletBC):
            result.append(DirichletBC(face))
        else:
            raise TypeError(
                f"Cannot transfer a boundary condition of type {type(bc).__name__} to coarser grids; "
                "multigrid supports Dirichlet, Neumann and Robin conditions."
            )
    return result


def _near_coupled_faces(grid: Grid, bcs: list[BoundaryCondition]) -> NDArray:
    """Boolean mask (flat) of the points close to a face with a Neumann or Robin condition.

    Next to such faces a Jacobi sweep leaves more of the error behind than
    elsewhere.  A few extra sweeps over these points only cost little and
    restore the convergence rate of the Dirichlet problem.
    """
    from numgrids.boundary import DirichletBC

    mask = np.zeros(grid.shape, dtype=bool)
    for bc in bcs:
        if not isinstance(bc, DirichletBC):
            index = [slice(None)] * grid.ndims
            width = _BOUNDARY_WIDTH + 1
            index[bc.face.axis_index] = slice(0, width) if bc.face.side == "low" else slice(-width, None)
            mask[tuple(index)] = True
    return mask.ravel()


def _bc_mask(grid: Grid, bcs: list[BoundaryCondition]) -> NDArray:
    """Boolean mask (flat) of the rows that hold boundary conditions."""
    mask = np.zeros(grid.size, dtype=bool)
    for bc in bcs:
        mask[bc.face.flat_indices] = True
    return mask


def _with_rows(M, rows: NDArray, R):
    """Copy of the matrix *M* in which the masked *rows* are replaced by those of *R*."""
    M = scipy.sparse.csr_matrix(M, copy=True)
    M.data[np.repeat(rows, np.diff(M.indptr))] = 0
    M.eliminate_zeros()
    return scipy.sparse.csr_matrix(M + scipy.sparse.csr_matrix(R))


def _level_matrix(operator: Callable, grid: Grid, bcs: list[BoundaryCondition]):
    """System matrix on a level of the hierarchy."""
    from numgrids.boundary import apply_bcs

    try:
        op = operator(grid)
        level_bcs = _rebind(bcs, grid)
        R, _ = apply_bcs(level_bcs, scipy.sparse.csr_matrix((grid.size, grid.size)), np.zeros(grid.size))
    except ValueError as err:
        raise ValueError(
            f"Could not set up the operator on the level of shape {grid.shape}: {err} "
            "If the level is too coarse for the operator, create the MultiGrid with a larger `min_size`."
        ) from err
    return _with_rows(_matrix(op, grid), _bc_mask(grid, level_bcs), R)


def _interpolation_1d(coarse: Axis, fine: Axis):
    """Sparse matrix of linear interpolation from one axis to a finer one."""
    # Both axes are uniform in their internal coordinate.
    xc, xf = coarse.coords_internal, fine.coords_internal
    left = np.clip(np.searchsorted(xc, xf, side="right") - 1, 0, len(xc) - 2)
    w = np.clip((xf - xc[left]) / (xc[left + 1] - xc[left]), 0.0, 1.0)
    rows = np.arange(len(xf))
    P = scipy.sparse.csr_matrix(
        (np.concatenate([1 - w, w]), (np.concatenate([rows, rows]), np.concatenate([left, left + 1]))),
        shape=(len(xf), len(xc)),
    )
    P.eliminate_zeros()
    return P


def _prolongation(coarse: Grid, fine: Grid):
    """Multilinear interpolation from a coarse to a fine grid as a sparse matrix."""
    P = _interpolation_1d(coarse.axes[0], fine.axes[0])
    for ac, af in zip(coarse.axes[1:], fine.axes[1:]):
        P = scipy.sparse.kron(P, _interpolation_1d(ac, af), format="csr")
    return P


def _factorized(A) -> Callable[[NDArray], NDArray]:
    """Sparse LU factorization of *A*, returned as a solve function."""
    A = scipy.sparse.csc_matrix(A)
    try:
        lu = scipy.sparse.linalg.splu(A)
    except RuntimeError as err:
        raise ValueError(
            "The problem is singular. With Neumann conditions on all faces the solution of a "
            "Poisson-type problem is only determined up to a constant; fix it with a Dirichlet "
            "or Robin condition on at least one face."
        ) from err
    real = not np.iscomplexobj(A)

    def solve(b: NDArray) -> NDArray:
        if real and np.iscomplexobj(b):
            return lu.solve(np.ascontiguousarray(b.real)) + 1j * lu.solve(np.ascontiguousarray(b.imag))
        return lu.solve(b)

    return solve


# ---------------------------------------------------------------------------
# Iterations
# ---------------------------------------------------------------------------

def _iterate(matvec, precond, b: NDArray, x: NDArray, reference: float, tol: float, maxiter: int):
    """Repeat ``x += precond(b - A x)`` until the residual is below ``tol * reference``."""
    residuals: list[float] = []
    while True:
        r = b - matvec(x)
        res = float(np.linalg.norm(r) / reference)
        residuals.append(res)
        if not np.isfinite(res) or res > 1e8 * max(residuals[0], 1.0):
            return x, residuals, False
        if res <= tol:
            return x, residuals, True
        if len(residuals) > maxiter:
            return x, residuals, False
        x = x + precond(r)


def _gmres(matvec, precond, b: NDArray, x: NDArray, reference: float, tol: float, maxiter: int, restart: int):
    """Restarted GMRES with right preconditioning.

    Right preconditioning minimizes the residual of the original system, so
    the recorded residual norms (relative to *reference*) are those of
    ``b - A x``.  The preconditioner must be linear: it is applied to the
    combined Krylov vectors at the end of each restart cycle instead of
    storing the preconditioned vectors.
    """
    complex_valued = np.iscomplexobj(b)
    residuals: list[float] = []
    iterations = 0
    stalled = 0
    previous = np.inf
    while True:
        r = b - matvec(x)
        beta = np.linalg.norm(r)
        res = float(beta / reference)
        # If two restart cycles in a row hardly reduce the residual, rounding
        # errors (or a singular problem) prevent further progress.
        stalled = stalled + 1 if res > 0.5 * previous else 0
        previous = res
        if residuals:
            residuals[-1] = res      # the true residual replaces the estimate of the last iteration
        else:
            residuals.append(res)
        if not np.isfinite(res):
            return x, residuals, False
        if res <= tol:
            return x, residuals, True
        if iterations >= maxiter or stalled >= 2:
            return x, residuals, False

        m = min(restart, maxiter - iterations)
        V = np.empty((m + 1, len(b)), dtype=b.dtype)
        H = np.zeros((m + 1, m), dtype=b.dtype)       # Hessenberg matrix, rotated to triangular form
        cs = np.zeros(m, dtype=b.dtype)
        sn = np.zeros(m)
        g = np.zeros(m + 1, dtype=b.dtype)
        g[0] = beta
        V[0] = r / beta
        k = 0
        for j in range(m):
            w = matvec(precond(V[j]))
            # Classical Gram-Schmidt, done twice for numerical stability.
            for _ in range(2):
                h = (V[:j + 1].conj() if complex_valued else V[:j + 1]) @ w
                w -= h @ V[:j + 1]
                H[:j + 1, j] += h
            hnorm = np.linalg.norm(w)
            if not np.isfinite(hnorm):
                break
            # Givens rotations turn the least-squares problem into a triangular system
            # and give the residual norm of the current iterate for free.
            for i in range(j):
                H[i, j], H[i + 1, j] = (np.conj(cs[i]) * H[i, j] + sn[i] * H[i + 1, j],
                                        -sn[i] * H[i, j] + cs[i] * H[i + 1, j])
            rho = np.sqrt(abs(H[j, j]) ** 2 + hnorm ** 2)
            if rho == 0:
                break
            cs[j], sn[j] = H[j, j] / rho, hnorm / rho
            H[j, j] = rho
            g[j + 1] = -sn[j] * g[j]
            g[j] = np.conj(cs[j]) * g[j]
            k = j + 1
            iterations += 1
            res = float(abs(g[j + 1]) / reference)
            residuals.append(res)
            if res <= tol or hnorm <= 1e-14 * beta:
                break
            V[j + 1] = w / hnorm
        if k == 0:
            residuals.append(float("nan"))
            return x, residuals, False
        y = scipy.linalg.solve_triangular(H[:k, :k], g[:k])
        x = x + precond(y @ V[:k])

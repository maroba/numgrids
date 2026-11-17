"""Nonlinear boundary value problems.

:func:`solve_nonlinear` finds a solution of :math:`F(u) = 0` on a grid,
subject to boundary conditions, with a damped Newton method.  The residual
*F* is an ordinary function of grid-shaped arrays, written with the usual
numgrids tools (``grid.diff``, ``grid.laplacian``, operators, ...).

Examples
--------
The Bratu problem :math:`u'' + e^u = 0`, :math:`u(0) = u(1) = 0`:

>>> import numpy as np
>>> from numgrids import Grid, chebyshev, solve_nonlinear
>>> grid = Grid(x=chebyshev(31, 0, 1))
>>> u = solve_nonlinear(grid, lambda u: grid.diff(u, 2) + np.exp(u), u0=0, bcs=0)
>>> round(float(u.max()), 8)
0.14053921
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numgrids.grids import Grid


# Linear systems up to this size are solved with dense LU.
_DENSE_LIMIT = 3000


class NonlinearSolverError(RuntimeError):
    """Raised when :func:`solve_nonlinear` does not converge.

    Attributes
    ----------
    u : NDArray or tuple of NDArray
        The last iterate.
    residuals : list of float
        Maximum norm of the residual after each Newton step.
    """

    def __init__(self, message: str, u, residuals: list[float]) -> None:
        super().__init__(message)
        self.u = u
        self.residuals = residuals


def solve_nonlinear(
    grid: Grid,
    residual: Callable,
    u0,
    bcs=None,
    jacobian: Callable | None = None,
    tol: float = 1e-10,
    maxiter: int = 50,
    verbose: bool = False,
):
    r"""Solve the nonlinear boundary value problem :math:`F(u) = 0`.

    A damped Newton method: each step solves the linear system
    :math:`J \, \delta u = -F(u)` with the Jacobian :math:`J = \partial F /
    \partial u` (boundary conditions included) and shortens the step if the
    residual would not decrease.

    Parameters
    ----------
    grid : Grid
        The grid.
    residual : callable
        ``residual(u)`` returns :math:`F(u)` as an array of grid shape.  For
        coupled systems with initial guess ``(u0, v0, ...)``,
        ``residual(u, v, ...)`` returns a tuple of arrays.  The residual is
        only used at interior points; on boundary points the boundary
        conditions take its place.
    u0 : float, NDArray, callable, or tuple of these
        Initial guess (scalars are broadcast, callables evaluated on the
        grid).  Newton's method converges to the solution *near* the
        initial guess, so for problems with several solutions the guess
        selects the solution.
    bcs : optional
        Boundary conditions, in any form accepted by
        :func:`~numgrids.boundary.resolve_bcs` (Dirichlet, Neumann, Robin).
        For coupled systems, a tuple with one specification per field
        (``None`` for none); a single specification applies to every field.
    jacobian : callable, optional
        ``jacobian(u)`` (or ``jacobian(u, v, ...)``) returns the Jacobian
        :math:`\partial F / \partial u` as an :class:`~numgrids.Operator`, a
        sparse matrix or a dense array.  For systems, return a nested list
        of blocks, ``J[i][j]`` = :math:`\partial F_i / \partial u_j`, each an
        Operator, matrix, array of grid shape (multiplication), scalar, or
        ``None`` (zero).  If omitted, the Jacobian is approximated by finite
        differences, which costs one evaluation of the residual per grid
        point and Newton step: fine for small and medium grids, but an
        exact Jacobian is much faster for large ones.
    tol : float
        Convergence tolerance: Newton stops when the update is smaller than
        ``tol * (1 + max|u|)`` (maximum norm).
    maxiter : int
        Maximum number of Newton steps.
    verbose : bool
        Print the residual after each step.

    Returns
    -------
    NDArray or tuple of NDArray
        The solution (a tuple for coupled systems).

    Raises
    ------
    NonlinearSolverError
        If Newton's method does not converge.  The exception holds the last
        iterate and the residual history.

    Examples
    --------
    The Bratu problem with an exact Jacobian, written with operators:

    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev, solve_nonlinear
    >>> grid = Grid(x=chebyshev(31, 0, 1))
    >>> D2 = grid.diff_operator(2)
    >>> u = solve_nonlinear(grid, lambda u: D2(u) + np.exp(u), u0=0, bcs=0,
    ...                     jacobian=lambda u: D2 + np.exp(u))
    >>> round(float(u.max()), 8)
    0.14053921
    """
    from numgrids.boundary import BoundaryCondition, apply_bcs, resolve_bcs

    system = isinstance(u0, (tuple, list))
    guesses = list(u0) if system else [u0]
    nfields = len(guesses)
    size = grid.size
    shape = grid.shape

    fields = []
    for g in guesses:
        if callable(g):
            g = grid.evaluate(g)
        g = np.broadcast_to(np.asarray(g), shape)
        fields.append(np.array(g, dtype=np.result_type(g, float)))

    # Boundary conditions per field: rows R u = g replace the residual there.
    per_field = (system and isinstance(bcs, (tuple, list)) and len(bcs) == nfields
                 and not any(isinstance(b, BoundaryCondition) for b in bcs))
    specs = list(bcs) if per_field else [bcs] * nfields
    bc_rows = np.zeros(nfields * size, dtype=bool)
    R_blocks, g_parts = [], []
    for f, spec in enumerate(specs):
        bc_list = resolve_bcs(grid, spec)
        R, g = apply_bcs(bc_list, scipy.sparse.csr_matrix((size, size)), np.zeros(size))
        for bc in bc_list:
            bc_rows[f * size + bc.face.flat_indices] = True
        R_blocks.append(scipy.sparse.csr_matrix(R))
        g_parts.append(np.asarray(g))
    R = scipy.sparse.block_diag(R_blocks, format="csr")
    g = np.concatenate(g_parts)
    keep_rows = scipy.sparse.diags((~bc_rows).astype(float))

    def split(y):
        return [y[f * size:(f + 1) * size].reshape(shape) for f in range(nfields)]

    def F(y):
        out = residual(*split(y))
        if system:
            if len(out) != nfields:
                raise ValueError(f"residual returned {len(out)} arrays, expected {nfields}.")
            r = np.concatenate([np.broadcast_to(np.asarray(o), shape).ravel() for o in out])
        else:
            r = np.broadcast_to(np.asarray(out), shape).ravel()
        r = np.array(r, dtype=np.result_type(r, y), copy=True)
        r[bc_rows] = (R @ y - g)[bc_rows]
        return r

    R_dense_rows = R[np.flatnonzero(bc_rows)].toarray() if jacobian is None else None

    def newton_step(y, r):
        """Solve J dy = -r, with the boundary conditions in the boundary rows."""
        if jacobian is None:
            Jm = _fd_jacobian(F, y, r)
            Jm[bc_rows] = R_dense_rows
            return scipy.linalg.solve(Jm, -r)
        Jm = _assemble_jacobian(grid, jacobian(*split(y)), nfields)
        Jm = scipy.sparse.csc_matrix(keep_rows @ Jm + R)
        if Jm.shape[0] <= _DENSE_LIMIT:
            # Spectral differentiation matrices fill in heavily under sparse LU;
            # for moderate sizes, dense LU is much faster.
            return scipy.linalg.solve(Jm.toarray(), -r)
        return scipy.sparse.linalg.spsolve(Jm, -r)

    y = np.concatenate([f.ravel() for f in fields])
    r = F(y)
    norm = np.max(np.abs(r))
    residuals = []
    for it in range(1, maxiter + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", scipy.sparse.linalg.MatrixRankWarning)
                warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
                dy = newton_step(y, r)
        except (RuntimeError, np.linalg.LinAlgError, scipy.sparse.linalg.MatrixRankWarning) as err:
            raise NonlinearSolverError(f"Singular Jacobian in Newton step {it}: {err}",
                                       _shape_output(split(y), system), residuals) from err
        if not np.all(np.isfinite(dy)):
            raise NonlinearSolverError(f"Singular Jacobian in Newton step {it}.",
                                       _shape_output(split(y), system), residuals)
        # Backtracking: halve the step until the residual decreases.
        alpha = 1.0
        while True:
            y_new = y + alpha * dy
            r_new = F(y_new)
            norm_new = np.max(np.abs(r_new))
            if np.isfinite(norm_new) and norm_new <= (1 - 1e-4 * alpha) * norm:
                break
            if alpha < 1e-3:
                break
            alpha /= 2
        step = alpha * np.max(np.abs(dy))
        y, r, norm = y_new, r_new, norm_new
        residuals.append(float(norm))
        if verbose:
            print(f"Newton step {it}: |F| = {norm:.3e}, |du| = {step:.3e}, damping = {alpha:g}")
        if not np.isfinite(norm):
            break
        if step <= tol * (1 + np.max(np.abs(y))):
            return _shape_output(split(y), system)
    raise NonlinearSolverError(
        f"Newton's method did not converge in {maxiter} steps (residual {norm:.3e}). "
        "Try a better initial guess (e.g. continuation from a nearby problem).",
        _shape_output(split(y), system),
        residuals,
    )


def _shape_output(parts, system):
    return tuple(parts) if system else parts[0]


def _fd_jacobian(F, y: NDArray, r: NDArray) -> NDArray:
    """Forward-difference approximation of the Jacobian of *F* at *y*."""
    n = y.size
    eps = np.sqrt(np.finfo(float).eps)
    Jm = np.empty((n, n), dtype=np.result_type(r, y))
    for j in range(n):
        h = eps * max(1.0, abs(y[j]))
        yp = y.copy()
        yp[j] += h
        h = yp[j] - y[j]  # exactly representable step
        Jm[:, j] = (F(yp) - r) / h
    return Jm


def _assemble_jacobian(grid: Grid, Jspec, nfields: int):
    """Turn a user-supplied Jacobian (operator, matrix, or block list) into a sparse matrix."""
    from numgrids.operators import Operator

    size = grid.size

    def block(b):
        if b is None:
            return None
        if scipy.sparse.issparse(b):
            return scipy.sparse.csr_matrix(b)
        if isinstance(b, np.ndarray) and b.ndim == 2 and b.shape == (size, size) \
                and b.shape != grid.shape:
            return scipy.sparse.csr_matrix(b)
        M = Operator.identity(grid)._coerce(b)
        if M is None:
            raise TypeError(f"Cannot use an object of type {type(b).__name__} as a Jacobian.")
        return scipy.sparse.csr_matrix(M)

    if nfields == 1 and not isinstance(Jspec, (list, tuple)):
        return block(Jspec)
    if (not isinstance(Jspec, (list, tuple)) or len(Jspec) != nfields
            or any(not isinstance(row, (list, tuple)) or len(row) != nfields for row in Jspec)):
        raise ValueError(f"The Jacobian of a system of {nfields} fields must be a "
                         f"{nfields} x {nfields} nested list of blocks.")
    blocks = [[block(b) for b in row] for row in Jspec]
    # bmat needs at least one block per row and column to infer shapes.
    for i in range(nfields):
        if all(blocks[i][j] is None for j in range(nfields)):
            blocks[i][i] = scipy.sparse.csr_matrix((size, size))
    for j in range(nfields):
        if all(blocks[i][j] is None for i in range(nfields)):
            blocks[j][j] = scipy.sparse.csr_matrix((size, size))
    return scipy.sparse.bmat(blocks, format="csr")

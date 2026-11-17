"""Nonlinear boundary value problems.

:func:`solve_nonlinear` finds a solution of :math:`F(u) = 0` on a grid,
subject to boundary conditions, with a damped Newton method.  The residual
*F* is an ordinary function of grid-shaped arrays, written with the usual
numgrids tools (``grid.diff``, ``grid.laplacian``, operators, ...).

:func:`continuation` follows the solutions of :math:`F(u, \\lambda) = 0`
while the parameter :math:`\\lambda` changes (pseudo-arclength
continuation), also around folds, where the branch turns back.  It
locates the folds and, on request, computes the stability of the solutions
and detects the bifurcations along the branch.  :func:`linear_stability`
(see :mod:`numgrids.stability`) tells if a single solution is stable as a
steady state of the evolution equation :math:`u_t = F(u)`.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from numgrids.stability import (
    _NEUTRAL,
    Bifurcation,
    StabilityResult,
    _analyze_branch,
    _Linearization,
    _parabola_extremum,
    _turning_points,
    linear_stability,
)

if TYPE_CHECKING:
    from numgrids.grids import Grid

__all__ = [
    "Bifurcation",
    "ContinuationResult",
    "NonlinearSolverError",
    "StabilityResult",
    "continuation",
    "linear_stability",
    "solve_nonlinear",
]


# Linear systems up to this size are solved with dense LU.
_DENSE_LIMIT = 3000

# Step size control of the continuation: a step is rejected if the tangent turns by
# more than 60 degrees; the step grows if the corrector needs at most 3 Newton steps.
_MIN_COS = 0.5
_FAST_CORRECTOR = 3
_GROWTH = 1.5


class NonlinearSolverError(RuntimeError):
    """Raised when Newton's method does not converge (see :func:`solve_nonlinear`).

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
    problem = _Problem(grid, residual, u0, bcs, jacobian)
    return problem.output(_newton(problem, problem.y0, (), tol, maxiter, verbose))


@dataclass
class ContinuationResult:
    """Result of :func:`continuation`: a branch of solutions.

    Attributes
    ----------
    params : NDArray
        The parameter at the points of the branch, shape ``(n,)``, in the
        order in which the branch was traced.
    u : NDArray or tuple of NDArray
        The solutions, shape ``(n, *grid.shape)``: ``u[k]`` solves the
        problem for ``params[k]``.  For coupled systems, a tuple with one
        such array per field.
    success : bool
        ``False`` if the continuation broke down (the corrector failed even
        with the smallest step size).  The branch up to that point is still
        valid.
    message : str
        Why the continuation stopped.
    eigenvalues : NDArray, optional
        With ``stability=k``: the *k* leading eigenvalues of the
        linearization at each point, shape ``(n, k)``, largest real part
        first (see :func:`~numgrids.linear_stability`).  A real
        array if all of them are real.
    stable : NDArray of bool, optional
        With ``stability=k``: whether the solution at each point is
        linearly stable (all computed eigenvalues have a negative real
        part).
    n_unstable : NDArray of int, optional
        With ``stability=k``: the number of eigenvalues with a positive
        real part at each point, among the computed ones.
    bifurcations : list of Bifurcation, optional
        The bifurcations in the order in which they occur along the
        branch: the folds and, with ``stability=k``, the branch points and
        Hopf bifurcations (see :class:`~numgrids.Bifurcation`).
    """

    params: NDArray
    u: Any
    success: bool
    message: str
    eigenvalues: NDArray | None = None
    stable: NDArray | None = None
    n_unstable: NDArray | None = None
    bifurcations: list[Bifurcation] | None = None

    def __repr__(self) -> str:
        if len(self.params) == 0:
            return "ContinuationResult(0 points)"
        text = (f"ContinuationResult({len(self.params)} points, parameter from "
                f"{self.params[0]:.6g} to {self.params[-1]:.6g}, {len(self.folds)} fold(s)")
        if self.stable is not None:
            text += f", {int(np.sum(self.stable))} stable point(s)"
            kinds = [bif.kind for bif in self.bifurcations or []]
            for kind, name in (("branch_point", "branch point(s)"), ("hopf", "Hopf bifurcation(s)")):
                if kind in kinds:
                    text += f", {kinds.count(kind)} {name}"
        return text + f", success={self.success})"

    def __len__(self) -> int:
        """Number of points on the branch."""
        return len(self.params)

    @property
    def folds(self) -> NDArray:
        r"""Indices of the points where the parameter turns around.

        At a fold (turning point, saddle-node bifurcation) the branch
        reverses its direction in the parameter: :math:`d\lambda / ds`
        changes sign.  ``params[folds]`` are the largest (or smallest)
        parameters among the points of the branch; the folds themselves
        lie between the points, see :attr:`fold_params`.
        """
        return _turning_points(self.params)[1]

    @property
    def fold_params(self) -> NDArray:
        r"""The parameters of the folds, one for each entry of :attr:`folds`.

        :func:`continuation` locates each fold on the branch as the point
        where :math:`d\lambda / ds = 0`, to the accuracy of the solver
        (not of the step *ds*).  Without that (``refine=False``), the
        values are estimates: the extremum of a parabola through the three
        points of the branch around the fold.
        """
        if self.bifurcations is not None:
            return np.array([bif.param for bif in self.bifurcations if bif.kind == "fold"])
        fields = self.u if isinstance(self.u, tuple) else (self.u,)
        found = []
        for k in self.folds:
            # Chord lengths in the arclength of the continuation.
            chords = [np.sqrt(np.mean([np.mean((f[j + 1] - f[j]) ** 2) for f in fields])
                              + (self.params[j + 1] - self.params[j]) ** 2) for j in (k - 1, k)]
            found.append(_parabola_extremum(np.cumsum([0.0, *chords]), self.params[k - 1:k + 2]))
        return np.array(found)

    @property
    def fold_u(self):
        """The solutions at the folds, shape ``(len(folds), *grid.shape)``.

        A tuple with one such array per field for coupled systems.  If the
        folds have not been located (see :attr:`fold_params`), the nearest
        points of the branch.
        """
        if self.bifurcations is None:
            return tuple(f[self.folds] for f in self.u) if isinstance(self.u, tuple) else self.u[self.folds]
        at_folds = [bif.u for bif in self.bifurcations if bif.kind == "fold"]
        if isinstance(self.u, tuple):
            return tuple(np.array([u[f] for u in at_folds]).reshape(-1, *field.shape[1:])
                         for f, field in enumerate(self.u))
        return np.array(at_folds).reshape(-1, *self.u.shape[1:])

    def measure(self, func: Callable) -> NDArray:
        """Evaluate a function on every solution of the branch.

        Parameters
        ----------
        func : callable
            ``func(u)`` (or ``func(u, v, ...)`` for coupled systems) maps
            one solution to a number, e.g. ``np.max`` or a norm.

        Returns
        -------
        NDArray
            ``func`` of each solution, e.g. to plot against :attr:`params`
            in a bifurcation diagram.
        """
        if isinstance(self.u, tuple):
            return np.array([func(*fields) for fields in zip(*self.u)])
        return np.array([func(u) for u in self.u])


def continuation(
    grid: Grid,
    residual: Callable,
    u0,
    param: float,
    bcs=None,
    jacobian: Callable | None = None,
    *,
    ds: float = 0.1,
    max_steps: int = 100,
    param_range: tuple[float, float] | None = None,
    direction: int = 1,
    ds_min: float = 1e-6,
    ds_max: float | None = None,
    tol: float = 1e-10,
    maxiter: int = 20,
    callback: Callable | None = None,
    verbose: bool = False,
    stability: int | None = None,
    mass=None,
    sign: int = 1,
    refine: bool = True,
    tangent=None,
) -> ContinuationResult:
    r"""Follow a branch of solutions of :math:`F(u, \lambda) = 0` in the parameter.

    Pseudo-arclength continuation: the branch is traced as a curve
    :math:`(u(s), \lambda(s))`.  Each step predicts the next point along
    the tangent of the curve and corrects it with Newton's method, in the
    plane perpendicular to the tangent.  As the parameter is an unknown
    like the solution, the method passes folds (turning points), where the
    branch turns back in the parameter and solving at a fixed parameter
    fails.

    Parameters
    ----------
    grid : Grid
        The grid.
    residual : callable
        ``residual(u, lam)`` returns :math:`F(u, \lambda)` as an array of
        grid shape, as in :func:`solve_nonlinear`, with the parameter as
        the last argument.  For coupled systems with initial guess
        ``(u0, v0, ...)``, ``residual(u, v, ..., lam)`` returns a tuple of
        arrays.
    u0 : float, NDArray, callable, or tuple of these
        Initial guess for the solution at the parameter *param*.  The
        continuation starts from the solution that Newton's method finds
        from this guess.
    param : float
        Parameter at the start of the branch.
    bcs : optional
        Boundary conditions, as in :func:`solve_nonlinear`.  They must not
        depend on the parameter.
    jacobian : callable, optional
        ``jacobian(u, lam)`` (or ``jacobian(u, v, ..., lam)``) returns
        :math:`\partial F / \partial u`, in the forms that
        :func:`solve_nonlinear` accepts.  If omitted, it is approximated by
        finite differences.  The derivative with respect to the parameter
        always is a finite difference (one evaluation of the residual).
    ds : float
        Step along the branch, measured in the arclength
        :math:`ds^2 = \overline{du^2} + d\lambda^2` (mean over all grid
        points and fields), so the same *ds* resolves a branch equally on
        coarse and fine grids.
    max_steps : int
        Maximum number of steps (points on the branch after the first).
    param_range : tuple of float, optional
        ``(low, high)``: stop when the parameter leaves this interval.  The
        last point then lies exactly on the end of the interval.
    direction : {1, -1}
        Start towards increasing (``1``) or decreasing (``-1``) parameters.
    ds_min : float
        Smallest step.  The step is halved whenever the corrector fails;
        below *ds_min* the continuation stops with ``success=False``.
    ds_max : float, optional
        Largest step.  The step grows again after a quickly converging
        corrector, up to *ds_max*.  By default *ds*: the step only shrinks
        where the branch requires it.
    tol : float
        Convergence tolerance of Newton's method, as in
        :func:`solve_nonlinear`.
    maxiter : int
        Maximum number of Newton steps, for the first solution and for
        each corrector.
    callback : callable, optional
        ``callback(u, lam)`` (or ``callback(u, v, ..., lam)``) is called
        for every point of the branch.  If it returns ``True``, the
        continuation stops.
    verbose : bool
        Print the parameter and the step size after each step.
    stability : int, optional
        Compute this many leading eigenvalues of the linearization at
        every point of the branch (see
        :func:`~numgrids.linear_stability`), and detect the
        bifurcations where eigenvalues cross the imaginary axis.  The
        result then has the attributes ``eigenvalues``, ``stable`` and
        ``n_unstable``, and ``bifurcations`` lists branch points and Hopf
        bifurcations besides the folds.  The eigenvalues belong to the
        evolution equation :math:`u_t = F(u, \lambda)`.
    mass : optional
        With *stability*: the operator :math:`M` of the evolution equation
        :math:`M u_t = F(u, \lambda)`, as in
        :func:`~numgrids.linear_stability`.
    sign : {1, -1}
        With *stability*: ``-1`` for the evolution equation
        :math:`u_t = -F(u, \lambda)`.
    refine : bool
        Locate the folds and the other bifurcations on the branch, to the
        accuracy of the solver (a few additional solves for each of them).
        If ``False``, they are only interpolated between the points of the
        branch.
    tangent : NDArray or tuple of NDArray, optional
        Branch switching: start from *u0* at *param* along this direction
        in the solution (fields like *u0*) instead of along the parameter;
        *direction* selects the side.  For a branch point ``bp`` detected
        on another branch, ``u0=bp.u, param=bp.param, tangent=bp.mode``
        follows the branch that bifurcates there.  *u0* is taken as the
        solution at *param* without correcting it (the Jacobian is
        singular at a branch point).

    Returns
    -------
    ContinuationResult
        The branch: parameters ``params``, solutions ``u``, the indices
        ``folds`` of the turning points with their accurate parameters
        ``fold_params``, the ``bifurcations``, and ``measure(func)`` to
        reduce each solution to a number for a bifurcation diagram.

    Raises
    ------
    NonlinearSolverError
        If Newton's method does not find the first solution at *param*.

    Notes
    -----
    The continuation stays on the branch it follows, also at branch points,
    where several branches cross.  With *stability* the branch points are
    detected, and *tangent* starts a continuation of the other branch.

    Bifurcations are found from the eigenvalues at neighbouring points of
    the branch: an eigenvalue that crosses the imaginary axis and returns
    within one step goes unnoticed, as does one that is not among the
    *stability* leading eigenvalues at these points.  In particular,
    ``n_unstable`` cannot exceed the number of computed eigenvalues; choose
    it larger than the number of unstable modes of interest.  Branch
    switching with *tangent* assumes that the new branch leaves the branch
    point roughly along the critical mode, as at a pitchfork bifurcation.

    Examples
    --------
    The Bratu problem :math:`u'' + \lambda e^u = 0`, :math:`u(0) = u(1) = 0`
    has solutions only up to :math:`\lambda_c = 3.5138...`, where the
    branch folds back:

    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev, continuation
    >>> grid = Grid(x=chebyshev(31, 0, 1))
    >>> D2 = grid.diff_operator(2)
    >>> branch = continuation(grid, lambda u, lam: D2(u) + lam * np.exp(u), u0=0, param=0,
    ...                       bcs=0, jacobian=lambda u, lam: D2 + lam * np.exp(u),
    ...                       ds=0.05, max_steps=100)
    >>> len(branch), len(branch.folds)
    (101, 1)
    >>> round(float(branch.params[branch.folds[0]]), 2)
    3.51

    The fold itself lies between two points of the branch and is located
    accurately:

    >>> round(float(branch.fold_params[0]), 8)
    3.51383072

    Beyond the fold, the parameter decreases again, on the upper branch of
    solutions:

    >>> max_u = branch.measure(np.max)
    >>> bool(branch.params[-1] < 3.51 and max_u[-1] > max_u[branch.folds[0]])
    True

    With ``stability=k``, the *k* leading eigenvalues of the linearization
    are computed along the branch.  The solutions are stable up to the
    fold and unstable (with one growing mode) behind it:

    >>> branch = continuation(grid, lambda u, lam: D2(u) + lam * np.exp(u), u0=0, param=0,
    ...                       bcs=0, jacobian=lambda u, lam: D2 + lam * np.exp(u),
    ...                       ds=0.05, max_steps=100, stability=4)
    >>> branch
    ContinuationResult(101 points, parameter from 0 to 2.42244, 1 fold(s), 74 stable point(s), success=True)
    >>> branch.bifurcations
    [Bifurcation(fold at param = 3.513830719, between the points 73 and 74)]
    >>> bool(np.all(branch.stable[:74])), int(branch.n_unstable[-1])
    (True, 1)
    """
    problem = _Problem(grid, residual, u0, bcs, jacobian)
    if np.iscomplexobj(problem.y0):
        raise TypeError("continuation needs real-valued fields.")
    if direction not in (1, -1):
        raise ValueError(f"direction must be 1 or -1, got {direction!r}.")
    if not 0 < ds_min <= ds:
        raise ValueError(f"Need 0 < ds_min <= ds, got ds = {ds}, ds_min = {ds_min}.")
    ds_max = ds if ds_max is None else ds_max
    lam0 = float(param)
    lo, hi = (-np.inf, np.inf) if param_range is None else sorted(map(float, param_range))
    if not lo <= lam0 <= hi:
        raise ValueError(f"param = {param} lies outside of param_range = {tuple(param_range)}.")
    linearization = None if stability is None else _Linearization(problem, stability, mass, sign)

    n = problem.y0.size
    eps = np.sqrt(np.finfo(float).eps)

    def dot(y1, l1, y2, l2):
        # Mean instead of sum over the grid: arclengths independent of the resolution.
        return float(y1 @ y2) / n + l1 * l2

    def bordered(y, r, lam, ty, tl, accurate=False):
        """Factorize [[dF/dy, dF/dlam], [ty / n, tl]], the matrix of the bordered systems."""
        if accurate:
            # Central difference: tangents to 1e-10, as needed to locate bifurcations.
            h = np.cbrt(np.finfo(float).eps) * max(1.0, abs(lam))
            Fl = (problem.F(y, lam + h) - problem.F(y, lam - h)) / ((lam + h) - (lam - h))
        else:
            h = eps * max(1.0, abs(lam))
            h = (lam + h) - lam  # exactly representable step
            Fl = (problem.F(y, lam + h) - r) / h
        J = problem.jacobian(y, r, lam, accurate=accurate)
        if scipy.sparse.issparse(J):
            M = scipy.sparse.bmat([[J, scipy.sparse.csr_matrix(Fl[:, None])],
                                   [scipy.sparse.csr_matrix(ty[None, :] / n), np.array([[tl]])]],
                                  format="csc")
        else:
            M = np.empty((n + 1, n + 1))
            M[:n, :n], M[:n, n], M[n, :n], M[n, n] = J, Fl, ty / n, tl
        return _factorize(M)

    def unit_tangent(solve):
        """Unit tangent, and the cosine of its angle with the tangent in the border row."""
        e = np.zeros(n + 1)
        e[n] = 1.0
        t = solve(e)
        norm = np.sqrt(dot(t[:n], t[n], t[:n], t[n]))
        return t[:n] / norm, t[n] / norm, 1 / norm

    def advance(y, lam, ty, tl, h, accurate=False):
        """One predictor-corrector step of length *h*; ``None`` if it fails."""
        y_pred, lam_pred = y + h * ty, lam + h * tl
        y, lam = y_pred, lam_pred
        its = 0
        try:
            # Newton's method for F = 0 in the plane through the predictor, normal to the tangent.
            for _ in range(maxiter):
                its += 1
                r = problem.F(y, lam)
                if not np.all(np.isfinite(r)):
                    return None
                solve = bordered(y, r, lam, ty, tl, accurate)
                dz = solve(-np.append(r, dot(ty, tl, y - y_pred, lam - lam_pred)))
                y, lam = y + dz[:n], lam + dz[n]
                if np.max(np.abs(dz)) <= tol * (1 + np.max(np.abs(y))):
                    break
            else:
                return None
            # The matrix of the last Newton step also yields the tangent at the new point,
            # oriented like the old one.
            ty_new, tl_new, cos = unit_tangent(solve)
        except np.linalg.LinAlgError:
            return None
        # A long way from the predictor or a sharp turn: the step is too long for the
        # curvature of the branch (the corrector may even have jumped to another branch).
        dy, dl = y - y_pred, lam - lam_pred
        if dot(dy, dl, dy, dl) > h ** 2 or cos < _MIN_COS:
            return None
        return y, lam, ty_new, tl_new, its

    ys: list[NDArray] = []
    lams: list[float] = []
    # Unit tangent (ty, tl) of the branch at each point, None where it is not known.
    tangents: list[tuple[NDArray, float] | None] = []
    eigs: list[NDArray] = []

    def accept(y, lam, info="", t=None):
        ys.append(y)
        lams.append(lam)
        tangents.append(t)
        if linearization is not None:
            try:
                eigs.append(linearization.leading(y, (lam,))[0])
            except np.linalg.LinAlgError:
                # The eigensolver failed; the branch itself is not affected.
                eigs.append(np.full(linearization.k, np.nan, dtype=complex))
            info += f", {int(np.sum(eigs[-1].real > _NEUTRAL))} unstable mode(s)"
        if verbose:
            print(f"Continuation step {len(lams) - 1}: param = {lam:.8g}{info}")
        return callback is not None and bool(callback(*problem.split(y), lam))

    def trace(y, lam, start):
        """Trace the branch from the solution *y* at *lam*; returns ``(success, message)``.

        *start* is the tangent to start along; by default the tangent of
        the branch through the starting point.
        """
        if accept(y, lam):
            return True, "Stopped by the callback."
        if start is None:
            try:
                ty, tl, _ = unit_tangent(bordered(y, problem.F(y, lam), lam, np.zeros(n), float(direction)))
            except np.linalg.LinAlgError:
                return False, ("The branch has no unique tangent at the starting point "
                               "(singular Jacobian).")
        else:
            ty, tl = start
        tangents[0] = (ty, tl)
        h = min(ds, ds_max)
        while len(lams) <= max_steps:  # the starting point does not count
            step = advance(y, lam, ty, tl, h)
            if step is None:
                h /= 2
                if verbose:
                    print(f"    corrector failed, new step size ds = {h:.3g}")
                if h < ds_min:
                    return False, (f"The corrector failed near param = {lam:.6g}, even with "
                                   f"the smallest step size ds_min = {ds_min:g}.")
                continue
            y_new, lam_new, ty, tl, its = step
            if not lo <= lam_new <= hi:
                # End exactly on the end of the range: an ordinary Newton solve there,
                # starting from the interpolation between the last two points.
                end = lo if lam_new < lo else hi
                if end != lam:
                    theta = (end - lam) / (lam_new - lam)
                    try:
                        y_end = _newton(problem, y + theta * (y_new - y), (end,), tol, maxiter, False)
                    except NonlinearSolverError:
                        pass
                    else:
                        t_end = None
                        if refine:
                            try:
                                # Oriented like the tangent at the point beyond the end.
                                t_end = unit_tangent(bordered(y_end, problem.F(y_end, end), end, ty, tl))[:2]
                            except np.linalg.LinAlgError:
                                pass
                        accept(y_end, end, " (end of param_range)", t_end)
                return True, f"The parameter reached the end of param_range at {end:g}."
            y, lam = y_new, lam_new
            if accept(y, lam, f", ds = {h:.3g}, {its} Newton steps", (ty, tl)):
                return True, "Stopped by the callback."
            if its <= _FAST_CORRECTOR:
                h = min(_GROWTH * h, ds_max)
        return True, f"Reached max_steps = {max_steps}."

    def step(y, lam, ty, tl, h):
        """The point of the branch at the arclength *h* from a point with tangent ``(ty, tl)``."""
        out = advance(y, lam, ty, tl, h, accurate=True)
        return None if out is None else out[:4]

    if tangent is None:
        y0 = _newton(problem, problem.y0, (lam0,), tol, maxiter, False)
        start = None
    else:
        # Branch switching: the starting point is a branch point, where Newton's method
        # for a fixed parameter has a singular Jacobian.
        y0 = problem.y0
        ty0 = problem.stack(tangent)
        norm = np.sqrt(dot(ty0, 0.0, ty0, 0.0)) if not np.iscomplexobj(ty0) else 0.0
        if not norm > 0:
            raise ValueError("tangent must be a real, nonzero direction.")
        start = (direction * ty0 / norm, 0.0)
    # Rejected trial steps may leave the domain of the residual; that is no reason to warn.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        success, message = trace(y0, lam0, start)
        eigenvalues = stable = n_unstable = None
        if linearization is not None:
            eigenvalues = np.full((len(eigs), max(len(e) for e in eigs)), np.nan, dtype=complex)
            for j, e in enumerate(eigs):
                eigenvalues[j, :len(e)] = e
            stable = np.all(eigenvalues.real < -_NEUTRAL, axis=1)
            n_unstable = np.sum(eigenvalues.real > _NEUTRAL, axis=1)
        bifurcations = _analyze_branch(problem, ys, lams, tangents, eigenvalues, step, dot,
                                       linearization, refine, tangent is not None)
        if eigenvalues is not None and np.all(eigenvalues.imag == 0):
            eigenvalues = eigenvalues.real

    size = grid.size
    Y = np.array(ys)
    fields = [Y[:, f * size:(f + 1) * size].reshape(-1, *grid.shape) for f in range(problem.nfields)]
    return ContinuationResult(np.array(lams), tuple(fields) if problem.system else fields[0],
                              success, message, eigenvalues, stable, n_unstable, bifurcations)


class _Problem:
    """Discrete form of the boundary value problem :math:`F(u, ...) = 0`.

    All fields are stacked in one vector *y*.  On boundary points, the rows
    ``R y = g`` of the boundary conditions replace the residual.  Extra
    arguments (the parameter in :func:`continuation`) are passed on to the
    residual and the Jacobian of the user.
    """

    def __init__(self, grid: Grid, residual: Callable, u0, bcs, jacobian: Callable | None) -> None:
        from numgrids.boundary import BoundaryCondition, apply_bcs, resolve_bcs

        self.grid = grid
        self.residual = residual
        self.user_jacobian = jacobian
        self.system = isinstance(u0, (tuple, list))
        self.nfields = nfields = len(u0) if self.system else 1
        size = grid.size

        self.y0 = self.stack(u0)

        # Boundary conditions per field: rows R u = g replace the residual there.
        per_field = (self.system and isinstance(bcs, (tuple, list)) and len(bcs) == nfields
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
        self.bc_rows = bc_rows
        self.R = scipy.sparse.block_diag(R_blocks, format="csr")
        self.g = np.concatenate(g_parts)
        self.keep_rows = scipy.sparse.diags((~bc_rows).astype(float))
        self.R_dense_rows = self.R[np.flatnonzero(bc_rows)].toarray() if jacobian is None else None

    def stack(self, u) -> NDArray:
        """The stacked vector of fields given like the initial guess (scalars, arrays, callables)."""
        guesses = list(u) if self.system and isinstance(u, (tuple, list)) else [u]
        if len(guesses) != self.nfields:
            raise ValueError(f"Expected {self.nfields} field(s), got {len(guesses)}.")
        fields = []
        for g in guesses:
            if callable(g):
                g = self.grid.evaluate(g)
            g = np.broadcast_to(np.asarray(g), self.grid.shape)
            fields.append(np.array(g, dtype=np.result_type(g, float)))
        return np.concatenate([f.ravel() for f in fields])

    def split(self, y: NDArray) -> list[NDArray]:
        """The fields in the stacked vector *y*, in grid shape."""
        size, shape = self.grid.size, self.grid.shape
        return [y[f * size:(f + 1) * size].reshape(shape) for f in range(self.nfields)]

    def output(self, y: NDArray):
        """The stacked vector *y* as the user gets it: an array, or a tuple for systems."""
        parts = self.split(y)
        return tuple(parts) if self.system else parts[0]

    def F(self, y: NDArray, *args) -> NDArray:
        """The residual, with the boundary conditions in the boundary rows."""
        shape = self.grid.shape
        out = self.residual(*self.split(y), *args)
        if self.system:
            if len(out) != self.nfields:
                raise ValueError(f"residual returned {len(out)} arrays, expected {self.nfields}.")
            r = np.concatenate([np.broadcast_to(np.asarray(o), shape).ravel() for o in out])
        else:
            r = np.broadcast_to(np.asarray(out), shape).ravel()
        r = np.array(r, dtype=np.result_type(r, y), copy=True)
        r[self.bc_rows] = (self.R @ y - self.g)[self.bc_rows]
        return r

    def jacobian(self, y: NDArray, r: NDArray, *args, accurate: bool = False):
        """The Jacobian dF/dy at *y*, where ``F(y, *args) = r``, with the boundary rows.

        Dense for the finite-difference approximation and for moderate
        sizes, sparse (CSC) otherwise.  With *accurate*, the approximation
        uses central differences: twice the work, but errors of 1e-10
        instead of 1e-7 (relative).  Newton's method does not need that,
        eigenvalues do.
        """
        if self.user_jacobian is None:
            Jm = _fd_jacobian(lambda v: self.F(v, *args), y, r, central=accurate)
            Jm[self.bc_rows] = self.R_dense_rows
            return Jm
        Jm = _assemble_jacobian(self.grid, self.user_jacobian(*self.split(y), *args), self.nfields)
        Jm = scipy.sparse.csc_matrix(self.keep_rows @ Jm + self.R)
        if Jm.shape[0] <= _DENSE_LIMIT:
            # Spectral differentiation matrices fill in heavily under sparse LU;
            # for moderate sizes, dense LU is much faster.
            return Jm.toarray()
        return Jm


def _newton(problem: _Problem, y: NDArray, args: tuple, tol: float, maxiter: int,
            verbose: bool) -> NDArray:
    """Damped Newton iteration for ``problem.F(y, *args) = 0``, starting from *y*."""
    F = problem.F
    r = F(y, *args)
    norm = np.max(np.abs(r))
    residuals = []
    for it in range(1, maxiter + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", scipy.sparse.linalg.MatrixRankWarning)
                warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
                # Solve J dy = -r, with the boundary conditions in the boundary rows.
                Jm = problem.jacobian(y, r, *args)
                if scipy.sparse.issparse(Jm):
                    dy = scipy.sparse.linalg.spsolve(Jm, -r)
                else:
                    dy = scipy.linalg.solve(Jm, -r)
        except (RuntimeError, np.linalg.LinAlgError, scipy.sparse.linalg.MatrixRankWarning) as err:
            raise NonlinearSolverError(f"Singular Jacobian in Newton step {it}: {err}",
                                       problem.output(y), residuals) from err
        if not np.all(np.isfinite(dy)):
            raise NonlinearSolverError(f"Singular Jacobian in Newton step {it}.",
                                       problem.output(y), residuals)
        # Backtracking: halve the step until the residual decreases.
        alpha = 1.0
        while True:
            y_new = y + alpha * dy
            r_new = F(y_new, *args)
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
            return y
    raise NonlinearSolverError(
        f"Newton's method did not converge in {maxiter} steps (residual {norm:.3e}). "
        "Try a better initial guess (e.g. continuation from a nearby problem).",
        problem.output(y),
        residuals,
    )


def _factorize(M) -> Callable[[NDArray], NDArray]:
    """LU factorization of a dense or sparse matrix: returns the function ``b -> M^-1 b``.

    The factorization or the returned function raise ``LinAlgError`` if *M*
    is singular.
    """
    if scipy.sparse.issparse(M):
        try:
            backsolve = scipy.sparse.linalg.splu(scipy.sparse.csc_matrix(M)).solve
        except RuntimeError as err:
            raise np.linalg.LinAlgError(str(err)) from err
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            lu = scipy.linalg.lu_factor(M, check_finite=False)

        def backsolve(b):
            return scipy.linalg.lu_solve(lu, b, check_finite=False)

    def solve(b: NDArray) -> NDArray:
        x = backsolve(b)
        if not np.all(np.isfinite(x)):
            raise np.linalg.LinAlgError("Singular matrix.")
        return x

    return solve


def _fd_jacobian(F, y: NDArray, r: NDArray, central: bool = False) -> NDArray:
    """Finite-difference approximation of the Jacobian of *F* at *y*, where ``F(y) = r``.

    Forward differences, or central differences (more accurate, twice the
    number of evaluations).
    """
    n = y.size
    eps = np.cbrt(np.finfo(float).eps) if central else np.sqrt(np.finfo(float).eps)
    Jm = np.empty((n, n), dtype=np.result_type(r, y))
    for j in range(n):
        h = eps * max(1.0, abs(y[j]))
        yp = y.copy()
        yp[j] += h
        if central:
            ym = y.copy()
            ym[j] -= h
            Jm[:, j] = (F(yp) - F(ym)) / (yp[j] - ym[j])
        else:
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

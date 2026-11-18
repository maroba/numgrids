r"""Linear stability of steady states and the bifurcations along a branch.

A steady state :math:`u^*` of an evolution equation :math:`u_t = F(u)` with
boundary conditions is a solution of the boundary value problem
:math:`F(u^*) = 0`, as :func:`~numgrids.solve_nonlinear` finds it.
A small perturbation of the steady state evolves like :math:`e^{\sigma t}
v(x)`, where :math:`\sigma` and :math:`v` are an eigenvalue and an
eigenfunction of the Jacobian,

.. math::

    F'(u^*) \, v = \sigma \, v ,

and :math:`v` satisfies the homogeneous boundary conditions.  The steady
state is *linearly stable* if all eigenvalues have a negative real part.
:func:`linear_stability` computes the leading eigenvalues, those with the
largest real part.

.. note::

    The sign convention is that of the evolution equation :math:`u_t = +F(u)`.
    A residual and its negative have the same solutions, but opposite
    stability.  For a residual that is written as :math:`-F`, i.e. for the
    evolution :math:`u_t = -F(u)`, pass ``sign=-1``.

Along a branch of steady states (see :func:`~numgrids.continuation`),
the stability changes where eigenvalues cross the imaginary axis.  These
points are the *bifurcations* of the branch: folds, branch points and Hopf
bifurcations, each described by a :class:`Bifurcation`.

Examples
--------
The steady state :math:`u = 0` of :math:`u_t = u'' + a u`,
:math:`u(0) = u(1) = 0`, is stable for :math:`a < \pi^2`:

>>> import numpy as np
>>> from numgrids import Grid, chebyshev
>>> from numgrids.stability import linear_stability
>>> grid = Grid(x=chebyshev(31, 0, 1))
>>> result = linear_stability(grid, lambda u: grid.diff(u, 2) + 5 * u, 0, bcs=0)
>>> result.stable
True
>>> bool(np.allclose(result.eigenvalues[:3], 5 - np.pi ** 2 * np.array([1, 4, 9])))
True
>>> linear_stability(grid, lambda u: grid.diff(u, 2) + 12 * u, 0, bcs=0)
StabilityResult(unstable, 1 unstable mode, growth rate 2.1304)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numgrids.grids import Grid


# Eigenproblems up to this size are solved with dense linear algebra (all eigenvalues),
# larger ones with shift-invert Arnoldi iterations (the eigenvalues near a shift).
_DENSE_EIG_LIMIT = 2000

# Eigenvalues with a relative imaginary part below this are real up to rounding errors.
_IMAG_TOL = 1e-8

# Eigenvalues with a real part up to this size count as zero (neutral modes).
_NEUTRAL = 1e-8

# Rows without time derivative give infinite eigenvalues; rounding makes some of them
# finite but huge.  Everything beyond this multiple of the scale ||A|| / ||B|| is dropped.
_INFINITE = 1e6

# A root between two points of the branch counts as located if it is enclosed within this
# fraction of their distance (the interpolation error is of third order in the distance).
_ENCLOSED = 2e-2
# An eigenvalue jumps (instead of passing through zero) if it changes more than this many
# times as fast as on average between two points of the branch.
_JUMP = 20.0
# ... and by more than this fraction of its change between the two points (a located root
# is enclosed so tightly that only the rounding errors of the eigenvalues remain).
_JUMP_MIN = 1e-3


@dataclass
class StabilityResult:
    r"""Result of :func:`linear_stability`.

    Attributes
    ----------
    eigenvalues : NDArray
        The leading eigenvalues :math:`\sigma`, largest real part first.
        Perturbations grow like :math:`e^{\sigma t}`.  A real array if all
        of them are real.  Complex eigenvalues come in conjugate pairs, the
        one with the positive imaginary part first.
    eigenfunctions : NDArray or tuple of NDArray
        The eigenfunctions, shape ``(k, *grid.shape)``:
        ``eigenfunctions[j]`` belongs to ``eigenvalues[j]``.  For coupled
        systems, a tuple with one such array per field.  They satisfy the
        homogeneous boundary conditions, have unit Euclidean norm (over all
        fields) and a positive largest component.
    stable : bool
        ``True`` if all eigenvalues have a negative real part.
    growth_rate : float
        The largest real part: the rate at which a generic perturbation
        grows in the long run (decays, if negative).
    n_unstable : int
        Number of eigenvalues with a positive real part, among the
        computed ones.
    """

    eigenvalues: NDArray
    eigenfunctions: Any
    stable: bool
    growth_rate: float
    n_unstable: int

    def __repr__(self) -> str:
        if self.stable:
            state = "stable"
        elif self.n_unstable == 0:
            state = "marginal"
        else:
            state = f"unstable, {self.n_unstable} unstable mode{'s' if self.n_unstable > 1 else ''}"
        return f"StabilityResult({state}, growth rate {self.growth_rate:.5g})"


@dataclass
class Bifurcation:
    r"""A bifurcation on a branch of steady states (see :func:`~numgrids.continuation`).

    Attributes
    ----------
    kind : str
        ``"fold"``: the branch turns back in the parameter (turning point,
        saddle-node bifurcation); a real eigenvalue is zero there.
        ``"branch_point"``: a real eigenvalue crosses zero away from a
        fold; another branch of steady states crosses (transcritical or
        pitchfork bifurcation).  ``"hopf"``: a complex conjugate pair of
        eigenvalues crosses the imaginary axis; a branch of oscillations
        starts.
    param : float
        The parameter at the bifurcation.
    index : int
        The bifurcation lies between the points ``index`` and ``index + 1``
        of the branch.
    u : NDArray or tuple of NDArray
        The steady state at the bifurcation.  At a branch point, where two
        branches of solutions cross, it is only determined up to a small
        multiple (about 1e-6) of the critical mode.
    mode : NDArray or tuple of NDArray, optional
        The critical eigenfunction, normalized as in
        :class:`StabilityResult` (complex at a Hopf bifurcation).  At a
        branch point, the new branch leaves along this direction.
    frequency : float
        Angular frequency :math:`\omega` of the oscillation that sets in at
        a Hopf bifurcation (the eigenvalues are :math:`\pm i \omega`); zero
        otherwise.
    multiplicity : int
        Number of eigenvalues that cross together (more than one for
        degenerate eigenvalues, e.g. symmetry-related modes on a square).
    refined : bool
        ``True`` if the bifurcation was located by solving for points on
        the branch: the parameter then is as accurate as the solutions and
        their eigenvalues, whatever the step of the continuation.
        ``False`` if it is only interpolated between the two neighbouring
        points of the branch.
    """

    kind: str
    param: float
    index: int
    u: Any
    mode: Any = None
    frequency: float = 0.0
    multiplicity: int = 1
    refined: bool = True

    def __repr__(self) -> str:
        extra = f", frequency {self.frequency:.6g}" if self.kind == "hopf" else ""
        if self.multiplicity > 1:
            extra += f", multiplicity {self.multiplicity}"
        if not self.refined:
            extra += ", interpolated"
        return (f"Bifurcation({self.kind} at param = {self.param:.10g}, between the points "
                f"{self.index} and {self.index + 1}{extra})")


def linear_stability(
    grid: Grid,
    residual: Callable,
    u,
    bcs=None,
    jacobian: Callable | None = None,
    *,
    k: int = 6,
    mass=None,
    sign: int = 1,
    param: float | None = None,
    tol: float = _NEUTRAL,
    sigma: float = 0.0,
) -> StabilityResult:
    r"""Linear stability of a steady state of :math:`u_t = F(u)`.

    Computes the leading eigenvalues :math:`\sigma` of the Jacobian
    :math:`F'(u)`, restricted to perturbations that satisfy the homogeneous
    boundary conditions.  Perturbations of the steady state grow like
    :math:`e^{\sigma t}`: the state is stable if all eigenvalues have a
    negative real part.

    The arguments follow :func:`~numgrids.solve_nonlinear`, so
    that the residual, the boundary conditions and the Jacobian of the
    solve can be passed on unchanged.

    .. note::

        The eigenvalues belong to the evolution equation :math:`u_t = +F(u)`
        (or :math:`M u_t = F(u)` with *mass*).  If the residual is the
        negative of the right-hand side of the evolution equation, pass
        ``sign=-1``.

    Parameters
    ----------
    grid : Grid
        The grid.
    residual : callable
        ``residual(u)`` returns :math:`F(u)` as an array of grid shape; for
        coupled systems, ``residual(u, v, ...)`` returns a tuple of arrays.
    u : float, NDArray, callable, or tuple of these
        The steady state, e.g. the result of
        :func:`~numgrids.solve_nonlinear`; a tuple for coupled
        systems.  It is not checked that *u* solves :math:`F(u) = 0`.
    bcs : optional
        Boundary conditions, as in :func:`~numgrids.solve_nonlinear`.
        Perturbations satisfy their homogeneous form (e.g. they vanish
        where the solution is prescribed).
    jacobian : callable, optional
        ``jacobian(u)`` (or ``jacobian(u, v, ...)``) returns the Jacobian,
        as in :func:`~numgrids.solve_nonlinear` (a nested list of
        blocks for systems).  If omitted, it is approximated by finite
        differences.
    k : int
        Number of eigenvalues to compute.
    mass : optional
        The operator :math:`M` in front of the time derivative,
        :math:`M u_t = F(u)`; the identity by default.  A scalar or an
        array of grid shape (multiplication), an
        :class:`~numgrids.Operator` or a matrix.  For coupled systems, a
        tuple with one such entry per field (``0`` for an equation without
        time derivative, e.g. an algebraic constraint) or a nested list of
        blocks like the Jacobian.
    sign : {1, -1}
        ``-1`` for the evolution equation :math:`u_t = -F(u)`.
    param : float, optional
        If given, the residual and the Jacobian are called with this
        parameter as their last argument, as in
        :func:`~numgrids.continuation`.
    tol : float
        Eigenvalues with a real part of at most *tol* in absolute value
        count as zero: the state then is neither stable nor is the mode
        counted as unstable.
    sigma : float
        Only for large problems (more than 2000 unknowns), which are
        solved with shift-invert Arnoldi iterations: the eigenvalues
        closest to *sigma* are computed, and the *k* with the largest real
        part are returned.  The default 0 finds the eigenvalues that decide
        on stability unless some lie far from the origin.

    Returns
    -------
    StabilityResult
        The eigenvalues (largest real part first), the eigenfunctions,
        ``stable``, ``growth_rate`` and ``n_unstable``.

    Examples
    --------
    The Bratu problem :math:`u_t = u'' + \lambda e^u`, :math:`u(0) = u(1) = 0`
    has two steady states for :math:`\lambda = 1`.  The small one is
    stable, the large one unstable:

    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev, solve_nonlinear
    >>> from numgrids.stability import linear_stability
    >>> grid = Grid(x=chebyshev(31, 0, 1))
    >>> x = grid.coords
    >>> F = lambda u: grid.diff(u, 2) + np.exp(u)
    >>> lower = solve_nonlinear(grid, F, 0, bcs=0)
    >>> upper = solve_nonlinear(grid, F, 4 * np.sin(np.pi * x), bcs=0)
    >>> linear_stability(grid, F, lower, bcs=0)
    StabilityResult(stable, growth rate -8.7397)
    >>> result = linear_stability(grid, F, upper, bcs=0)
    >>> result.stable, result.n_unstable
    (False, 1)
    >>> result.eigenfunctions.shape
    (6, 31)
    """
    from numgrids.nonlinear import _Problem

    problem = _Problem(grid, residual, u, bcs, jacobian)
    if np.iscomplexobj(problem.y0):
        raise TypeError("linear_stability needs real-valued fields.")
    linearization = _Linearization(problem, k, mass, sign, sigma)
    args = () if param is None else (param,)
    vals, vecs = linearization.leading(problem.y0, args, vectors=True)
    if np.all(vals.imag == 0):
        vals, vecs = vals.real, vecs.real
    growth = float(np.max(vals.real)) if len(vals) else -np.inf
    return StabilityResult(
        eigenvalues=vals,
        eigenfunctions=linearization.fields(vecs),
        stable=growth < -tol,
        growth_rate=growth,
        n_unstable=int(np.sum(vals.real > tol)),
    )


class _Linearization:
    r"""The eigenproblem :math:`J v = \sigma M v` of the perturbations of a steady state.

    *J* is the Jacobian of a :class:`~numgrids.nonlinear._Problem`.  The
    perturbations satisfy the homogeneous boundary conditions ``R v = 0``.
    For moderate sizes, these determine the boundary values in terms of
    the interior ones, ``v_b = G v_i``, which leaves an ordinary
    eigenproblem for the interior values.
    """

    def __init__(self, problem, k: int, mass=None, sign: int = 1, sigma: float = 0.0) -> None:
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
            raise ValueError(f"The number of eigenvalues must be a positive integer, got {k!r}.")
        if sign not in (1, -1):
            raise ValueError(f"sign must be 1 or -1, got {sign!r}.")
        self.problem = problem
        self.k = int(k)
        self.sign = sign
        self.sigma = float(sigma)
        self.n = n = problem.y0.size
        self.b = b = np.flatnonzero(problem.bc_rows)
        self.i = i = np.flatnonzero(~problem.bc_rows)
        M = _mass_matrix(problem, mass)
        self.G = None
        self.B = None
        self.row_scale = None
        self.dense = n <= _DENSE_EIG_LIMIT
        if not self.dense:
            # Rows of boundary conditions have no time derivative.
            self.B = scipy.sparse.csr_matrix(problem.keep_rows if M is None else problem.keep_rows @ M)
            return
        Rb = problem.R[b].toarray()
        if len(b) == 0 or np.linalg.cond(Rb[:, b]) < 1e10:
            self.G = G = -np.linalg.solve(Rb[:, b], Rb[:, i]) if len(b) else np.zeros((0, n))
            if M is not None:
                Md = M.toarray()
                B = Md[np.ix_(i, i)] + Md[np.ix_(i, b)] @ G
                d = np.diag(B)
                if np.all(d != 0) and not np.any(B - np.diag(d)):
                    self.row_scale = 1 / d
                else:
                    self.B = B
        else:
            # The conditions do not determine the boundary values by themselves:
            # keep them as rows without time derivative.
            self.B = (problem.keep_rows if M is None else problem.keep_rows @ M).toarray()

    def leading(self, y: NDArray, args: tuple = (), vectors: bool = False):
        """The *k* eigenvalues with the largest real part at the state *y*.

        Returns the eigenvalues as a complex array, largest real part first
        (of a conjugate pair, the one with positive imaginary part first),
        and the eigenvectors as columns of an array (``None`` unless
        *vectors*).
        """
        problem = self.problem
        J = problem.jacobian(y, problem.F(y, *args), *args, accurate=True)
        vals, vecs = self._dense(J, vectors) if self.dense else self._arnoldi(J, vectors)
        vals = self.sign * np.asarray(vals, dtype=complex)
        real = np.abs(vals.imag) <= _IMAG_TOL * np.maximum(1.0, np.abs(vals))
        vals = np.where(real, vals.real, vals)
        order = np.lexsort((-vals.imag, -vals.real))[:self.k]
        vals = vals[order]
        if not vectors:
            return vals, None

        from numgrids.operators import _real_eigenbasis

        vals, vecs = _real_eigenbasis(vals, vecs[:, order], rtol=_IMAG_TOL)
        if self.G is not None:
            full = np.zeros((self.n, len(vals)), dtype=complex)
            full[self.i] = vecs
            full[self.b] = self.G @ vecs
        else:
            full = np.array(vecs, dtype=complex)
        for j in range(len(vals)):
            full[:, j] = _normalized(full[:, j])
        return vals, full

    def fields(self, vecs: NDArray):
        """Eigenvectors (columns) as arrays ``(k, *grid.shape)``, one per field."""
        problem = self.problem
        size, shape = problem.grid.size, problem.grid.shape
        parts = [np.moveaxis(vecs[f * size:(f + 1) * size], 1, 0).reshape(-1, *shape)
                 for f in range(problem.nfields)]
        return tuple(parts) if problem.system else parts[0]

    def _dense(self, J, vectors: bool):
        J = J.toarray() if scipy.sparse.issparse(J) else np.asarray(J)
        if self.G is not None:
            A = J[np.ix_(self.i, self.i)]
            if len(self.b):
                A = A + J[np.ix_(self.i, self.b)] @ self.G
            if self.row_scale is not None:
                A = self.row_scale[:, None] * A
        else:
            A = J
        if self.B is None:
            out = scipy.linalg.eig(A, right=vectors, check_finite=False)
            return out if vectors else (out, None)
        out = scipy.linalg.eig(A, self.B, right=vectors, homogeneous_eigvals=True, check_finite=False)
        (alpha, beta), vecs = out if vectors else (out, None)
        scale = np.linalg.norm(A) / np.linalg.norm(self.B)
        finite = np.abs(alpha) < _INFINITE * scale * np.abs(beta)
        vals = alpha[finite] / beta[finite]
        return vals, (vecs[:, finite] if vectors else None)

    def _arnoldi(self, J, vectors: bool):
        from numgrids.nonlinear import _factorize

        n, B = self.n, self.B
        sigma = self.sign * self.sigma
        # A fixed (pseudo-random) start vector makes the results reproducible.
        v0 = np.random.default_rng(20231117).standard_normal(n)
        # A shift (nearly) on an eigenvalue, as at every bifurcation with the default shift,
        # costs accuracy: the errors of the other eigenvalues grow like 1e-16 |J| / gap.
        # The solution of one system estimates the gap; if it is too small, the shift moves.
        norm = scipy.sparse.linalg.norm(J, 1) if scipy.sparse.issparse(J) else np.linalg.norm(J, 1)
        gap = 1e-8 * norm
        solve = None
        for shift in (sigma, sigma + 10 * gap):
            try:
                if scipy.sparse.issparse(J):
                    candidate = _factorize(scipy.sparse.csc_matrix(J - shift * B))
                else:
                    candidate = _factorize(J - shift * B.toarray() if shift else J)
                x = candidate(B @ v0)
            except np.linalg.LinAlgError:
                continue
            solve = candidate
            if np.linalg.norm(x) * gap <= np.linalg.norm(v0):
                break
        if solve is None:
            raise np.linalg.LinAlgError("The shifted Jacobian is singular; choose another shift sigma.")
        # More eigenpairs than requested: the leading ones are selected among them, and
        # Arnoldi iterations can miss members of clusters at the edge of the set.
        k = min(n - 2, max(2 * self.k, self.k + 10))
        inverse = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda x: solve(B @ x), dtype=float)
        try:
            mu, vecs = scipy.sparse.linalg.eigs(inverse, k=k, which="LM", v0=v0)
        except scipy.sparse.linalg.ArpackError as err:
            raise np.linalg.LinAlgError(f"The Arnoldi iteration failed: {err}") from err
        found = mu != 0
        return shift + 1 / mu[found], (vecs[:, found] if vectors else None)


def _normalized(v: NDArray) -> NDArray:
    """Scale a vector to unit Euclidean norm, with a real and positive largest component."""
    size = np.abs(v)
    if not np.any(size):
        return v
    # Of several components of (nearly) the same size, the first one: rounding must not
    # decide on the sign of, e.g., an antisymmetric mode.
    pivot = v[np.flatnonzero(size >= (1 - 1e-6) * size.max())[0]]
    return v / np.linalg.norm(v) * (np.abs(pivot) / pivot)


def _mass_matrix(problem, mass):
    """Sparse matrix of the operator in front of the time derivative; ``None`` for the identity."""
    from numgrids.nonlinear import _assemble_jacobian

    if mass is None:
        return None
    grid, nfields = problem.grid, problem.nfields
    n = nfields * grid.size
    if scipy.sparse.issparse(mass) or (isinstance(mass, np.ndarray) and mass.ndim == 2
                                       and mass.shape == (n, n) and mass.shape != grid.shape):
        M = scipy.sparse.csr_matrix(mass)
        if M.shape != (n, n):
            raise ValueError(f"mass has shape {M.shape}, expected {(n, n)}.")
        return M
    if nfields > 1 and isinstance(mass, (list, tuple)):
        if len(mass) != nfields:
            raise ValueError(f"mass must have one entry per field ({nfields}), got {len(mass)}.")
        if not all(isinstance(row, (list, tuple)) for row in mass):
            # One entry per field: the diagonal blocks.
            mass = [[m if f == g else None for g in range(nfields)] for f, m in enumerate(mass)]
    elif nfields > 1:
        mass = [[mass if f == g else None for g in range(nfields)] for f in range(nfields)]
    try:
        return scipy.sparse.csr_matrix(_assemble_jacobian(grid, mass, nfields))
    except (TypeError, ValueError) as err:
        raise type(err)(f"Cannot use this as mass: {err}") from err


# ---------------------------------------------------------------------------
# Bifurcations along a branch
# ---------------------------------------------------------------------------

def _turning_points(params: NDArray) -> tuple[NDArray, NDArray]:
    """Where the parameter turns around along a branch.

    Returns two index arrays ``(before, folds)``: for each turning point,
    the parameter moves in one direction from the point ``before`` and in
    the other direction from the point ``folds`` on.
    """
    d = np.diff(params)
    # Steps without a significant change of the parameter (vertical parts of a
    # branch) carry no direction, only rounding noise.
    steps = np.flatnonzero(np.abs(d) > 1e-9 * (1 + np.abs(params[:-1])))
    signs = np.sign(d[steps])
    turn = np.flatnonzero(signs[1:] != signs[:-1])
    return steps[turn], steps[turn + 1]


def _parabola_extremum(s: NDArray, lam: NDArray) -> float:
    """Extremum of the parabola through three points ``(s, lam)``."""
    c = np.polyfit(np.asarray(s) - s[1], lam, 2)
    if c[0] == 0:
        return float(lam[1])
    return float(c[2] - c[1] ** 2 / (4 * c[0]))


def _assign(a: NDArray, b: NDArray) -> list[tuple[int, int]]:
    """Assign the eigenvalues *a* at one point of a branch to those at a neighbouring point, *b*.

    Returns pairs of indices ``(j, l)``: ``b[l]`` continues ``a[j]``.  The
    assignment minimizes the sum of the *squared* distances.  With plain
    distances, real eigenvalues would have many equally good assignments
    (every assignment that preserves their order, and others), and
    rounding errors would pick one of them.  The squared distances have
    one minimum: real eigenvalues are assigned in their order.
    """
    ja, jb = np.flatnonzero(np.isfinite(a)), np.flatnonzero(np.isfinite(b))
    if len(ja) == 0 or len(jb) == 0:
        return []
    rows, cols = scipy.optimize.linear_sum_assignment(np.abs(a[ja][:, None] - b[jb][None, :]) ** 2)
    return [(int(j), int(l)) for j, l in zip(ja[rows], jb[cols])]


def _crossings(a: NDArray, b: NDArray) -> list[tuple[int, int, bool]]:
    """Eigenvalues that cross the imaginary axis between two neighbouring points of a branch.

    *a* and *b* are the leading eigenvalues at the two points, assigned to
    each other by :func:`_assign`.  Returns ``(j, l, hopf)`` for each pair
    ``a[j]``, ``b[l]`` with real parts of opposite sign; *hopf* tells if
    the pair is complex.  Of a conjugate pair, only the member with the
    positive imaginary part is returned.
    """
    found = []
    for j, l in _assign(a, b):
        za, zb = a[j], b[l]
        if (za.real > 0) == (zb.real > 0):
            continue
        # Where real eigenvalues merge into a pair, the end closer to the axis decides.
        near = za if abs(za.real) <= abs(zb.real) else zb
        hopf = bool(near.imag != 0)
        if hopf and (za.imag + zb.imag < 0 or near.imag < 0):
            continue
        found.append((j, l, hopf))
    return found


def _follow(a: NDArray, j: int, b: NDArray) -> int:
    """Index of the eigenvalue in *b* that continues ``a[j]``, assigned as in :func:`_crossings`."""
    for row, col in _assign(a, b):
        if row == j:
            return col
    jb = np.flatnonzero(np.isfinite(b))
    return int(jb[np.argmin(np.abs(b[jb] - a[j]))])


@dataclass
class _Sample:
    """A point of the branch between two of its neighbouring points.

    *s* is the position along the tangent at the first of the two points,
    *f* the value of the function whose root is searched, *data* whatever
    the function needs to remember (the eigenvalues at the point and the
    index of the one that it follows).
    """

    s: float
    y: NDArray
    lam: float
    ty: NDArray | None
    tl: float | None
    f: float = 0.0
    data: Any = None


def _root_on_branch(first: _Sample, last: _Sample, step: Callable, dot: Callable, value: Callable,
                    xtol: float = 1e-8, maxiter: int = 12) -> tuple[bool, _Sample, _Sample, list[_Sample]]:
    """Enclose the root of a function along the branch between two neighbouring points.

    A bracketing secant method (Illinois variant).  Each new point is
    computed with a step from the closest known one, so that the steps
    get shorter while the iteration converges.

    Next to a branch point, the points of the branch are poorly
    determined (the Jacobian is singular there), and the corrector does
    not converge.  The search then encloses the root between two points
    at a small distance on either side, which is as good: the function is
    smooth, and :func:`_root_point` interpolates between them.

    Parameters
    ----------
    first, last : _Sample
        The two points, with function values of opposite sign.
    step : callable
        ``step(y, lam, ty, tl, h)`` returns the point ``(y, lam, ty, tl)``
        of the branch at the pseudo-arclength *h* from the given one, or
        ``None``.
    dot : callable
        The scalar product of the arclength.
    value : callable
        ``value(sample, samples)`` returns the function value (and may
        set ``sample.data``); *samples* are the known points, sorted.
    xtol : float
        Tolerance for the position of the root, relative to the distance
        of the two points.

    Returns
    -------
    converged : bool
        Whether the root is enclosed tightly.
    lo, hi : _Sample
        The closest known points on either side of the root.
    samples : list of _Sample
        All points, sorted by position.
    """
    samples = [first, last]
    width = last.s - first.s
    lo, hi = first, last
    f_lo, f_hi = lo.f, hi.f  # the values that the secant uses (Illinois: halved for a stagnating end)
    side = 0

    def evaluate(s):
        """The point of the branch at the position *s*, if it lies inside the bracket."""
        if not lo.s < s < hi.s:
            return None
        # Start from the closest points first.
        for near in sorted((p for p in samples if p.ty is not None), key=lambda p: abs(p.s - s))[:2]:
            cos = dot(first.ty, first.tl, near.ty, near.tl)
            if not cos > 0:
                continue
            point = step(near.y, near.lam, near.ty, near.tl, (s - near.s) / cos)
            if point is None:
                continue
            y, lam, ty, tl = point
            new = _Sample(dot(first.ty, first.tl, y - first.y, lam - first.lam), y, lam, ty, tl)
            if not lo.s < new.s < hi.s:
                continue
            try:
                new.f = value(new, samples)
            except np.linalg.LinAlgError:
                continue
            samples.append(new)
            samples.sort(key=lambda p: p.s)
            return new
        return None

    def narrow(new):
        """Replace the end of the bracket on the side of the new point."""
        nonlocal lo, hi, f_lo, f_hi, side
        if (new.f > 0) == (hi.f > 0):
            hi, f_hi = new, new.f
            if side == 1:
                f_lo /= 2
            side = 1
        else:
            lo, f_lo = new, new.f
            if side == -1:
                f_hi /= 2
            side = -1

    previous = None
    blocked = False
    converged = lo.f == 0 or hi.f == 0
    for _ in range(maxiter):
        if converged or f_lo == f_hi or hi.s - lo.s <= xtol * width:
            break
        s = (lo.s * f_hi - hi.s * f_lo) / (f_hi - f_lo)
        new = evaluate(s)
        if new is not None:
            converged = new.f == 0 or (previous is not None and abs(new.s - previous) <= xtol * width)
            previous = new.s
            narrow(new)
            continue
        # No point of the branch can be computed there: the root must be close.  Enclose
        # it between points at a small distance on either side.
        if blocked:
            break
        blocked = True
        for direction in (-1, 1):
            for distance in (1e-3, 4e-3, 1.6e-2):
                new = evaluate(s + direction * distance * width)
                if new is not None:
                    narrow(new)
                    break
        if hi.s - lo.s <= _ENCLOSED * width:
            break
    # Neither end of the bracket is close to a root, and the bracket is still wide: the
    # secant steps crept towards a jump of the function from one side (or ran out of
    # iterations).  Bisection encloses the sign change, so that the caller can tell a
    # jump from a root.
    scale = max(abs(first.f), abs(last.f))
    for _ in range(maxiter):
        if hi.s - lo.s <= _ENCLOSED * width / 2 or min(abs(lo.f), abs(hi.f)) <= 1e-3 * scale:
            break
        new = evaluate((lo.s + hi.s) / 2)
        if new is None:
            break
        narrow(new)
    return converged or hi.s - lo.s <= _ENCLOSED * width, lo, hi, samples


def _root_point(lo: _Sample, hi: _Sample, samples: list[_Sample], first: _Sample, dot: Callable):
    """The root between two samples that enclose it.

    Returns its position, the solution and the parameter there, and the
    fraction of the way from *lo* to *hi*.
    """
    width = hi.s - lo.s
    theta = lo.f / (lo.f - hi.f) if lo.f != hi.f else 0.5
    # Inverse quadratic interpolation with the next sample improves on the secant.
    others = [p for p in samples if p is not lo and p is not hi]
    if others and width > 0:
        third = min(others, key=lambda p: min(abs(p.s - lo.s), abs(p.s - hi.s)))
        f0, f1, f2 = lo.f, hi.f, third.f
        if f2 not in (f0, f1) and f0 != f1 and (f2 - f0) * (f2 - f1) > 0:
            s = (lo.s * f1 * f2 / ((f0 - f1) * (f0 - f2)) + hi.s * f0 * f2 / ((f1 - f0) * (f1 - f2))
                 + third.s * f0 * f1 / ((f2 - f0) * (f2 - f1)))
            if lo.s <= s <= hi.s:
                theta = (s - lo.s) / width
    s = lo.s + theta * width
    y = lo.y + theta * (hi.y - lo.y)
    lam = lo.lam + theta * (hi.lam - lo.lam)
    if lo.ty is not None and hi.ty is not None:
        # Cubic Hermite interpolation: the tangents are the derivatives along the branch.
        c0 = width * (theta ** 3 - 2 * theta ** 2 + theta) / dot(first.ty, first.tl, lo.ty, lo.tl)
        c1 = width * (theta ** 3 - theta ** 2) / dot(first.ty, first.tl, hi.ty, hi.tl)
        bend = 3 * theta ** 2 - 2 * theta ** 3 - theta  # Hermite minus linear weight of the second point
        y = y + bend * (hi.y - lo.y) + c0 * lo.ty + c1 * hi.ty
        lam = lam + bend * (hi.lam - lo.lam) + c0 * lo.tl + c1 * hi.tl
    return s, y, float(lam), theta


def _analyze_branch(problem, ys, lams, tangents, eigenvalues, step, dot, linearization,
                    refine: bool, switched: bool) -> list[Bifurcation]:
    """Locate the bifurcations along a traced branch.

    Parameters
    ----------
    problem : _Problem
        The boundary value problem.
    ys, lams : list
        The points of the branch (stacked solution vectors, parameters).
    tangents : list
        Unit tangent ``(ty, tl)`` of the branch at each point, oriented
        along the branch (``None`` where unknown).
    eigenvalues : NDArray, optional
        The leading eigenvalues at the points, shape ``(n, k)``, complex.
    step : callable
        ``step(y, lam, ty, tl, h)`` returns the point ``(y, lam, ty, tl)``
        of the branch at the pseudo-arclength *h* from the given point
        with the given tangent, or ``None`` if it cannot be computed.
    dot : callable
        The scalar product of the arclength.
    linearization : _Linearization, optional
        Computes eigenvalues at new points.
    refine : bool
        Locate the bifurcations by solving for points on the branch.
    switched : bool
        The branch starts at a branch point (which is not reported again).
    """
    lams = np.asarray(lams)
    found: list[tuple[float, Bifurcation]] = []

    def chord(a: int) -> float:
        dy, dl = ys[a + 1] - ys[a], lams[a + 1] - lams[a]
        return np.sqrt(dot(dy, dl, dy, dl))

    def followed(p: _Sample) -> complex:
        """The eigenvalue that is followed through its crossing, at a sample."""
        vals, index = p.data
        return vals[index]

    def ends(a: int) -> tuple[_Sample, _Sample]:
        """The points *a* and *a + 1* as samples; positions along the tangent at *a*."""
        ty, tl = tangents[a] or (None, None)
        H = chord(a) if ty is None else dot(ty, tl, ys[a + 1] - ys[a], lams[a + 1] - lams[a])
        return (_Sample(0.0, ys[a], lams[a], ty, tl),
                _Sample(H, ys[a + 1], lams[a + 1], *(tangents[a + 1] or (None, None))))

    # --- Folds: the tangent of the branch has no component along the parameter.
    fold_at: dict[int, Bifurcation] = {}
    for before, k in zip(*_turning_points(lams)):
        fold = None
        for m in range(before, k + 1):
            if not refine or tangents[m] is None or tangents[m + 1] is None:
                continue
            first, last = ends(m)
            first.f, last.f = first.tl, last.tl
            if first.f * last.f >= 0 or not last.s > 0:
                continue
            converged, lo, hi, samples = _root_on_branch(first, last, step, dot, lambda p, _: p.tl)
            if not converged:
                continue
            s, y, lam, _ = _root_point(lo, hi, samples, first, dot)
            near = lo if abs(lo.f) <= abs(hi.f) else hi
            # At a fold, the tangent is the eigenvector of the zero eigenvalue.
            fold = Bifurcation("fold", lam, m, problem.output(y), mode=problem.output(_normalized(near.ty)))
            fold_at[m] = fold
            found.append((m + s / last.s, fold))
            break
        if fold is None:
            # Estimate: the extremum of a parabola through the three points around the fold.
            lam = _parabola_extremum(np.cumsum([0.0, chord(k - 1), chord(k)]), lams[k - 1:k + 2])
            fold = Bifurcation("fold", lam, int(k - 1), problem.output(ys[k]), refined=False)
            fold_at[k - 1] = fold_at[k] = fold
            found.append((float(k), fold))

    # --- Eigenvalues that cross the imaginary axis.
    if eigenvalues is not None:
        claimed: set[int] = set()
        for i in range(1 if switched else 0, len(lams) - 1):
            a, b = eigenvalues[i], eigenvalues[i + 1]
            located: list[tuple[complex, complex, Bifurcation]] = []
            for j, l, hopf in _crossings(a, b):
                za, zb = a[j], b[l]
                # A degenerate eigenvalue crosses with all its members at once.
                twin = next((bif for wa, wb, bif in located if (bif.kind == "hopf") == hopf
                             and abs(wa - za) <= 1e-6 * (1 + abs(za))
                             and abs(wb - zb) <= 1e-6 * (1 + abs(zb))), None)
                if twin is not None:
                    twin.multiplicity += 1
                    continue
                if not hopf:
                    # The eigenvalue that crosses zero at a fold belongs to the fold.  Rounding
                    # may move its crossing into the interval next to that of the fold.
                    fold = fold_at.get(i)
                    if fold is None and abs(za.real) <= 1e-3 * abs(zb.real):
                        fold = fold_at.get(i - 1)
                    if fold is None and abs(zb.real) <= 1e-3 * abs(za.real):
                        fold = fold_at.get(i + 1)
                    if fold is not None and id(fold) not in claimed:
                        claimed.add(id(fold))
                        located.append((za, zb, fold))
                        continue
                first, last = ends(i)
                first.f, first.data, last.f, last.data = za.real, (a, j), zb.real, (b, l)

                def real_part(p, samples):
                    vals, _ = linearization.leading(p.y, (p.lam,))
                    # Follow the eigenvalue from the neighbouring samples on both sides.  If
                    # they disagree, the closer continuation counts.
                    left = max((q for q in samples if q.s < p.s), key=lambda q: q.s)
                    right = min((q for q in samples if q.s > p.s), key=lambda q: q.s)
                    index = min((_follow(*q.data, vals) for q in (left, right)),
                                key=lambda m: min(abs(vals[m] - followed(left)), abs(vals[m] - followed(right))))
                    p.data = (vals, index)
                    return vals[index].real

                converged, lo, hi, samples = False, first, last, [first, last]
                if refine and first.ty is not None and last.s > 0:
                    converged, lo, hi, samples = _root_on_branch(first, last, step, dot, real_part)
                    width = (hi.s - lo.s) / (last.s - first.s)
                    if (len(samples) > 2 and width <= _ENCLOSED
                            and abs(hi.f - lo.f) > max(_JUMP * width, _JUMP_MIN) * abs(zb.real - za.real)):
                        # The real part jumps instead of passing through zero: over the short
                        # distance that encloses the sign change, it changes far more than a
                        # continuous eigenvalue would.  These are two different eigenvalues.
                        # The set of computed eigenvalues has changed (large problems: the
                        # eigenvalues closest to the shift), nothing crossed.
                        continue
                else:
                    first.ty = last.ty = None  # plain linear interpolation
                s, y, lam, theta = _root_point(lo, hi, samples, first, dot)
                z = followed(lo) + theta * (followed(hi) - followed(lo))
                mode = None
                if converged:
                    try:
                        vals, vecs = linearization.leading(y, (lam,), vectors=True)
                        mode = vecs[:, np.nanargmin(np.abs(vals - z))]
                        mode = problem.output(mode if hopf else mode.real)
                    except np.linalg.LinAlgError:
                        mode = None
                bif = Bifurcation("hopf" if hopf else "branch_point", lam, i, problem.output(y), mode=mode,
                                  frequency=abs(float(z.imag)) if hopf else 0.0, refined=converged)
                located.append((za, zb, bif))
                found.append((i + s / last.s if last.s > 0 else i + theta, bif))

    return [bif for _, bif in sorted(found, key=lambda item: item[0])]

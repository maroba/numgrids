"""Time integration of grid-based PDEs (method of lines).

:func:`evolve` wraps :func:`scipy.integrate.solve_ivp` so that the
right-hand side can be written in terms of meshed arrays, optionally with
several coupled fields and Dirichlet, Neumann or Robin boundary conditions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from numgrids.grids import Grid

_IMPLICIT = ("Radau", "BDF", "LSODA")
# Implicit methods get a dense Jacobian up to this size, if the matrix is at least this dense.
_JAC_DENSE_LIMIT = 4000
_JAC_DENSE_DENSITY = 0.01
# Up to this many unknowns, the exponential integrator uses dense matrix exponentials.
_EXPM_DENSE_LIMIT = 2000


@dataclass
class EvolutionResult:
    """Result of :func:`evolve`.

    Attributes
    ----------
    t : NDArray
        Time points, shape ``(nt,)``.
    u : NDArray or tuple of NDArray
        The solution at the time points, shape ``(nt, *grid.shape)``.  For
        coupled systems, a tuple with one such array per field.
    success : bool
        Whether the integration succeeded.
    message : str
        Status message of the solver.
    sol : object
        The raw result of :func:`scipy.integrate.solve_ivp` (``None`` for
        ``method="expm"``).
    grid : Grid
        The grid of the solution, used by :meth:`animate` and :meth:`to_vtk`.
    """

    t: NDArray
    u: Any
    success: bool
    message: str
    sol: Any = field(repr=False, default=None)
    grid: Any = field(repr=False, default=None)

    @property
    def final(self):
        """The solution at the last time point (array, or tuple for systems)."""
        if isinstance(self.u, tuple):
            return tuple(ui[-1] for ui in self.u)
        return self.u[-1]

    def _require_grid(self) -> Grid:
        if self.grid is None:
            raise ValueError(
                "This result does not know its grid. Set `result.grid`, or use grid.animate(...) "
                "and grid.to_vtk_series(...) directly."
            )
        return self.grid

    def _field(self, field: int) -> NDArray:
        """The frames of one field (the field with index *field* of a coupled system)."""
        if not isinstance(self.u, tuple):
            if field != 0:
                raise IndexError(f"The solution is a single field, so `field` must be 0, got {field!r}.")
            return self.u
        is_index = isinstance(field, (int, np.integer)) and not isinstance(field, bool)
        if not is_index or not -len(self.u) <= field < len(self.u):
            raise IndexError(
                f"`field` must be the index of one of the {len(self.u)} fields of the system, got {field!r}."
            )
        return self.u[field]

    def animate(self, field: int = 0, **options):
        """Animate the solution in time (requires matplotlib).

        Parameters
        ----------
        field : int
            For coupled systems: the index of the field to animate
            (default 0, the first one).
        **options
            Options of :func:`numgrids.plots.animate_field`, e.g.
            ``fps``, ``vmin``, ``vmax``, ``cmap``, ``title``, ``part="abs"``
            for complex fields or ``save="movie.gif"``.

        Returns
        -------
        matplotlib.animation.FuncAnimation

        Examples
        --------
        >>> import numpy as np
        >>> from numgrids import Grid, chebyshev, evolve
        >>> grid = Grid(x=chebyshev(24, 0, 1))
        >>> res = evolve(grid, 0.1 * grid.laplacian_operator(), np.sin(np.pi * grid.coords), (0, 1),
        ...              bcs=0, method="BDF", t_eval=np.linspace(0, 1, 21))
        >>> anim = res.animate()                 # doctest: +SKIP
        >>> anim.save("heat.gif")                # doctest: +SKIP
        """
        from numgrids.plots import animate_field
        return animate_field(self._require_grid(), self._field(field), self.t, **options)

    def to_vtk(self, path, name: str | Sequence[str] = "u", **options):
        """Write the solution as a time series of VTK files for ParaView.

        One ``.vts`` file per time point plus a ``.pvd`` collection file
        with the time values, see :func:`numgrids.vtk.write_vtk_series`.

        Parameters
        ----------
        path : str or Path
            The collection file (suffix ``.pvd``).
        name : str or sequence of str
            Name of the field in the files (default ``"u"``).  For coupled
            systems, one name per field; a single name is numbered
            (``"u0"``, ``"u1"``, ...).
        **options
            ``binary`` and ``close``, see :func:`numgrids.vtk.write_vtk`.

        Returns
        -------
        pathlib.Path
            The path of the collection file.
        """
        from numgrids.vtk import _write_series
        grid = self._require_grid()
        if not isinstance(self.u, tuple):
            if not isinstance(name, str):
                raise ValueError(f"The solution is a single field and needs a single name, got {name!r}.")
            fields = {name: self.u}
        else:
            names = [f"{name}{i}" for i in range(len(self.u))] if isinstance(name, str) else list(name)
            if len(names) != len(self.u) or len(set(names)) != len(names):
                raise ValueError(f"Expected {len(self.u)} different names (one per field), got {names}.")
            fields = dict(zip(names, self.u))
        return _write_series(path, grid, fields, self.t, **options)


class _BoundaryConstraint:
    r"""Boundary conditions of one field, as an affine constraint.

    The conditions on the boundary points *b* read
    :math:`R_{bb} u_b + R_{bi} u_i = g` (identity rows for Dirichlet,
    normal-derivative rows for Neumann and Robin conditions).  Solving for
    :math:`u_b` expresses the boundary values through the interior values:
    :math:`u_b = S u_i + c`, so the time integration only evolves the
    interior and the conditions hold at every instant.
    """

    def __init__(self, grid: Grid, bc_list, t0: float = 0.0) -> None:
        from numgrids.boundary import apply_bcs

        size = grid.size
        rows = np.zeros(size, dtype=bool)
        for bc in bc_list:
            rows[bc.face.flat_indices] = True
        self.bc_list = bc_list
        #: whether the boundary values (not the type of the conditions) change in time
        self.time_dependent = any(bc.time_dependent for bc in bc_list)
        frozen = [bc.at_time(t0) for bc in bc_list]
        R, g = apply_bcs(frozen, scipy.sparse.csr_matrix((size, size)), np.zeros(size))
        R = scipy.sparse.csr_matrix(R)
        b = np.flatnonzero(rows)
        i = np.flatnonzero(~rows)
        R_bb = scipy.sparse.csc_matrix(R[b][:, b])
        R_bi = scipy.sparse.csc_matrix(R[b][:, i])
        try:
            lu = self._lu = scipy.sparse.linalg.splu(R_bb)
            S = -lu.solve(R_bi.toarray()) if R_bi.nnz else np.zeros((len(b), len(i)))
            c = lu.solve(np.asarray(g[b]))
        except RuntimeError as err:
            raise ValueError(
                "The boundary conditions do not determine the boundary values "
                "(singular boundary system)."
            ) from err
        if not (np.all(np.isfinite(S)) and np.all(np.isfinite(c))):
            raise ValueError(
                "The boundary conditions do not determine the boundary values "
                "(singular boundary system)."
            )
        self.rows = rows
        self.b, self.i = b, i
        self.S = scipy.sparse.csr_matrix(S)
        self.c = c
        self.size = size

    def values(self, t: float | None = None) -> NDArray:
        """The constants *c* of ``u_b = S u_i + c`` at time *t*."""
        if not self.time_dependent or t is None:
            return self.c
        g = np.zeros(self.size, dtype=self.c.dtype)
        for bc in self.bc_list:  # later conditions win on shared points, as in apply_bcs
            value = bc.at_time(t)._resolve_value()
            g = g.astype(np.result_type(g, value), copy=False)
            g[bc.face.flat_indices] = value
        return self._lu.solve(g[self.b])

    def project(self, y: NDArray, t: float | None = None) -> NDArray:
        """Overwrite the boundary values of *y* (flat) so that the conditions hold at time *t*."""
        c = self.values(t)
        y = np.array(y, dtype=np.result_type(y, c), copy=True)
        y[self.b] = self.S @ y[self.i] + c
        return y

    def matrix(self) -> scipy.sparse.csr_matrix:
        """*P* with ``project(y, t) == P @ y + offset(t)`` for flat arrays *y*."""
        n, nb, ni = self.size, len(self.b), len(self.i)
        E_b = scipy.sparse.csr_matrix((np.ones(nb), (self.b, np.arange(nb))), shape=(n, nb))
        E_i = scipy.sparse.csr_matrix((np.ones(ni), (self.i, np.arange(ni))), shape=(n, ni))
        return scipy.sparse.csr_matrix(E_i @ E_i.T + E_b @ self.S @ E_i.T)

    def offset(self, t: float | None = None) -> NDArray:
        """*d(t)* with ``project(y, t) == P @ y + d(t)``."""
        c = self.values(t)
        d = np.zeros(self.size, dtype=c.dtype)
        d[self.b] = c
        return d


def evolve(
    grid: Grid,
    rhs,
    u0,
    t_span: tuple[float, float],
    t_eval: Sequence[float] | NDArray | None = None,
    bcs=None,
    method: str = "RK45",
    source=None,
    **kwargs,
) -> EvolutionResult:
    r"""Integrate :math:`\partial_t u = F(t, u)` in time on a grid.

    Parameters
    ----------
    grid : Grid
        The grid.
    rhs : callable, Operator or nested list of operators
        The right-hand side.  Either a callable ``rhs(t, u)`` returning
        an array of shape ``grid.shape``, or -- for coupled systems with
        initial state ``(u0, v0, ...)`` -- ``rhs(t, u, v, ...)`` returning
        a tuple of arrays.  For linear problems an
        :class:`~numgrids.operators.Operator` ``L`` can be passed directly,
        meaning :math:`\partial_t u = L u` (its sparse matrix is used as
        Jacobian for implicit methods).  Linear systems are given as a
        nested list of blocks, ``rhs[i][j]`` acting on field *j* in the
        equation of field *i*; a block is an Operator, a matrix, an array
        of grid shape (multiplication), a scalar or ``None`` (zero).  The
        wave equation :math:`u_t = v,\ v_t = \nabla^2 u` is
        ``[[None, 1], [L, None]]``.
    u0 : NDArray or tuple of NDArray
        Initial state (one array, or a tuple of arrays for coupled fields).
        Complex arrays are supported by explicit methods.
    t_span : (float, float)
        Start and end time.
    t_eval : array_like, optional
        Times at which to store the solution.  Defaults to the solver's
        internal time steps.
    bcs : optional
        Boundary conditions, in any form accepted by
        :func:`~numgrids.boundary.resolve_bcs`: Dirichlet, Neumann or Robin
        conditions.  They hold at every instant: the boundary values are
        not evolved but computed from the interior values so that the
        conditions are satisfied (for Dirichlet conditions they are simply
        the prescribed values).  The initial state is adjusted accordingly.
        Values that change in time are wrapped in
        :class:`~numgrids.TimeDependent`, e.g.
        ``{"x_low": TimeDependent(lambda t: np.sin(t))}`` or
        ``Neumann(TimeDependent(flux))``.

        For coupled systems, pass a tuple with one specification per field
        (``None`` for a field without conditions).  A single specification
        applies to the first field, and the other fields are held fixed on
        the same boundary points (as suited for second-order equations
        written as :math:`u_t = v,\ v_t = \ldots`).
    method : str
        Integration method of :func:`scipy.integrate.solve_ivp`.  Use an
        implicit method (``"BDF"``, ``"Radau"``) for stiff problems such
        as diffusion.

        ``"expm"`` is an exponential integrator for linear problems (*rhs*
        an Operator or a list of blocks) with constant boundary values and
        source: the solution :math:`u(t) = e^{tL} u_0` (plus the
        contribution of source and boundary values) is evaluated directly
        at the output times, without time stepping error, for stiff
        problems and complex fields alike.  It preserves the norm in
        Schrödinger-type problems.
    source : array, callable or tuple, optional
        A source term added to the right-hand side,
        :math:`\partial_t u = F(t, u) + s`: an array of grid shape, or a
        callable ``source(t)`` returning one (for coupled systems a tuple
        with one entry per field, ``None`` for no source).  With an
        Operator right-hand side this keeps the problem linear, so that
        implicit methods still use the exact Jacobian.
    **kwargs
        Passed on to :func:`scipy.integrate.solve_ivp` (e.g. ``rtol``,
        ``atol``, ``max_step``).

    Returns
    -------
    EvolutionResult

    Examples
    --------
    Heat equation on a rod with fixed ends:

    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev, evolve
    >>> grid = Grid(x=chebyshev(24, 0, 1))
    >>> u0 = np.sin(np.pi * grid.coords)
    >>> res = evolve(grid, 0.1 * grid.laplacian_operator(), u0, (0, 1),
    ...              bcs=0, method="BDF", t_eval=[0, 1], rtol=1e-8, atol=1e-10)
    >>> exact = np.exp(-0.1 * np.pi ** 2) * u0
    >>> bool(np.allclose(res.final, exact, atol=1e-6))
    True

    An insulated rod (no heat flux through the ends) keeps its total heat:

    >>> from numgrids import Neumann
    >>> u0 = 1 + np.cos(np.pi * grid.coords)
    >>> res = evolve(grid, grid.laplacian_operator(), u0, (0, 1), bcs=Neumann(0),
    ...              method="BDF", rtol=1e-8, atol=1e-10)
    >>> round(grid.integrate(res.final), 6), round(grid.integrate(u0), 6)
    (1.0, 1.0)
    """
    from numgrids.boundary import BoundaryCondition, resolve_bcs
    from numgrids.operators import Operator

    system = isinstance(u0, (tuple, list))
    fields = [grid._check_field(np.array(u, copy=True), f"u0[{i}]")
              for i, u in enumerate(u0 if system else [u0])]
    nfields = len(fields)
    size = grid.size
    shape = grid.shape
    t0 = float(t_span[0])

    # Boundary conditions: an affine constraint per field, or "frozen"
    # boundary points (fields that are held fixed where field 0 has conditions).
    per_field = (system and isinstance(bcs, (tuple, list)) and len(bcs) == nfields
                 and not any(isinstance(b, BoundaryCondition) for b in bcs))
    specs = list(bcs) if per_field else [bcs] + [None] * (nfields - 1)
    constraints = []
    for spec in specs:
        bc_list = resolve_bcs(grid, spec)
        constraints.append(_BoundaryConstraint(grid, bc_list, t0) if bc_list else None)
    frozen = np.zeros((nfields, size), dtype=bool)
    if not per_field and constraints[0] is not None:
        frozen[1:] = constraints[0].rows
    for f, con in enumerate(constraints):
        if con is not None:
            fields[f] = con.project(fields[f].ravel(), t0).reshape(shape)
    zero_rows = frozen.copy()
    for f, con in enumerate(constraints):
        if con is not None:
            zero_rows[f] |= con.rows
    zero_rows = zero_rows.ravel()
    constrained = any(con is not None for con in constraints)
    moving_bcs = any(con is not None and con.time_dependent for con in constraints)

    source_fn, constant_source = _source_function(grid, source, nfields, system, zero_rows)

    dtype = np.result_type(*fields, float)
    y0 = np.concatenate([f.astype(dtype).ravel() for f in fields])

    linear = isinstance(rhs, (Operator, list, tuple))
    jac = None
    if linear:
        if isinstance(rhs, Operator):
            if system:
                raise TypeError(
                    "An Operator right-hand side only works for a single field; "
                    "for systems pass a nested list of blocks, e.g. [[None, 1], [L, None]]."
                )
            M = scipy.sparse.csr_matrix(rhs.matrix)
        else:
            from numgrids.nonlinear import _assemble_jacobian
            if not system:
                raise TypeError("A list of operator blocks needs a tuple of initial fields.")
            M = scipy.sparse.csr_matrix(_assemble_jacobian(grid, rhs, nfields))
        if zero_rows.any():
            M = scipy.sparse.csr_matrix(scipy.sparse.diags((~zero_rows).astype(float)) @ M)
        # Fold the boundary constraints u -> P u + d(t) into the matrix.
        A = M
        if constrained:
            P = scipy.sparse.block_diag(
                [scipy.sparse.identity(size) if con is None else con.matrix() for con in constraints],
                format="csr")
            A = scipy.sparse.csr_matrix(M @ P)

        def inhomogeneity(t):
            """The part of the right-hand side that does not depend on the solution."""
            q = None
            if constrained:
                d = np.concatenate([np.zeros(size) if con is None else con.offset(t) for con in constraints])
                q = M @ d
            if source_fn is not None:
                q = source_fn(t) if q is None else q + source_fn(t)
            return q

        constant = not moving_bcs and constant_source
        q0 = inhomogeneity(t0) if constant else None
        if q0 is not None and not np.any(q0):
            q0 = None
        y0 = y0.astype(np.result_type(y0, A.dtype, *([] if q0 is None else [q0])))

        if method == "expm":
            if not constant:
                raise ValueError(
                    'method="expm" needs boundary values and sources that are constant in time.'
                )
            if kwargs:
                raise TypeError(f'method="expm" takes no solver options, got {sorted(kwargs)}.')
            t, Y = _expm_solve(A, q0, y0, t_span, t_eval)
            return _result(grid, t, Y, constraints, nfields, system, True,
                           "The solution was computed with matrix exponentials.", None)

        def fun(t, y):
            if constant:
                return A @ y if q0 is None else A @ y + q0
            q = inhomogeneity(t)
            return A @ y if q is None else A @ y + q

        if method in _IMPLICIT:
            # Spectral matrices fill in under the sparse LU of the implicit solvers;
            # a dense Jacobian makes them use dense LU, which is much faster there.
            n = A.shape[0]
            jac = A.toarray() if n <= _JAC_DENSE_LIMIT and A.nnz >= _JAC_DENSE_DENSITY * n * n else A
    else:
        if method == "expm":
            raise ValueError(
                'method="expm" needs a linear right-hand side: an Operator, or a nested list '
                "of operator blocks for systems."
            )
        func: Callable = rhs

        def fun(t, y):
            parts = []
            for f in range(nfields):
                part = y[f * size:(f + 1) * size]
                if constraints[f] is not None:
                    part = constraints[f].project(part, t)
                parts.append(part.reshape(shape))
            out = func(t, *parts)
            if system:
                if len(out) != nfields:
                    raise ValueError(f"rhs returned {len(out)} arrays, expected {nfields}.")
                dy = np.concatenate([np.asarray(o).ravel() for o in out])
            else:
                dy = np.asarray(out).ravel()
            if dy.shape != y.shape:
                raise ValueError(
                    f"rhs returned an array of wrong size {dy.size}, expected {y.size}."
                )
            dy = dy.astype(np.result_type(dy, y), copy=True)
            dy[zero_rows] = 0
            if source_fn is not None:
                dy = dy + source_fn(t)
            return dy

    if jac is not None:
        kwargs.setdefault("jac", jac)
    sol = solve_ivp(fun, t_span, y0, method=method, t_eval=t_eval, **kwargs)
    return _result(grid, sol.t, sol.y.T, constraints, nfields, system,
                   bool(sol.success), str(sol.message), sol)


def _result(grid, t, Y, constraints, nfields, system, success, message, sol) -> EvolutionResult:
    """Package the flat states *Y* of shape ``(nt, nfields * size)`` as an :class:`EvolutionResult`."""
    size, shape = grid.size, grid.shape
    nt = len(t)
    us = []
    for f in range(nfields):
        Yf = Y[:, f * size:(f + 1) * size]
        if constraints[f] is not None:
            Yf = np.array([constraints[f].project(y, tk) for y, tk in zip(Yf, t)]).reshape(Yf.shape)
        us.append(Yf.reshape((nt,) + shape))
    us = tuple(us)
    return EvolutionResult(t=np.asarray(t), u=us if system else us[0], success=success,
                           message=message, sol=sol, grid=grid)


def _source_function(grid, source, nfields: int, system: bool, zero_rows: NDArray):
    """``(s, constant)``: the flat source term as a function of time (or ``None``).

    The source is set to zero on the points whose values are not evolved
    (boundary points with conditions).
    """
    if source is None:
        return None, True
    shape = grid.shape

    def flatten(value):
        if system and (not isinstance(value, (tuple, list)) or len(value) != nfields):
            raise ValueError(f"The source of a system must be a tuple of {nfields} entries.")
        parts = value if system else [value]
        flat = np.concatenate([
            np.zeros(grid.size) if part is None else np.broadcast_to(np.asarray(part), shape).ravel()
            for part in parts
        ])
        return np.where(zero_rows, 0, flat)

    if callable(source):
        return (lambda t: flatten(source(t))), False
    constant = flatten(source)
    return (lambda t: constant), True


def _expm_solve(A, q, y0: NDArray, t_span, t_eval) -> tuple[NDArray, NDArray]:
    r"""Solve :math:`y' = A y + q` exactly at the output times.

    With the constant inhomogeneity as an extra unknown :math:`z = (y, 1)`,
    the problem is homogeneous, :math:`z' = \hat A z`, so
    :math:`z(t + \Delta t) = e^{\Delta t \hat A} z(t)`.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    t = np.array([t0, t1]) if t_eval is None else np.asarray(t_eval, dtype=float)
    if t.ndim != 1 or (t.size and (t.min() < min(t0, t1) or t.max() > max(t0, t1))):
        raise ValueError("Values in t_eval are not within t_span.")
    if np.any(np.diff(t) * np.sign(t1 - t0) <= 0):
        raise ValueError("Values in t_eval are not properly sorted.")
    n = y0.size
    if q is not None:
        A = scipy.sparse.bmat([[A, scipy.sparse.csr_matrix(q.reshape(-1, 1))],
                               [None, scipy.sparse.csr_matrix((1, 1))]], format="csr")
        z = np.concatenate([y0, [1.0]])
    else:
        z = y0
    dense = A.shape[0] <= _EXPM_DENSE_LIMIT
    if dense:
        A = A.toarray()
    propagators: dict[float, NDArray] = {}
    Y = np.empty((len(t), n), dtype=np.result_type(z, A.dtype))
    now = t0
    for k, tk in enumerate(t):
        dt = tk - now
        if dt != 0:
            if dense:
                key = round(dt, 14)
                if key not in propagators:
                    propagators[key] = scipy.linalg.expm(dt * A)
                z = propagators[key] @ z
            else:
                z = scipy.sparse.linalg.expm_multiply(dt * A, z)
            now = tk
        Y[k] = z[:n]
    return t, Y

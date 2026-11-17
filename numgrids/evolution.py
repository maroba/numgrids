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
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from numgrids.grids import Grid

_IMPLICIT = ("Radau", "BDF", "LSODA")


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
        The raw result of :func:`scipy.integrate.solve_ivp`.
    """

    t: NDArray
    u: Any
    success: bool
    message: str
    sol: Any = field(repr=False, default=None)

    @property
    def final(self):
        """The solution at the last time point (array, or tuple for systems)."""
        if isinstance(self.u, tuple):
            return tuple(ui[-1] for ui in self.u)
        return self.u[-1]


class _BoundaryConstraint:
    r"""Boundary conditions of one field, as an affine constraint.

    The conditions on the boundary points *b* read
    :math:`R_{bb} u_b + R_{bi} u_i = g` (identity rows for Dirichlet,
    normal-derivative rows for Neumann and Robin conditions).  Solving for
    :math:`u_b` expresses the boundary values through the interior values:
    :math:`u_b = S u_i + c`, so the time integration only evolves the
    interior and the conditions hold at every instant.
    """

    def __init__(self, grid: Grid, bc_list) -> None:
        from numgrids.boundary import apply_bcs

        size = grid.size
        rows = np.zeros(size, dtype=bool)
        for bc in bc_list:
            rows[bc.face.flat_indices] = True
        R, g = apply_bcs(bc_list, scipy.sparse.csr_matrix((size, size)), np.zeros(size))
        R = scipy.sparse.csr_matrix(R)
        b = np.flatnonzero(rows)
        i = np.flatnonzero(~rows)
        R_bb = scipy.sparse.csc_matrix(R[b][:, b])
        R_bi = scipy.sparse.csc_matrix(R[b][:, i])
        try:
            lu = scipy.sparse.linalg.splu(R_bb)
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

    def project(self, y: NDArray) -> NDArray:
        """Overwrite the boundary values of *y* (flat) so that the conditions hold."""
        y = np.array(y, dtype=np.result_type(y, self.c), copy=True)
        y[self.b] = self.S @ y[self.i] + self.c
        return y

    def affine_map(self) -> tuple[scipy.sparse.csr_matrix, NDArray]:
        """``(P, d)`` with ``project(y) == P @ y + d`` for flat arrays *y*."""
        n, nb, ni = self.size, len(self.b), len(self.i)
        E_b = scipy.sparse.csr_matrix((np.ones(nb), (self.b, np.arange(nb))), shape=(n, nb))
        E_i = scipy.sparse.csr_matrix((np.ones(ni), (self.i, np.arange(ni))), shape=(n, ni))
        P = E_i @ E_i.T + E_b @ self.S @ E_i.T
        d = np.zeros(n, dtype=self.c.dtype)
        d[self.b] = self.c
        return scipy.sparse.csr_matrix(P), d


def evolve(
    grid: Grid,
    rhs,
    u0,
    t_span: tuple[float, float],
    t_eval: Sequence[float] | NDArray | None = None,
    bcs=None,
    method: str = "RK45",
    **kwargs,
) -> EvolutionResult:
    r"""Integrate :math:`\partial_t u = F(t, u)` in time on a grid.

    Parameters
    ----------
    grid : Grid
        The grid.
    rhs : callable or Operator
        The right-hand side.  Either a callable ``rhs(t, u)`` returning
        an array of shape ``grid.shape``, or -- for coupled systems with
        initial state ``(u0, v0, ...)`` -- ``rhs(t, u, v, ...)`` returning
        a tuple of arrays.  For linear problems an
        :class:`~numgrids.operators.Operator` ``L`` can be passed directly,
        meaning :math:`\partial_t u = L u` (its sparse matrix is used as
        Jacobian for implicit methods).
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
        conditions with time-independent values.  They hold at every
        instant: the boundary values are not evolved but computed from the
        interior values so that the conditions are satisfied (for Dirichlet
        conditions they are simply the prescribed values).  The initial
        state is adjusted accordingly.

        For coupled systems, pass a tuple with one specification per field
        (``None`` for a field without conditions).  A single specification
        applies to the first field, and the other fields are held fixed on
        the same boundary points (as suited for second-order equations
        written as :math:`u_t = v,\ v_t = \ldots`).
    method : str
        Integration method of :func:`scipy.integrate.solve_ivp`.  Use an
        implicit method (``"BDF"``, ``"Radau"``) for stiff problems such
        as diffusion.
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

    # Boundary conditions: an affine constraint per field, or "frozen"
    # boundary points (fields that are held fixed where field 0 has conditions).
    per_field = (system and isinstance(bcs, (tuple, list)) and len(bcs) == nfields
                 and not any(isinstance(b, BoundaryCondition) for b in bcs))
    specs = list(bcs) if per_field else [bcs] + [None] * (nfields - 1)
    constraints = []
    for spec in specs:
        bc_list = resolve_bcs(grid, spec)
        constraints.append(_BoundaryConstraint(grid, bc_list) if bc_list else None)
    frozen = np.zeros((nfields, size), dtype=bool)
    if not per_field and constraints[0] is not None:
        frozen[1:] = constraints[0].rows
    for f, con in enumerate(constraints):
        if con is not None:
            fields[f] = con.project(fields[f].ravel()).reshape(shape)
    zero_rows = frozen.copy()
    for f, con in enumerate(constraints):
        if con is not None:
            zero_rows[f] |= con.rows
    zero_rows = zero_rows.ravel()

    dtype = np.result_type(*fields, float)
    y0 = np.concatenate([f.astype(dtype).ravel() for f in fields])

    jac = None
    if isinstance(rhs, Operator):
        if system:
            raise TypeError("An Operator right-hand side only works for a single field.")
        M = scipy.sparse.csr_matrix(rhs.matrix)
        if zero_rows.any():
            M = scipy.sparse.csr_matrix(scipy.sparse.diags((~zero_rows).astype(float)) @ M)
        offset = None
        if constraints[0] is not None:
            P, d = constraints[0].affine_map()
            offset = M @ d
            M = scipy.sparse.csr_matrix(M @ P)
            if not np.any(offset):
                offset = None

        def fun(t, y):
            return M @ y if offset is None else M @ y + offset

        if method in _IMPLICIT:
            jac = M
    else:
        func: Callable = rhs

        def fun(t, y):
            parts = []
            for f in range(nfields):
                part = y[f * size:(f + 1) * size]
                if constraints[f] is not None:
                    part = constraints[f].project(part)
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
            return dy

    if jac is not None:
        kwargs.setdefault("jac", jac)
    sol = solve_ivp(fun, t_span, y0, method=method, t_eval=t_eval, **kwargs)

    nt = sol.t.shape[0]
    Y = sol.y.T  # (nt, nfields * size)
    us = []
    for f in range(nfields):
        Yf = Y[:, f * size:(f + 1) * size]
        if constraints[f] is not None:
            Yf = np.array([constraints[f].project(y) for y in Yf]).reshape(Yf.shape)
        us.append(Yf.reshape((nt,) + shape))
    us = tuple(us)
    u = us if system else us[0]
    return EvolutionResult(t=sol.t, u=u, success=bool(sol.success), message=str(sol.message), sol=sol)

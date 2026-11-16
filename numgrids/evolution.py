"""Time integration of grid-based PDEs (method of lines).

:func:`evolve` wraps :func:`scipy.integrate.solve_ivp` so that the
right-hand side can be written in terms of meshed arrays, optionally with
several coupled fields and Dirichlet boundary conditions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse
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
        Dirichlet boundary conditions, in any form accepted by
        :func:`~numgrids.boundary.resolve_bcs` (only Dirichlet conditions
        are allowed).  The boundary values are imposed on the (first field
        of the) initial state and held fixed: the time derivatives of all
        fields vanish on the boundary.
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
    """
    from numgrids.boundary import DirichletBC, resolve_bcs
    from numgrids.operators import Operator

    system = isinstance(u0, (tuple, list))
    fields = [grid._check_field(np.array(u, copy=True), f"u0[{i}]")
              for i, u in enumerate(u0 if system else [u0])]
    nfields = len(fields)

    bc_list = resolve_bcs(grid, bcs)
    for bc in bc_list:
        if not isinstance(bc, DirichletBC):
            raise TypeError(
                f"evolve only supports Dirichlet conditions, got {type(bc).__name__}. "
                "Implement other conditions inside the right-hand side."
            )
    if fields and bc_list:
        dtype = np.result_type(fields[0], *[bc._resolve_value() for bc in bc_list])
        fields[0] = fields[0].astype(dtype)
        for bc in bc_list:
            bc.apply(fields[0])
    frozen = np.zeros(grid.shape, dtype=bool)
    for bc in bc_list:
        frozen |= bc.face.mask
    frozen_flat = np.tile(frozen.ravel(), nfields)

    dtype = np.result_type(*fields, float)
    y0 = np.concatenate([f.astype(dtype).ravel() for f in fields])
    size = grid.size
    shape = grid.shape

    jac = None
    if isinstance(rhs, Operator):
        if system:
            raise TypeError("An Operator right-hand side only works for a single field.")
        op = rhs
        M = scipy.sparse.csr_matrix(op.matrix)
        if frozen.any():
            M = scipy.sparse.diags((~frozen.ravel()).astype(float)) @ M
            M = scipy.sparse.csr_matrix(M)

        def fun(t, y):
            return M @ y

        if method in _IMPLICIT:
            jac = M
    else:
        func: Callable = rhs

        def fun(t, y):
            parts = [y[i * size:(i + 1) * size].reshape(shape) for i in range(nfields)]
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
            dy[frozen_flat] = 0
            return dy

    if jac is not None:
        kwargs.setdefault("jac", jac)
    sol = solve_ivp(fun, t_span, y0, method=method, t_eval=t_eval, **kwargs)

    nt = sol.t.shape[0]
    Y = sol.y.T  # (nt, nfields * size)
    us = tuple(Y[:, i * size:(i + 1) * size].reshape((nt,) + shape) for i in range(nfields))
    u = us if system else us[0]
    return EvolutionResult(t=sol.t, u=u, success=bool(sol.success), message=str(sol.message), sol=sol)

"""Convergence studies: how fast does a numerical result converge with resolution?"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from numgrids.grids import Grid
from numgrids.utils import vector_norm


@dataclass
class ConvergenceResult:
    """Result of :func:`convergence_study`.

    Attributes
    ----------
    ns : NDArray
        Resolutions (as passed to the grid factory).
    errors : NDArray
        Error for each resolution.
    order : float
        Fitted algebraic convergence order *p* in ``error ~ n**(-p)``
        (least-squares fit in log-log space).  For spectrally converging
        methods this keeps growing with *n*; see :attr:`rate`.
    rate : float
        Fitted exponential rate *c* in ``error ~ exp(-c n)``.
    orders : NDArray
        Pairwise orders between consecutive resolutions.
    """

    ns: NDArray
    errors: NDArray
    order: float
    rate: float
    orders: NDArray = field(repr=False)

    def plot(self, ax=None, **kwargs):
        """Log-log plot of error versus resolution (requires matplotlib)."""
        from numgrids.plots import require_matplotlib
        plt = require_matplotlib()
        if ax is None:
            _, ax = plt.subplots()
        kwargs.setdefault("marker", "o")
        ax.loglog(self.ns, self.errors, **kwargs)
        ax.set_xlabel("$n$")
        ax.set_ylabel("error")
        return ax


def convergence_study(
    make_grid: Callable[[int], Grid],
    func: Callable[[Grid], NDArray | float],
    ns: Sequence[int],
    exact: Callable[[Grid], NDArray | float] | float | None = None,
    norm: str = "max",
    interpolation: str = "spectral",
) -> ConvergenceResult:
    """Measure the error of a computation for a sequence of resolutions.

    Parameters
    ----------
    make_grid : callable
        ``make_grid(n) -> Grid``, creates a grid for resolution *n*.
    func : callable
        ``func(grid)`` returns the quantity of interest: a scalar (e.g. an
        integral or eigenvalue) or an array of shape ``grid.shape``.
    ns : sequence of int
        The resolutions to test.
    exact : callable or float, optional
        The exact result: a number, or ``exact(grid)`` returning a scalar
        or an array.  If omitted, the result on the finest grid serves as
        reference (it is then not included in the output); array results
        are interpolated from the finest grid.
    norm : str
        ``"max"``, ``"l2"`` or ``"mean"`` (for array-valued results).
    interpolation : str
        Interpolation method used when comparing array results without
        *exact* (default ``"spectral"``).

    Returns
    -------
    ConvergenceResult

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, equidistant, convergence_study
    >>> res = convergence_study(
    ...     lambda n: Grid(equidistant(n, 0, 1)),
    ...     lambda g: g.diff(np.sin(g.coords), acc=2),
    ...     ns=[20, 40, 80, 160],
    ...     exact=lambda g: np.cos(g.coords),
    ... )
    >>> round(res.order)
    2
    """
    from numgrids.interpol import Interpolator

    ns = [int(n) for n in ns]
    if len(ns) < 2:
        raise ValueError("Need at least two resolutions.")
    grids = [make_grid(n) for n in ns]
    results = [func(g) for g in grids]

    if exact is None:
        ref_grid, ref = grids[-1], results[-1]
        grids, results, ns = grids[:-1], results[:-1], ns[:-1]
        if np.ndim(ref) == 0:
            def exact_on(g):
                return ref
        else:
            inter = Interpolator(ref_grid, ref, method=interpolation)

            def exact_on(g):
                return inter(g)
    elif callable(exact):
        exact_on = exact
    else:
        def exact_on(g):
            return exact

    errors = np.array([
        vector_norm(np.asarray(r) - np.asarray(exact_on(g)), norm)
        for g, r in zip(grids, results)
    ])
    ns_arr = np.array(ns, dtype=float)

    positive = errors > 0
    if positive.sum() >= 2:
        order = -np.polyfit(np.log(ns_arr[positive]), np.log(errors[positive]), 1)[0]
        rate = -np.polyfit(ns_arr[positive], np.log(errors[positive]), 1)[0]
    else:
        order = rate = float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        orders = np.log(errors[:-1] / errors[1:]) / np.log(ns_arr[1:] / ns_arr[:-1])

    return ConvergenceResult(
        ns=np.array(ns), errors=errors, order=float(order), rate=float(rate), orders=orders
    )

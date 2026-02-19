"""Adaptive mesh refinement for numerical grids.

Provides error estimation by comparing solutions at different resolutions
and automatic per-axis resolution selection to meet a prescribed error
tolerance.

The core idea is **Richardson-extrapolation-style error estimation**: for
each axis, refine that axis alone, re-evaluate the function, interpolate
the result back onto the original grid, and measure the difference.  The
axis whose refinement changes the answer the most is the resolution
bottleneck and gets refined first.

Because numgrids uses tensor-product grids, refinement is performed
per-axis rather than per-cell.  This keeps all existing operators
(differentiation, integration, interpolation) working unchanged while
still allowing anisotropic resolution---e.g., fine radial resolution
with coarse angular resolution.

Classes
-------
ErrorEstimator
    Estimates discretization error by comparing a function evaluated on
    grids of different resolution.
AdaptationResult
    Container for the result of an adaptation step.

Functions
---------
adapt
    Iteratively refine a grid per-axis until an error tolerance is met.
estimate_error
    One-shot error estimation comparing a function on two resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from numgrids.grids import Grid
from numgrids.interpol import Interpolator


class ErrorEstimator:
    """Estimate discretization error by comparing grid resolutions.

    The estimator evaluates a user-supplied function on the current grid
    and on a refined version, then computes the difference.  This gives a
    Richardson-extrapolation-style error indicator without requiring an
    analytical solution.

    Parameters
    ----------
    grid : Grid
        The grid on which to estimate error.
    func : callable
        ``func(grid) -> NDArray`` --- a function that accepts a
        :class:`~numgrids.grids.Grid` and returns an array of shape
        ``grid.shape``.  For analytical functions, use the grid's
        meshed coordinates; for PDE solvers, use the grid to build
        operators.
    norm : str
        The norm used for the scalar error measure.  One of ``"max"``
        (L-infinity), ``"l2"`` (root-mean-square), or ``"mean"``
        (mean absolute error).

    Examples
    --------
    >>> from numgrids import Grid
    >>> from numgrids.axes import ChebyshevAxis
    >>> from numgrids.amr import ErrorEstimator
    >>> ax = ChebyshevAxis(10, 0, 1)
    >>> grid = Grid(ax)
    >>> est = ErrorEstimator(grid, lambda g: g.meshed_coords[0] ** 5)
    >>> errors = est.per_axis_errors()
    """

    def __init__(
        self,
        grid: Grid,
        func: Callable[[Grid], NDArray],
        norm: str = "max",
    ) -> None:
        self.grid = grid
        self.func = func
        if norm not in ("max", "l2", "mean"):
            raise ValueError(
                f"Unknown norm {norm!r}. Use 'max', 'l2', or 'mean'."
            )
        self.norm = norm
        self._f_current: NDArray | None = None

    def _compute_norm(self, diff: NDArray) -> float:
        """Compute a scalar error measure from a difference array."""
        if self.norm == "max":
            return float(np.max(np.abs(diff)))
        elif self.norm == "l2":
            return float(np.sqrt(np.mean(diff ** 2)))
        else:  # mean
            return float(np.mean(np.abs(diff)))

    @property
    def f_current(self) -> NDArray:
        """The function evaluated on the current grid (cached)."""
        if self._f_current is None:
            self._f_current = self.func(self.grid)
        return self._f_current

    def global_error(self, refinement_factor: float = 2.0) -> float:
        """Estimate the global discretization error.

        Refines *all* axes by *refinement_factor*, evaluates the function
        on the fine grid, interpolates back to the coarse grid, and
        returns the norm of the difference.

        Parameters
        ----------
        refinement_factor : float
            Factor by which to refine each axis (default 2).

        Returns
        -------
        float
            Scalar error estimate.
        """
        fine_grid = self.grid
        for i in range(self.grid.ndims):
            fine_grid = fine_grid.refine_axis(i, refinement_factor)

        f_fine = self.func(fine_grid)
        interp = Interpolator(fine_grid, f_fine, method="linear")
        f_fine_on_coarse = interp(self.grid)

        return self._compute_norm(self.f_current - f_fine_on_coarse)

    def per_axis_errors(
        self, refinement_factor: float = 2.0
    ) -> dict[int, float]:
        """Estimate the error contribution from each axis independently.

        For each axis *i*, creates a grid refined only along axis *i*,
        evaluates the function, interpolates back, and computes the
        difference.  Axes with the largest error are the ones that most
        need additional resolution.

        Parameters
        ----------
        refinement_factor : float
            Factor by which to refine each axis in turn (default 2).

        Returns
        -------
        dict[int, float]
            Mapping from axis index to scalar error estimate.
        """
        errors: dict[int, float] = {}
        for i in range(self.grid.ndims):
            refined_grid = self.grid.refine_axis(i, refinement_factor)
            f_refined = self.func(refined_grid)
            interp = Interpolator(refined_grid, f_refined, method="linear")
            f_refined_on_current = interp(self.grid)
            errors[i] = self._compute_norm(
                self.f_current - f_refined_on_current
            )
        return errors

    def axis_needing_refinement(
        self, refinement_factor: float = 2.0
    ) -> int | None:
        """Return the index of the axis with the largest error.

        Parameters
        ----------
        refinement_factor : float
            Factor by which to test-refine each axis (default 2).

        Returns
        -------
        int or None
            Axis index with the largest error contribution, or ``None``
            if all errors are zero (function is fully resolved).
        """
        errors = self.per_axis_errors(refinement_factor)
        if not errors or max(errors.values()) == 0.0:
            return None
        return max(errors, key=errors.get)


@dataclass
class AdaptationResult:
    """Container for the result of an adaptive refinement.

    Attributes
    ----------
    grid : Grid
        The adapted grid.
    f : NDArray
        The function evaluated on the adapted grid.
    errors : dict[int, float]
        Per-axis error estimates on the final grid.
    global_error : float
        Global error estimate on the final grid.
    iterations : int
        Number of adaptation iterations performed.
    converged : bool
        Whether the error tolerance was met.
    history : list[dict]
        Per-iteration diagnostics (shape, errors, which axes were
        refined).
    """

    grid: Grid
    f: NDArray
    errors: dict[int, float]
    global_error: float
    iterations: int
    converged: bool
    history: list[dict] = field(default_factory=list)


def adapt(
    grid: Grid,
    func: Callable[[Grid], NDArray],
    tol: float = 1e-6,
    norm: str = "max",
    max_iterations: int = 20,
    max_points_per_axis: int = 1024,
    refinement_factor: float = 2.0,
    refine_all: bool = False,
) -> AdaptationResult:
    """Iteratively refine a grid per-axis to meet an error tolerance.

    At each iteration the axis contributing the most discretization
    error is refined by *refinement_factor*.  The loop stops when the
    global error drops below *tol*, when *max_iterations* is reached,
    or when all axes have hit *max_points_per_axis*.

    Parameters
    ----------
    grid : Grid
        The initial grid.
    func : callable
        ``func(grid) -> NDArray`` --- evaluates the quantity of interest
        on any grid.
    tol : float
        Target error tolerance (default ``1e-6``).
    norm : str
        Error norm: ``"max"``, ``"l2"``, or ``"mean"`` (default
        ``"max"``).
    max_iterations : int
        Maximum number of refinement iterations (default 20).
    max_points_per_axis : int
        Safety cap on points per axis to prevent runaway refinement
        (default 1024).
    refinement_factor : float
        Factor by which to increase axis resolution at each step
        (default 2).
    refine_all : bool
        If ``True``, refine *all* axes whose error exceeds
        ``tol / grid.ndims`` at each step instead of only the worst
        axis (default ``False``).

    Returns
    -------
    AdaptationResult
        The adapted grid, function values, error estimates, and
        diagnostic history.

    Examples
    --------
    >>> from numgrids import Grid
    >>> from numgrids.axes import EquidistantAxis
    >>> from numgrids.amr import adapt
    >>> grid = Grid(EquidistantAxis(10, 0, 1), EquidistantAxis(10, 0, 1))
    >>> result = adapt(
    ...     grid,
    ...     lambda g: g.meshed_coords[0] ** 5 + g.meshed_coords[1] ** 2,
    ...     tol=1e-4,
    ... )
    >>> result.converged
    True
    """
    history: list[dict] = []
    iterations = 0

    for iteration in range(max_iterations):
        estimator = ErrorEstimator(grid, func, norm=norm)
        axis_errors = estimator.per_axis_errors(refinement_factor)
        g_error = estimator.global_error(refinement_factor)

        record: dict = {
            "iteration": iteration,
            "shape": grid.shape,
            "global_error": g_error,
            "axis_errors": dict(axis_errors),
        }
        history.append(record)
        iterations = iteration + 1

        if g_error < tol:
            return AdaptationResult(
                grid=grid,
                f=estimator.f_current,
                errors=axis_errors,
                global_error=g_error,
                iterations=iterations,
                converged=True,
                history=history,
            )

        # Determine which axes to refine
        if refine_all:
            per_axis_tol = tol / grid.ndims
            axes_to_refine = [
                i
                for i, err in axis_errors.items()
                if err > per_axis_tol
                and len(grid.axes[i]) < max_points_per_axis
            ]
        else:
            worst = max(axis_errors, key=axis_errors.get)
            if len(grid.axes[worst]) < max_points_per_axis:
                axes_to_refine = [worst]
            else:
                axes_to_refine = []

        if not axes_to_refine:
            break

        for axis_idx in axes_to_refine:
            grid = grid.refine_axis(axis_idx, refinement_factor)
            record[f"refined_axis_{axis_idx}"] = True

    # Did not converge — compute final error on the last grid
    estimator = ErrorEstimator(grid, func, norm=norm)
    axis_errors = estimator.per_axis_errors(refinement_factor)
    g_error = estimator.global_error(refinement_factor)

    return AdaptationResult(
        grid=grid,
        f=estimator.f_current,
        errors=axis_errors,
        global_error=g_error,
        iterations=iterations,
        converged=g_error < tol,
        history=history,
    )


def estimate_error(
    grid: Grid,
    func: Callable[[Grid], NDArray],
    norm: str = "max",
    refinement_factor: float = 2.0,
) -> dict[str, float | dict[int, float]]:
    """One-shot error estimation for a function on a grid.

    A convenience wrapper around :class:`ErrorEstimator` for cases
    where you only need a single error report without running the
    full adaptation loop.

    Parameters
    ----------
    grid : Grid
        The grid on which the function is evaluated.
    func : callable
        ``func(grid) -> NDArray``.
    norm : str
        Error norm (default ``"max"``).
    refinement_factor : float
        Factor by which to test-refine (default 2).

    Returns
    -------
    dict
        ``{"global": float, "per_axis": {0: float, 1: float, ...}}``.

    Examples
    --------
    >>> from numgrids import Grid
    >>> from numgrids.axes import EquidistantAxis
    >>> from numgrids.amr import estimate_error
    >>> grid = Grid(EquidistantAxis(20, 0, 1))
    >>> result = estimate_error(grid, lambda g: g.meshed_coords[0] ** 2)
    >>> result["global"]  # doctest: +SKIP
    """
    est = ErrorEstimator(grid, func, norm=norm)
    return {
        "global": est.global_error(refinement_factor),
        "per_axis": est.per_axis_errors(refinement_factor),
    }

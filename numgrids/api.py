from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from numgrids.grids import Grid
from numgrids.axes import Axis, EquidistantAxis, ChebyshevAxis, LogAxis


class Diff:

    def __init__(self, grid: Grid, order: int, axis_index: int = 0) -> None:
        """Constructor for partial derivative operator.

        Parameters
        ----------
        grid: Grid
            The numerical grid on which to apply the partial derivative.
        order: positive int
            The order of the derivative.
        axis_index: int
            The axis index (which axis in the grid).
        """
        if order <= 0:
            raise ValueError("Derivative order must be positive integer.")

        if not isinstance(grid, Grid):
            raise TypeError("Parameter 'grid' must be of type Grid.")

        if axis_index < 0:
            raise ValueError("axis must be nonnegative integer.")

        if axis_index > grid.ndims - 1:
            raise ValueError("No such axis index in this grid!")

        axis = grid.get_axis(axis_index)
        self.operator = axis.create_diff_operator(grid, order, axis_index)

    def __call__(self, f: NDArray) -> NDArray:
        """Apply the derivative to the array f."""
        return self.operator(f)

    def as_matrix(self) -> spmatrix:
        """
        Returns a matrix representation of the differential operator.

        The data type is a scipy sparse matrix.
        """
        return self.operator.as_matrix()


class AxisType(str, enum.Enum):
    """Enumeration of the available axis types in this package."""
    EQUIDISTANT = "equidistant"
    EQUIDISTANT_PERIODIC = "equidistant_periodic"
    CHEBYSHEV = "chebyshev"
    LOGARITHMIC = "log"


def create_axis(axis_type: AxisType, num_points: int, low: float, high: float,
                **kwargs) -> Axis:
    """Create an Axis object of a given type.

    Parameters
    ----------
    axis_type: AxisType
        The type of axis (equidistant, periodic, logarithmic, Chebyshev, etc.)
    num_points: positive int
        Number of grid points along this axis.
    low: float
        The lowest coordinate value on the axis.
    high: float
        The highest coordinate value on the axis.

    Returns
    -------
    Axis object of specified type.
    """
    if axis_type == AxisType.EQUIDISTANT:
        return EquidistantAxis(num_points, low, high, **kwargs)
    elif axis_type == AxisType.EQUIDISTANT_PERIODIC:
        return EquidistantAxis(num_points, low, high, periodic=True, **kwargs)
    elif axis_type == AxisType.CHEBYSHEV:
        return ChebyshevAxis(num_points, low, high, **kwargs)
    elif axis_type == AxisType.LOGARITHMIC:
        return LogAxis(num_points, low, high, **kwargs)
    else:
        raise NotImplementedError(f"No such axis type: {axis_type}")


class SphericalGrid(Grid):
    """A spherical grid in spherical coordinates (r, theta, phi)."""

    def __init__(self, raxis: Axis, theta_axis: Axis, phi_axis: Axis) -> None:
        """Constructor

        Parameters
        ----------
        raxis: Axis
            The radial axis.
        theta_axis: Axis
            The polar axis (theta).
        phi_axis: Axis
            The azimuthal axis (phi). Must be periodic.
        """
        super().__init__(raxis, theta_axis, phi_axis)
        self._laplacian_fn: callable | None = None

    def _setup_laplacian(self) -> None:
        """Lazily initialize the laplacian operator."""
        dr2 = Diff(self, 2, 0)
        dr = Diff(self, 1, 0)
        dtheta = Diff(self, 1, 1)
        dphi2 = Diff(self, 2, 2)

        R, Phi, Theta = self.meshed_coords

        def laplacian(f: NDArray) -> NDArray:
            return dr2(f) + 2 / R * dr(f) + \
                   1 / (R ** 2 * np.sin(Theta)) * dtheta(np.sin(Theta) * dtheta(f)) + \
                   1 / (R ** 2 * np.sin(Theta) ** 2) * dphi2(f)

        self._laplacian_fn = laplacian

    def laplacian(self, f: NDArray) -> NDArray:
        """Apply the laplacian in spherical coordinates to f."""
        if self._laplacian_fn is None:
            self._setup_laplacian()
        return self._laplacian_fn(f)


def diff(grid: Grid, f: NDArray, order: int = 1, axis_index: int = 0) -> NDArray:
    """Convenience function for differentiation with caching."""
    if (order, axis_index) in grid.cache.get("diffs"):
        d = grid.cache["diffs"][order, axis_index]
    else:
        d = Diff(grid, order, axis_index)
        grid.cache["diffs"][order, axis_index] = d
    return d(f)


def interpolate(grid: Grid, f: NDArray, locations) -> NDArray:
    """Convenience function for interpolation."""
    from numgrids.interpol import Interpolator
    inter = Interpolator(grid, f)
    return inter(locations)


def integrate(grid: Grid, f: NDArray) -> float:
    """Convenience function for integration with caching."""
    from numgrids.integration import Integral
    if grid.cache.get("integral"):
        I = grid.cache["integral"]
    else:
        I = Integral(grid)
        grid.cache["integral"] = I
    return I(f)

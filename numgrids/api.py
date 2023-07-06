import numpy as np

from numgrids.grids import Grid
from numgrids.axes import EquidistantAxis, ChebyshevAxis, LogAxis
from numgrids.diff import FiniteDifferenceDiff, FFTDiff, ChebyshevDiff, LogDiff


class Diff:

    def __init__(self, grid, order, axis_index=0):
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

        if isinstance(axis, EquidistantAxis):
            if axis.periodic:
                self.operator = FFTDiff(grid, order, axis_index)
            else:
                self.operator = FiniteDifferenceDiff(grid, order, axis_index)
        elif isinstance(axis, ChebyshevAxis):
            self.operator = ChebyshevDiff(grid, order, axis_index)
        elif isinstance(axis, LogAxis):
            self.operator = LogDiff(grid, order, axis_index)
        else:
            raise NotImplementedError

    def __call__(self, f):
        """Apply the derivative to the array f."""
        return self.operator(f)

    def as_matrix(self):
        """
        Returns a matrix representation of the differential operator.

        The data type is a scipy sparse matrix.
        """
        return self.operator.as_matrix()


class AxisType:
    """Enumeration of the available axis types in this package.

        Available constants:
            EQUIDISTANT
            EQUIDISTANT_PERIODIC
            CHEBYSHEV
            LOGARITHMIC
    """
    EQUIDISTANT = "equidistant"
    EQUIDISTANT_PERIODIC = "equidistant_periodic"
    CHEBYSHEV = "chebyshev"
    LOGARITHMIC = "log"


def Axis(axis_type, num_points, low, high, **kwargs):
    """Creates an Axis object of a given type.

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

    def __init__(self, raxis, theta_axis, phi_axis):
        """
        Constructor

        Parameters
        ----------
        raxis: Axis
            The radial axis.
        theta_axis: Axis
            The polar axis (theta).
        phi_axis: Axis
            The azimuthal axis (phi). Must be periodic.
        """
        super(SphericalGrid, self).__init__(raxis, theta_axis, phi_axis)

        dr2 = Diff(self, 2, 0)
        dr = Diff(self, 1, 0)
        dtheta = Diff(self, 1, 1)
        dphi2 = Diff(self, 2, 2)

        R, Phi, Theta = self.meshed_coords

        def laplacian(f):
            return dr2(f) + 2 / R * dr(f) + \
                   1 / (R ** 2 * np.sin(Theta)) * dtheta(np.sin(Theta) * dtheta(f)) + \
                   1 / (R ** 2 * np.sin(Theta) ** 2) * dphi2(f)

        self._laplacian = laplacian

    def laplacian(self, f):
        """Returns the laplacian in spherical coordinates as a callable."""
        return self._laplacian(f)


def diff(grid, f, order=1, axis_index=0):
    if (order, axis_index) in grid.cache.get("diffs"):
        d = grid.cache["diffs"][order, axis_index]
    else:
        d = Diff(grid, order, axis_index)
        grid.cache["diffs"][order, axis_index] = d
    return d(f)


def interpolate(grid, f, locations):
    from numgrids import Interpolator
    inter = Interpolator(grid, f)
    return inter(locations)


def integrate(grid, f):
    from numgrids.integration import Integral
    if grid.cache.get("integral"):
        I = grid.cache["integral"]
    else:
        I = Integral(grid)
        grid.cache["integral"] = I
    return I(f)

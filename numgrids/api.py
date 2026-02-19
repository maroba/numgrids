from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from numgrids.grids import Grid
from numgrids.curvilinear import CurvilinearGrid
from numgrids.axes import Axis, EquidistantAxis, ChebyshevAxis, LogAxis


class Diff:
    """Partial derivative operator on a numerical grid.

    Automatically selects the best differentiation strategy (finite
    differences, FFT spectral, Chebyshev spectral, or log-scale) based
    on the axis type at *axis_index*.

    Examples
    --------
    >>> from numgrids import *
    >>> ax = create_axis(AxisType.CHEBYSHEV, 30, 0, 1)
    >>> grid = Grid(ax)
    >>> d = Diff(grid, 1)      # first derivative
    >>> d(grid.coords ** 2)    # apply to x**2
    """

    def __init__(self, grid: Grid, order: int, axis_index: int = 0, acc: int = 4) -> None:
        """Constructor for partial derivative operator.

        Parameters
        ----------
        grid: Grid
            The numerical grid on which to apply the partial derivative.
        order: positive int
            The order of the derivative.
        axis_index: int
            The axis index (which axis in the grid).
        acc: int
            The accuracy order for finite-difference based methods. Higher values
            use wider stencils for better accuracy. Ignored by spectral methods
            (Chebyshev). Default is 4.
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
        self.operator = axis.create_diff_operator(grid, order, axis_index, acc=acc)

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
    """Enumeration of available axis types.

    Members
    -------
    EQUIDISTANT
        Uniformly spaced grid points (non-periodic).
    EQUIDISTANT_PERIODIC
        Uniformly spaced grid points with periodic boundary conditions.
    CHEBYSHEV
        Chebyshev-node spacing for spectral accuracy on non-periodic domains.
    LOGARITHMIC
        Logarithmically spaced grid points (requires ``low > 0``).
    """
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


class SphericalGrid(CurvilinearGrid):
    r"""A spherical grid in spherical coordinates (r, theta, phi).

    Provides built-in vector calculus operators:

    - :meth:`laplacian` — scalar Laplacian
    - :meth:`gradient` — gradient of a scalar field
    - :meth:`divergence` — divergence of a vector field
    - :meth:`curl` — curl of a vector field

    Coordinate singularities at *r = 0* and *theta = 0, pi* are handled
    gracefully: non-finite values are replaced by zero.
    """

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
        super().__init__(
            raxis, theta_axis, phi_axis,
            scale_factors=(
                lambda c: np.ones_like(c[0]),       # h_r = 1
                lambda c: c[0],                      # h_theta = r
                lambda c: c[0] * np.sin(c[1]),       # h_phi = r sin(theta)
            ),
        )


class CylindricalGrid(CurvilinearGrid):
    r"""A grid in cylindrical coordinates (r, phi, z).

    Provides built-in vector calculus operators:

    - :meth:`laplacian` — scalar Laplacian
    - :meth:`gradient` — gradient of a scalar field
    - :meth:`divergence` — divergence of a vector field
    - :meth:`curl` — curl of a vector field

    Coordinate singularities at *r = 0* are handled gracefully.

    Examples
    --------
    >>> from numgrids import *
    >>> import numpy as np
    >>> grid = CylindricalGrid(
    ...     create_axis(AxisType.CHEBYSHEV, 20, 0.1, 2),
    ...     create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2 * np.pi),
    ...     create_axis(AxisType.CHEBYSHEV, 20, -1, 1),
    ... )
    >>> R, Phi, Z = grid.meshed_coords
    >>> f = R ** 2 + Z ** 2
    >>> lap_f = grid.laplacian(f)  # should be ~6
    """

    def __init__(self, raxis: Axis, phi_axis: Axis, zaxis: Axis) -> None:
        """Constructor

        Parameters
        ----------
        raxis: Axis
            The radial axis.
        phi_axis: Axis
            The azimuthal axis (phi). Should typically be periodic.
        zaxis: Axis
            The axial (z) axis.
        """
        super().__init__(
            raxis, phi_axis, zaxis,
            scale_factors=(
                lambda c: np.ones_like(c[0]),   # h_r = 1
                lambda c: c[0],                  # h_phi = r
                lambda c: np.ones_like(c[0]),   # h_z = 1
            ),
        )


class PolarGrid(CurvilinearGrid):
    r"""A 2D grid in polar coordinates (r, phi).

    Provides built-in vector calculus operators:

    - :meth:`laplacian` — scalar Laplacian
    - :meth:`gradient` — gradient of a scalar field
    - :meth:`divergence` — divergence of a 2D vector field
    - :meth:`curl` — curl (returns the scalar *z*-component)

    Coordinate singularities at *r = 0* are handled gracefully.

    Examples
    --------
    >>> from numgrids import *
    >>> import numpy as np
    >>> grid = PolarGrid(
    ...     create_axis(AxisType.CHEBYSHEV, 30, 0.1, 1),
    ...     create_axis(AxisType.EQUIDISTANT_PERIODIC, 40, 0, 2 * np.pi),
    ... )
    >>> R, Phi = grid.meshed_coords
    >>> f = R ** 2 * np.cos(Phi)
    >>> lap_f = grid.laplacian(f)
    """

    def __init__(self, raxis: Axis, phi_axis: Axis) -> None:
        """Constructor

        Parameters
        ----------
        raxis: Axis
            The radial axis.
        phi_axis: Axis
            The azimuthal axis (phi). Should typically be periodic.
        """
        super().__init__(
            raxis, phi_axis,
            scale_factors=(
                lambda c: np.ones_like(c[0]),   # h_r = 1
                lambda c: c[0],                  # h_phi = r
            ),
        )


def diff(grid: Grid, f: NDArray, order: int = 1, axis_index: int = 0, acc: int = 4) -> NDArray:
    """Differentiate a meshed function (with operator caching).

    Creates a :class:`Diff` operator on first call and caches it on the
    grid for subsequent calls with the same ``(order, axis_index, acc)``.

    Parameters
    ----------
    grid : Grid
        The grid on which *f* is defined.
    f : NDArray
        Meshed function values with shape ``grid.shape``.
    order : int, optional
        Derivative order (default ``1``).
    axis_index : int, optional
        Axis along which to differentiate (default ``0``).
    acc : int, optional
        Accuracy order for finite-difference methods (default ``4``).

    Returns
    -------
    NDArray
        The derivative array, same shape as *f*.
    """
    cache_key = (order, axis_index, acc)
    if cache_key in grid.cache.get("diffs"):
        d = grid.cache["diffs"][cache_key]
    else:
        d = Diff(grid, order, axis_index, acc=acc)
        grid.cache["diffs"][cache_key] = d
    return d(f)


def interpolate(grid: Grid, f: NDArray, locations) -> NDArray:
    """Interpolate a meshed function at arbitrary locations.

    Parameters
    ----------
    grid : Grid
        The grid on which *f* is defined.
    f : NDArray
        Meshed function values.
    locations : tuple, list[tuple], zip, or Grid
        Point(s) at which to interpolate.

    Returns
    -------
    NDArray
        Interpolated value(s).
    """
    from numgrids.interpol import Interpolator
    inter = Interpolator(grid, f)
    return inter(locations)


def integrate(grid: Grid, f: NDArray) -> float:
    """Integrate a meshed function over the entire grid domain.

    The :class:`~numgrids.integration.Integral` operator is cached on the
    grid after the first call.

    Parameters
    ----------
    grid : Grid
        The grid on which *f* is defined.
    f : NDArray
        Meshed function values.

    Returns
    -------
    float
        The definite integral over the grid domain.
    """
    from numgrids.integration import Integral
    if grid.cache.get("integral"):
        I = grid.cache["integral"]
    else:
        I = Integral(grid)
        grid.cache["integral"] = I
    return I(f)

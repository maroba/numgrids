"""High-level API: axis factories, the ``Diff`` operator and convenience functions."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from numgrids.axes import Axis, ChebyshevAxis, ColatitudeAxis, EquidistantAxis, LogAxis, RadialAxis
from numgrids.curvilinear import (  # noqa: F401  (re-exported for backward compatibility)
    CurvilinearGrid,
    CylindricalGrid,
    EllipticCylindricalGrid,
    OblateSpheroidalGrid,
    PolarGrid,
    ProlateSpheroidalGrid,
    SphericalGrid,
    SphericalSurfaceGrid,
)
from numgrids.grids import Grid

if TYPE_CHECKING:
    from numgrids.operators import Operator


class Diff:
    """Partial derivative operator on a numerical grid.

    Automatically selects the best differentiation strategy (finite
    differences, FFT spectral, Chebyshev spectral, or log-scale) based
    on the axis type at *axis_index*.

    ``Diff`` objects can be combined algebraically with each other, with
    scalars and with arrays (interpreted as multiplication operators); the
    result is an :class:`~numgrids.operators.Operator`.

    Examples
    --------
    >>> from numgrids import *
    >>> ax = create_axis(AxisType.CHEBYSHEV, 30, 0, 1)
    >>> grid = Grid(ax)
    >>> d = Diff(grid, 1)      # first derivative
    >>> df = d(grid.coords ** 2)    # apply to x**2
    >>> L = Diff(grid, 2) - 4 * Diff(grid, 1) + 3   # an Operator
    """

    __array_ufunc__ = None  # let numpy defer to our reflected operators

    def __init__(self, grid: Grid, order: int, axis_index: int | str = 0, acc: int = 4) -> None:
        """Constructor for partial derivative operator.

        Parameters
        ----------
        grid: Grid
            The numerical grid on which to apply the partial derivative.
        order: positive int
            The order of the derivative.
        axis_index: int or str
            The axis index (which axis in the grid), or the axis name.
        acc: int
            The accuracy order for finite-difference based methods. Higher values
            use wider stencils for better accuracy. Ignored by spectral methods
            (Chebyshev). Default is 4.
        """
        if isinstance(order, bool) or int(order) != order:
            raise TypeError(f"Derivative order must be an integer, got {order!r}.")
        if order <= 0:
            raise ValueError("Derivative order must be positive integer.")

        if not isinstance(grid, Grid):
            raise TypeError("Parameter 'grid' must be of type Grid.")

        if isinstance(axis_index, str):
            axis_index = grid.axis_index(axis_index)

        if axis_index < 0:
            raise ValueError("axis must be nonnegative integer.")

        if axis_index > grid.ndims - 1:
            raise ValueError("No such axis index in this grid!")

        self.grid = grid
        self.order = int(order)
        self.axis_index = axis_index
        self.acc = acc
        axis = grid.get_axis(axis_index)
        self.operator = axis.create_diff_operator(grid, int(order), axis_index, acc=acc)

    def __call__(self, f: NDArray) -> NDArray:
        """Apply the derivative to the array f."""
        return self.operator(f)

    def as_matrix(self) -> spmatrix:
        """
        Returns a matrix representation of the differential operator.

        The data type is a scipy sparse matrix.
        """
        return self.operator.as_matrix()

    def to_operator(self) -> Operator:
        """Convert to a general :class:`~numgrids.operators.Operator`."""
        from numgrids.operators import Operator
        return Operator(self.grid, self.as_matrix())

    def __repr__(self) -> str:
        grid = f"{type(self.grid).__name__}{self.grid.shape}"
        return f"Diff(order={self.order}, axis_index={self.axis_index}, grid={grid})"

    # Operator algebra: delegate to Operator
    def __add__(self, other):
        return self.to_operator() + other

    def __radd__(self, other):
        return other + self.to_operator()

    def __sub__(self, other):
        return self.to_operator() - other

    def __rsub__(self, other):
        return other - self.to_operator()

    def __mul__(self, other):
        return self.to_operator() * other

    def __rmul__(self, other):
        return other * self.to_operator()

    def __matmul__(self, other):
        return self.to_operator() @ other

    def __rmatmul__(self, other):
        return other @ self.to_operator()

    def __truediv__(self, other):
        return self.to_operator() / other

    def __neg__(self):
        return -self.to_operator()

    def __pow__(self, n: int):
        return self.to_operator() ** n


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
    RADIAL
        Radial coordinate through the origin of polar, cylindrical and
        spherical grids (requires ``low = 0``).
    COLATITUDE
        Polar angle of spherical grids through both poles (requires
        ``low = 0, high = pi``).
    """
    EQUIDISTANT = "equidistant"
    EQUIDISTANT_PERIODIC = "equidistant_periodic"
    CHEBYSHEV = "chebyshev"
    LOGARITHMIC = "log"
    RADIAL = "radial"
    COLATITUDE = "colatitude"


_AXIS_TYPE_ALIASES = {
    "periodic": AxisType.EQUIDISTANT_PERIODIC,
    "cheb": AxisType.CHEBYSHEV,
    "logarithmic": AxisType.LOGARITHMIC,
    "equi": AxisType.EQUIDISTANT,
}


def create_axis(axis_type: AxisType | str, num_points: int, low: float, high: float,
                **kwargs) -> Axis:
    """Create an Axis object of a given type.

    Parameters
    ----------
    axis_type: AxisType or str
        The type of axis (equidistant, periodic, logarithmic, Chebyshev, etc.).
        Strings such as ``"chebyshev"``, ``"equidistant"``,
        ``"equidistant_periodic"`` (or ``"periodic"``) and ``"log"`` are
        accepted, too.
    num_points: positive int
        Number of grid points along this axis.
    low: float
        The lowest coordinate value on the axis.
    high: float
        The highest coordinate value on the axis.
    **kwargs
        Further keyword arguments for the axis, e.g. ``name``.

    Returns
    -------
    Axis object of specified type.
    """
    if isinstance(axis_type, str) and not isinstance(axis_type, AxisType):
        key = axis_type.lower()
        if key in _AXIS_TYPE_ALIASES:
            axis_type = _AXIS_TYPE_ALIASES[key]
        else:
            try:
                axis_type = AxisType(key)
            except ValueError:
                valid = [t.value for t in AxisType] + list(_AXIS_TYPE_ALIASES)
                raise NotImplementedError(
                    f"No such axis type: {axis_type!r}. Valid types: {valid}"
                ) from None
    if axis_type == AxisType.EQUIDISTANT:
        return EquidistantAxis(num_points, low, high, **kwargs)
    elif axis_type == AxisType.EQUIDISTANT_PERIODIC:
        return EquidistantAxis(num_points, low, high, periodic=True, **kwargs)
    elif axis_type == AxisType.CHEBYSHEV:
        return ChebyshevAxis(num_points, low, high, **kwargs)
    elif axis_type == AxisType.LOGARITHMIC:
        return LogAxis(num_points, low, high, **kwargs)
    elif axis_type == AxisType.RADIAL:
        return RadialAxis(num_points, low, high, **kwargs)
    elif axis_type == AxisType.COLATITUDE:
        return ColatitudeAxis(num_points, low, high, **kwargs)
    else:
        raise NotImplementedError(f"No such axis type: {axis_type}")


# ── short-hand axis factories ─────────────────────────────────────

def chebyshev(num_points: int, low: float = -1.0, high: float = 1.0,
              name: str | None = None) -> ChebyshevAxis:
    """Chebyshev axis on ``[low, high]`` (spectral accuracy, non-periodic).

    Examples
    --------
    >>> from numgrids import chebyshev
    >>> chebyshev(5, 0, 1).coords.round(3)
    array([0.   , 0.146, 0.5  , 0.854, 1.   ])
    """
    return ChebyshevAxis(num_points, low, high, name=name)


def equidistant(num_points: int, low: float = 0.0, high: float = 1.0,
                name: str | None = None) -> EquidistantAxis:
    """Equidistant, non-periodic axis on ``[low, high]`` (finite differences).

    Examples
    --------
    >>> from numgrids import equidistant
    >>> equidistant(5, 0, 1).coords
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    return EquidistantAxis(num_points, low, high, name=name)


def periodic(num_points: int, low: float = 0.0, high: float = 2 * np.pi,
             name: str | None = None) -> EquidistantAxis:
    """Equidistant, periodic axis on ``[low, high)`` (FFT spectral accuracy).

    The endpoint ``high`` is identified with ``low`` and not included.

    Examples
    --------
    >>> from numgrids import periodic
    >>> periodic(4, 0, 1).coords
    array([0.  , 0.25, 0.5 , 0.75])
    """
    return EquidistantAxis(num_points, low, high, periodic=True, name=name)


def logarithmic(num_points: int, low: float, high: float,
                name: str | None = None) -> LogAxis:
    """Logarithmically spaced axis on ``[low, high]`` (requires ``low > 0``).

    Examples
    --------
    >>> from numgrids import logarithmic
    >>> logarithmic(3, 1, 100).coords
    array([  1.,  10., 100.])
    """
    return LogAxis(num_points, low, high, name=name)


def radial(num_points: int, radius: float = 1.0, name: str | None = None) -> RadialAxis:
    """Radial axis on ``[0, radius]`` that passes through the origin.

    Use it as the radial axis of a :class:`PolarGrid`, :class:`CylindricalGrid`
    or :class:`SphericalGrid` to cover a full disk, cylinder or ball. There is
    no grid point on the origin itself and no boundary condition is needed
    there; see :class:`~numgrids.axes.RadialAxis`.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import PolarGrid, radial, periodic
    >>> disk = PolarGrid(radial(16, 1.0), periodic(32))
    >>> round(disk.integrate(np.ones(disk.shape)), 12) == round(np.pi, 12)
    True
    >>> list(disk.faces)      # only the rim is a boundary
    ['0_high']
    """
    return RadialAxis(num_points, 0.0, radius, name=name)


def colatitude(num_points: int, name: str | None = None) -> ColatitudeAxis:
    """Polar angle on ``(0, pi)`` that passes through both poles.

    Use it as the polar axis of a :class:`SphericalGrid` or
    :class:`SphericalSurfaceGrid` to cover the full sphere including the
    poles. There is no grid point on a pole and no boundary condition is
    needed there; see :class:`~numgrids.axes.ColatitudeAxis`.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import SphericalGrid, colatitude, periodic, radial
    >>> ball = SphericalGrid(radial(12, 1.0), colatitude(12), periodic(24))
    >>> round(ball.integrate(np.ones(ball.shape)), 12) == round(4 / 3 * np.pi, 12)
    True
    >>> list(ball.faces)      # only the surface of the ball is a boundary
    ['0_high']
    """
    return ColatitudeAxis(num_points, 0.0, np.pi, name=name)


# ── convenience functions ─────────────────────────────────────────

def diff(grid: Grid, f: NDArray, order: int = 1, axis_index: int | str = 0, acc: int = 4) -> NDArray:
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
    axis_index : int or str, optional
        Axis (index or name) along which to differentiate (default ``0``).
    acc : int, optional
        Accuracy order for finite-difference methods (default ``4``).

    Returns
    -------
    NDArray
        The derivative array, same shape as *f*.
    """
    axis_index = grid.axis_index(axis_index)
    cache_key = (order, axis_index, acc)
    if cache_key in grid.cache.get("diffs"):
        d = grid.cache["diffs"][cache_key]
    else:
        d = Diff(grid, order, axis_index, acc=acc)
        grid.cache["diffs"][cache_key] = d
    return d(f)


def interpolate(grid: Grid, f: NDArray, locations, method: str = "cubic") -> NDArray:
    """Interpolate a meshed function at arbitrary locations.

    Parameters
    ----------
    grid : Grid
        The grid on which *f* is defined.
    f : NDArray
        Meshed function values.
    locations : float, tuple, list[tuple], array, zip, or Grid
        Point(s) at which to interpolate.
    method : str
        ``"linear"``, ``"cubic"`` (default), ``"quintic"`` or
        ``"spectral"``.  See :class:`~numgrids.interpol.Interpolator`.

    Returns
    -------
    NDArray
        Interpolated value(s).
    """
    from numgrids.interpol import Interpolator
    inter = Interpolator(grid, f, method=method)
    return inter(locations)


def integrate(grid: Grid, f: NDArray) -> float:
    """Integrate a meshed function over the entire grid domain.

    On curvilinear grids the volume element is included, i.e. the result
    is :math:`\\int f\\, dV`.  The :class:`~numgrids.integration.Integral`
    operator is cached on the grid after the first call.

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
    if grid.cache.get("integral") is not None:
        I = grid.cache["integral"]  # noqa: E741
    else:
        I = Integral(grid)  # noqa: E741
        grid.cache["integral"] = I
    return I(f)

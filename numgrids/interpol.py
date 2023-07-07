import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicSpline, UnivariateSpline, interp1d

from numgrids import Grid


class Interpolator:

    def __init__(self, grid, f, method="cubic"):
        """
        Create an interpolating function for the array data.

        Call the Interpolator object like a normal function to interpolate.

        Parameters
        ----------
        grid: Grid
            The grid on which the array f is meshed.
        f: numpy.ndarray
            The data to interpolate.
        method: str
            Interpolation order. May be 'linear', 'quadratic' or 'cubic'.
        """
        self.grid = grid
        if grid.ndims > 1:
            self._inter = RegularGridInterpolator(grid.coords, f, method=method)
        else:
            methods = {
                "linear": 1,
                "quadratic": 2,
                "cubic": 3
            }
            # self._inter = UnivariateSpline(grid.coords, f, k=methods[method])
            self._inter = interp1d(grid.coords, f, kind=method)

    def __call__(self, locations):
        """
        Return the interpolation for one or more points.

        Parameters
        ----------
        locations: tuple, list of tuples, or Grid
            The location(s) where to interpolate. In case of only one point to interpolate,
            pass a tuple with the coordinates of the point. In case of several points, pass
            a list of tuples. Alternatively, you can pass a grid instance. In that case,
            the interpolation will be performed for all grid points of the passed grid.

        Returns
        -------
        The interpolated value or values.
        """

        if isinstance(locations, Grid):
            grid = locations
            locations = np.array([c.reshape(-1) for c in grid.meshed_coords])
            return self._inter(locations.T).reshape(grid.shape)
        else:
            if not hasattr(locations, "__iter__") and not hasattr(locations, "__len__"):
                locations = [locations]
            if isinstance(locations, zip):
                locations = list(locations)
            locations = np.array(locations)
        if locations.ndim == 1:
            return self._inter(locations)[0]
        else:
            return np.array([self._inter(point)[0] for point in locations])

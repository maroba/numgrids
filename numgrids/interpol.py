from types import GeneratorType

import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator, CubicSpline

from numgrids import Grid


class Interpolator:

    def __init__(self, grid, f):
        """
        Create an interpolating function for the array data.

        Call the Interpolator object like a normal function to interpolate.

        Parameters
        ----------
        grid: Grid
            The grid on which the array f is meshed.
        f: numpy.ndarray
            The data to interpolate.
        """
        self.grid = grid
        if grid.ndims > 1:
            self._inter = RegularGridInterpolator(grid.coords, f, method="cubic")
        else:
            self._inter = CubicSpline(grid.coords, f)

    def __call__(self, points):
        """
        Return the interpolation for one or more points.

        Parameters
        ----------
        points

        Returns
        -------

        """

        if isinstance(points, Grid):
            grid = points
            points = np.array([c.reshape(-1) for c in grid.meshed_coords])
            return self._inter(points.T).reshape(grid.shape)
        else:
            if not hasattr(points, "__iter__") and not hasattr(points, "__len__"):
                points = [points]
            if isinstance(points, zip):
                points = list(points)
            points = np.array(points)
        if points.ndim == 1:
            return self._inter(points)[0]
        else:
            return np.array([self._inter(point)[0] for point in points])

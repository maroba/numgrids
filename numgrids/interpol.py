from scipy.interpolate import interpn, RegularGridInterpolator, CubicSpline


class Interpolant:

    def __init__(self, grid, f):
        # self._inter = interpn(grid.coords, f, )
        if grid.ndims > 1:
            self._inter = RegularGridInterpolator(grid.coords, f, method="cubic")
        else:
            self._inter = CubicSpline(grid.coords, f)

    def __call__(self, points):
        return self._inter(points)

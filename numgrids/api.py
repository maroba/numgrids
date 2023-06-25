from numgrids.axes import EquidistantAxis, ChebyshevAxis, LogAxis
from numgrids.diff import FiniteDifferenceDiff, FFTDiff, ChebyshevDiff, LogDiff


class Diff:

    def __init__(self, grid, order, axis_index=0):
        if order <= 0:
            raise ValueError("Derivative order must be positive integer.")

        if axis_index < 0:
            raise ValueError("axis must be nonnegative integer.")

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


class AxisType:
    EQUIDISTANT = "equidistant"
    EQUIDISTANT_PERIODIC = "equidistant_periodic"
    CHEBYSHEV = "chebyshev"
    LOGARITHMIC = "log"


class Axis:

    @classmethod
    def of_type(self, axis_type, num_points, low, high):
        if axis_type == AxisType.EQUIDISTANT:
            return EquidistantAxis(num_points, low, high)
        elif axis_type == AxisType.EQUIDISTANT_PERIODIC:
            return EquidistantAxis(num_points, low, high, periodic=True)
        elif axis_type == AxisType.CHEBYSHEV:
            return ChebyshevAxis(num_points, low, high)
        elif axis_type == AxisType.LOGARITHMIC:
            return LogAxis(num_points, low, high)
        else:
            raise NotImplementedError(f"No such axis type: {axis_type}")

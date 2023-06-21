from numgrids.axes import EquidistantAxis
from numgrids.diff import FiniteDifferenceDiff


class Diff:

    def __init__(self, grid, order, axis_index=0):
        if order <= 0:
            raise ValueError("Derivative order must be positive integer.")

        if axis_index < 0:
            raise ValueError("axis must be nonnegative integer.")

        axis = grid.get_axis(axis_index)
        if isinstance(axis, EquidistantAxis):
            if axis.periodic:
                # TODO implement FFT derivative
                raise NotImplementedError
            self.operator = FiniteDifferenceDiff(grid, order, axis_index)

    def __call__(self, f):
        """Apply the derivative to the array f."""
        return self.operator(f)
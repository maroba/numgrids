from findiff import FinDiff

# Differentiators go here

# For equidistant, periodic axes, use FFT
# For Chebyshev grids, etc.
from numgrids.axes import EquidistantAxis


class FiniteDifferenceDiff:

    def __init__(self, grid, order, axis_index):
        self.order = order
        self.axis = axis_index

        axis = grid.get_axis(axis_index)
        if not isinstance(axis, EquidistantAxis):
            raise TypeError("Axis must be of type EquidistantAxis. Got: {}".format(type(axis)))

        self.operator = FinDiff(axis_index, axis.spacing, order, acc=4)

    def __call__(self, f):
        return self.operator(f)

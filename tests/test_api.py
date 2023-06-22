import unittest

from numgrids.api import Diff
from numgrids.axes import EquidistantAxis
from numgrids.diff import FiniteDifferenceDiff, FFTDiff
from numgrids.grids import Grid


class TestDiff(unittest.TestCase):

    def test_equidistant_nonperiodic(self):
        axis = EquidistantAxis(100, 0, 1)
        grid = Grid(axis, axis)

        d_dx = Diff(grid, 1, 0)

        self.assertEqual(type(d_dx.operator), FiniteDifferenceDiff)

    def test_equidistant_periodic(self):
        axis = EquidistantAxis(100, 0, 1, periodic=True)
        grid = Grid(axis, axis)

        d_dx = Diff(grid, 1, 0)

        self.assertEqual(type(d_dx.operator), FFTDiff)
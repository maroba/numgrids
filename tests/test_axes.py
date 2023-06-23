import unittest

import numpy as np
import numpy.testing as npt

from numgrids.axes import ChebyshevAxis


class TestChebyshevAxis(unittest.TestCase):

    def test_setup_axis(self):
        axis = ChebyshevAxis(10, -1, 1)
        x = axis.coords
        self.assertEqual(x[0], -1)
        self.assertEqual(x[9], 1)
        n = len(axis)
        self.assertAlmostEqual(x[-2], np.cos(np.pi/(n-1)))

    def test_setup_axis_shifted_scaled(self):
        axis = ChebyshevAxis(10, 1, 2)
        x = axis.coords
        self.assertEqual(x[0], 1)
        self.assertEqual(x[9], 2)

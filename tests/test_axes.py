import unittest

import numpy as np
import numpy.testing as npt
from unittest.mock import patch, Mock
from numgrids.axes import ChebyshevAxis, EquidistantAxis


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

    def test_repr(self):
        axis = ChebyshevAxis(10, -1, 1)
        with patch("matplotlib.pyplot.subplots") as mock:
            mock.return_value = Mock(), Mock()
            repr(axis)
            mock.assert_called()


class TestAxis(unittest.TestCase):

    def test_repr(self):
        axis = EquidistantAxis(10, -1, 1, periodic=True)
        with patch("matplotlib.pyplot.subplots") as mock:
            fig, ax = Mock(), Mock()
            mock.return_value = fig, ax
            repr(axis)
            mock.assert_called()
            ax.annotate.assert_called()


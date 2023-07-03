import unittest

import numpy as np
import numpy.testing as npt
from unittest.mock import patch, Mock

from numpy import testing as npt

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


class TestEquidistantAxis(unittest.TestCase):

    def test_init_axis_with_scaling_and_offset(self):

        axis = EquidistantAxis(11, -3, 7)
        self.assertEqual(11, len(axis))
        self.assertEqual(-3, axis[0])
        self.assertEqual(-2, axis[1])
        npt.assert_array_almost_equal(np.linspace(-3, 7, 11), axis.coords)

        with self.assertRaises(IndexError):
            axis[20]

    def test_init_periodic(self):
        axis = EquidistantAxis(10, 0, 1, periodic=True)
        self.assertEqual(10, len(axis))
        self.assertEqual(0, axis[0])
        self.assertEqual(0.9, axis[-1])
        npt.assert_array_almost_equal(np.linspace(0, 1, 10, endpoint=False), axis.coords)

        self.assertEqual(axis[0], axis[10])
        self.assertEqual(axis[1], axis[11])

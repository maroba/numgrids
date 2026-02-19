import unittest

import numpy as np
import numpy.testing as npt
from unittest.mock import patch, Mock

from numgrids.axes import ChebyshevAxis, EquidistantAxis, LogAxis


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
        result = repr(axis)
        self.assertIn("ChebyshevAxis", result)
        self.assertIn("num_points=10", result)


class TestAxis(unittest.TestCase):

    def test_repr(self):
        axis = EquidistantAxis(10, -1, 1, periodic=True)
        result = repr(axis)
        self.assertIn("EquidistantAxis", result)
        self.assertIn("periodic=True", result)

    def test_str(self):
        axis = EquidistantAxis(10, -1, 1)
        result = str(axis)
        self.assertIn("EquidistantAxis", result)
        self.assertIn("10 points", result)

    def test_high_must_be_greater_than_low(self):
        with self.assertRaises(ValueError):
            EquidistantAxis(10, 5, 3)

    def test_num_points_must_be_positive(self):
        with self.assertRaises(ValueError):
            EquidistantAxis(0, 0, 1)
        with self.assertRaises(ValueError):
            EquidistantAxis(-5, 0, 1)

    def test_plot_non_periodic(self):
        axis = EquidistantAxis(10, -1, 1)
        with patch("matplotlib.pyplot.subplots") as mock_subplots, \
             patch("matplotlib.pyplot.show"):
            mock_fig, mock_ax = Mock(), Mock()
            mock_subplots.return_value = mock_fig, mock_ax
            axis.plot()
            mock_subplots.assert_called()

    def test_plot_periodic(self):
        axis = EquidistantAxis(10, 0, 2 * np.pi, periodic=True)
        with patch("matplotlib.pyplot.subplots") as mock_subplots, \
             patch("matplotlib.pyplot.show"):
            mock_fig, mock_ax = Mock(), Mock()
            mock_subplots.return_value = mock_fig, mock_ax
            axis.plot()
            mock_subplots.assert_called()

    def test_plot_chebyshev(self):
        axis = ChebyshevAxis(10, -1, 1)
        with patch("matplotlib.pyplot.subplots") as mock_subplots, \
             patch("matplotlib.pyplot.show"):
            mock_fig, mock_ax = Mock(), Mock()
            mock_subplots.return_value = mock_fig, mock_ax
            axis.plot()
            mock_subplots.assert_called()

    def test_create_diff_operator_equidistant(self):
        from numgrids.grids import Grid
        from numgrids.diff import FiniteDifferenceDiff
        axis = EquidistantAxis(20, 0, 1)
        grid = Grid(axis)
        op = axis.create_diff_operator(grid, 1, 0)
        self.assertIsInstance(op, FiniteDifferenceDiff)

    def test_create_diff_operator_equidistant_periodic(self):
        from numgrids.grids import Grid
        from numgrids.diff import FFTDiff
        axis = EquidistantAxis(20, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        op = axis.create_diff_operator(grid, 1, 0)
        self.assertIsInstance(op, FFTDiff)

    def test_create_diff_operator_chebyshev(self):
        from numgrids.grids import Grid
        from numgrids.diff import ChebyshevDiff
        axis = ChebyshevAxis(20, 0, 1)
        grid = Grid(axis)
        op = axis.create_diff_operator(grid, 1, 0)
        self.assertIsInstance(op, ChebyshevDiff)

    def test_create_diff_operator_log(self):
        from numgrids.grids import Grid
        from numgrids.diff import LogDiff
        axis = LogAxis(20, 0.1, 10)
        grid = Grid(axis)
        op = axis.create_diff_operator(grid, 1, 0)
        self.assertIsInstance(op, LogDiff)

    def test_log_axis_requires_positive_low(self):
        with self.assertRaises(ValueError):
            LogAxis(10, -1, 10)
        with self.assertRaises(ValueError):
            LogAxis(10, 0, 10)


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

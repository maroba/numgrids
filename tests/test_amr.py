"""Tests for adaptive mesh refinement."""

import unittest

import numpy as np
import numpy.testing as npt

from numgrids.grids import Grid
from numgrids.axes import EquidistantAxis, ChebyshevAxis, LogAxis
from numgrids.amr import ErrorEstimator, AdaptationResult, adapt, estimate_error


# ---------------------------------------------------------------------------
# Grid.refine_axis()
# ---------------------------------------------------------------------------

class TestGridRefineAxis(unittest.TestCase):
    """Tests for Grid.refine_axis()."""

    def test_refine_single_axis_2d(self):
        grid = Grid(EquidistantAxis(10, 0, 1), EquidistantAxis(10, 0, 1))
        refined = grid.refine_axis(0, 2.0)
        self.assertEqual(refined.shape, (20, 10))

    def test_refine_second_axis(self):
        grid = Grid(EquidistantAxis(10, 0, 1), EquidistantAxis(10, 0, 1))
        refined = grid.refine_axis(1, 3.0)
        self.assertEqual(refined.shape, (10, 30))

    def test_coarsen_via_factor_less_than_one(self):
        grid = Grid(EquidistantAxis(20, 0, 1), EquidistantAxis(20, 0, 1))
        coarsened = grid.refine_axis(0, 0.5)
        self.assertEqual(coarsened.shape, (10, 20))

    def test_preserves_axis_type_chebyshev(self):
        grid = Grid(ChebyshevAxis(20, -1, 1), EquidistantAxis(20, 0, 1))
        refined = grid.refine_axis(0, 2.0)
        self.assertIsInstance(refined.axes[0], ChebyshevAxis)
        self.assertIsInstance(refined.axes[1], EquidistantAxis)

    def test_preserves_domain(self):
        grid = Grid(ChebyshevAxis(20, -3, 7))
        refined = grid.refine_axis(0, 2.0)
        self.assertAlmostEqual(refined.axes[0].coords[0], -3)
        self.assertAlmostEqual(refined.axes[0].coords[-1], 7)

    def test_preserves_periodic_flag(self):
        grid = Grid(EquidistantAxis(20, 0, 2 * np.pi, periodic=True))
        refined = grid.refine_axis(0, 2.0)
        self.assertTrue(refined.axes[0].periodic)

    def test_preserves_name(self):
        grid = Grid(EquidistantAxis(20, 0, 1, name="x"))
        refined = grid.refine_axis(0, 2.0)
        self.assertEqual(refined.axes[0].name, "x")

    def test_minimum_two_points(self):
        grid = Grid(EquidistantAxis(3, 0, 1))
        coarsened = grid.refine_axis(0, 0.1)
        self.assertEqual(len(coarsened.axes[0]), 2)

    def test_invalid_axis_index_raises(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        with self.assertRaises(ValueError):
            grid.refine_axis(1, 2.0)

    def test_negative_factor_raises(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        with self.assertRaises(ValueError):
            grid.refine_axis(0, -1.0)

    def test_log_axis(self):
        grid = Grid(LogAxis(20, 0.1, 100))
        refined = grid.refine_axis(0, 2.0)
        self.assertIsInstance(refined.axes[0], LogAxis)
        self.assertEqual(len(refined.axes[0]), 40)

    def test_1d_identity(self):
        """Factor of 1.0 should produce the same number of points."""
        grid = Grid(EquidistantAxis(15, 0, 1))
        same = grid.refine_axis(0, 1.0)
        self.assertEqual(same.shape, grid.shape)


# ---------------------------------------------------------------------------
# ErrorEstimator
# ---------------------------------------------------------------------------

class TestErrorEstimator(unittest.TestCase):

    def test_polynomial_well_resolved_on_chebyshev(self):
        """A low-degree polynomial on a Chebyshev grid should have small error."""
        ax = ChebyshevAxis(30, 0, 1)
        grid = Grid(ax)
        est = ErrorEstimator(grid, lambda g: g.meshed_coords[0] ** 2, norm="max")
        err = est.global_error()
        # Linear interpolation limits accuracy; the function itself is exact
        # but the coarse-to-fine comparison involves interpolation error.
        self.assertLess(err, 1e-3)

    def test_high_frequency_needs_refinement(self):
        """sin(50x) on a coarse equidistant grid should show significant error."""
        ax = EquidistantAxis(10, 0, 2 * np.pi)
        grid = Grid(ax)
        est = ErrorEstimator(
            grid, lambda g: np.sin(50 * g.meshed_coords[0]), norm="max"
        )
        err = est.global_error()
        self.assertGreater(err, 0.01)

    def test_per_axis_identifies_hard_axis(self):
        """f(x,y) = sin(50x) + y^2: axis 0 should dominate the error."""
        ax_x = EquidistantAxis(10, 0, 2 * np.pi)
        ax_y = EquidistantAxis(50, 0, 1)
        grid = Grid(ax_x, ax_y)

        def func(g):
            X, Y = g.meshed_coords
            return np.sin(50 * X) + Y ** 2

        est = ErrorEstimator(grid, func, norm="max")
        errors = est.per_axis_errors()
        self.assertGreater(errors[0], errors[1])

    def test_axis_needing_refinement(self):
        ax_x = EquidistantAxis(10, 0, 2 * np.pi)
        ax_y = EquidistantAxis(100, 0, 1)
        grid = Grid(ax_x, ax_y)

        def func(g):
            X, Y = g.meshed_coords
            return np.sin(50 * X) + Y ** 2

        est = ErrorEstimator(grid, func, norm="max")
        worst = est.axis_needing_refinement()
        self.assertEqual(worst, 0)

    def test_invalid_norm_raises(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        with self.assertRaises(ValueError):
            ErrorEstimator(grid, lambda g: g.meshed_coords[0], norm="bogus")

    def test_l2_norm(self):
        ax = EquidistantAxis(10, 0, 2 * np.pi)
        grid = Grid(ax)
        est = ErrorEstimator(
            grid, lambda g: np.sin(50 * g.meshed_coords[0]), norm="l2"
        )
        err = est.global_error()
        self.assertGreater(err, 0.0)

    def test_mean_norm(self):
        ax = EquidistantAxis(10, 0, 2 * np.pi)
        grid = Grid(ax)
        est = ErrorEstimator(
            grid, lambda g: np.sin(50 * g.meshed_coords[0]), norm="mean"
        )
        err = est.global_error()
        self.assertGreater(err, 0.0)

    def test_f_current_cached(self):
        """Accessing f_current twice should return the same object."""
        grid = Grid(EquidistantAxis(10, 0, 1))
        est = ErrorEstimator(grid, lambda g: g.meshed_coords[0] ** 2)
        f1 = est.f_current
        f2 = est.f_current
        self.assertIs(f1, f2)


# ---------------------------------------------------------------------------
# adapt()
# ---------------------------------------------------------------------------

class TestAdapt(unittest.TestCase):

    def test_converges_for_polynomial(self):
        ax_x = EquidistantAxis(5, 0, 1)
        ax_y = EquidistantAxis(5, 0, 1)
        grid = Grid(ax_x, ax_y)

        def func(g):
            X, Y = g.meshed_coords
            return X ** 3 + Y ** 2

        result = adapt(grid, func, tol=1e-4, max_iterations=10)
        self.assertTrue(result.converged)
        self.assertLess(result.global_error, 1e-4)

    def test_asymmetric_refinement(self):
        """Function hard in x, easy in y: x-axis should be refined more."""
        ax_x = EquidistantAxis(5, 0, 1)
        ax_y = EquidistantAxis(5, 0, 1)
        grid = Grid(ax_x, ax_y)

        def func(g):
            X, Y = g.meshed_coords
            return np.sin(20 * X) + Y

        result = adapt(grid, func, tol=1e-3, max_iterations=15)
        self.assertGreater(result.grid.shape[0], result.grid.shape[1])

    def test_returns_adaptation_result(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        result = adapt(grid, lambda g: g.meshed_coords[0] ** 2, tol=1e-3)
        self.assertIsInstance(result, AdaptationResult)
        self.assertEqual(result.f.shape, result.grid.shape)
        self.assertGreater(len(result.history), 0)

    def test_max_iterations_respected(self):
        grid = Grid(EquidistantAxis(5, 0, 1))
        result = adapt(
            grid,
            lambda g: np.sin(100 * g.meshed_coords[0]),
            tol=1e-15,
            max_iterations=3,
        )
        self.assertLessEqual(result.iterations, 3)

    def test_max_points_respected(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        result = adapt(
            grid,
            lambda g: np.sin(100 * g.meshed_coords[0]),
            tol=1e-15,
            max_iterations=50,
            max_points_per_axis=20,
        )
        for ax in result.grid.axes:
            self.assertLessEqual(len(ax), 20)

    def test_works_with_log_axis(self):
        grid = Grid(LogAxis(10, 0.1, 100))
        result = adapt(
            grid,
            lambda g: np.log(g.meshed_coords[0]),
            tol=1e-3,
            max_iterations=10,
        )
        self.assertIsInstance(result.grid.axes[0], LogAxis)

    def test_works_with_chebyshev_axis(self):
        grid = Grid(ChebyshevAxis(5, 0, 1))
        result = adapt(
            grid,
            lambda g: g.meshed_coords[0] ** 5,
            tol=1e-6,
            max_iterations=10,
        )
        self.assertIsInstance(result.grid.axes[0], ChebyshevAxis)

    def test_history_records_iterations(self):
        grid = Grid(EquidistantAxis(5, 0, 1))
        result = adapt(
            grid,
            lambda g: np.sin(10 * g.meshed_coords[0]),
            tol=1e-3,
            max_iterations=5,
        )
        self.assertEqual(len(result.history), result.iterations)
        for record in result.history:
            self.assertIn("iteration", record)
            self.assertIn("shape", record)
            self.assertIn("global_error", record)
            self.assertIn("axis_errors", record)

    def test_refine_all_mode(self):
        """With refine_all=True, both axes should be refined."""
        ax_x = EquidistantAxis(5, 0, 1)
        ax_y = EquidistantAxis(5, 0, 1)
        grid = Grid(ax_x, ax_y)

        def func(g):
            X, Y = g.meshed_coords
            return np.sin(10 * X) + np.sin(10 * Y)

        result = adapt(grid, func, tol=1e-3, max_iterations=10, refine_all=True)
        # Both axes should have grown beyond the initial 5
        self.assertGreater(result.grid.shape[0], 5)
        self.assertGreater(result.grid.shape[1], 5)


# ---------------------------------------------------------------------------
# estimate_error()
# ---------------------------------------------------------------------------

class TestEstimateError(unittest.TestCase):

    def test_returns_expected_keys(self):
        grid = Grid(EquidistantAxis(20, 0, 1))
        result = estimate_error(grid, lambda g: g.meshed_coords[0] ** 2)
        self.assertIn("global", result)
        self.assertIn("per_axis", result)
        self.assertIn(0, result["per_axis"])

    def test_2d_returns_both_axes(self):
        grid = Grid(EquidistantAxis(10, 0, 1), EquidistantAxis(10, 0, 1))

        def func(g):
            X, Y = g.meshed_coords
            return X ** 2 + Y ** 2

        result = estimate_error(grid, func)
        self.assertEqual(len(result["per_axis"]), 2)

    def test_global_error_positive_for_underresolved(self):
        grid = Grid(EquidistantAxis(5, 0, 2 * np.pi))
        result = estimate_error(
            grid, lambda g: np.sin(20 * g.meshed_coords[0])
        )
        self.assertGreater(result["global"], 0.0)

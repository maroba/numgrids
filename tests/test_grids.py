import unittest

import numpy as np
import numpy.testing as npt

from numgrids.grids import Grid, MultiGrid
from numgrids.axes import EquidistantAxis, ChebyshevAxis, LogAxis


class TestGrid(unittest.TestCase):

    def test_init_line_segment(self):
        nx = 11
        x_axis = EquidistantAxis(nx, -3, 7)
        grid = Grid(x_axis)
        x = grid.get_axis().coords
        npt.assert_array_almost_equal(x, np.linspace(-3, 7, nx))

        self.assertAlmostEqual(-3, grid[0])
        self.assertAlmostEqual(-2, grid[1])
        self.assertAlmostEqual(7, grid[-1])

    def test_init_square(self):
        nx = ny = 11
        x_axis = EquidistantAxis(nx, -3, 7)
        y_axis = EquidistantAxis(ny, -3, 7)
        grid = Grid(x_axis, y_axis)
        x = grid.get_axis(0).coords
        y = grid.get_axis(1).coords
        npt.assert_array_almost_equal(x, np.linspace(-3, 7, nx))
        npt.assert_array_almost_equal(y, np.linspace(-3, 7, ny))

        npt.assert_array_almost_equal((-3, -3), grid[0, 0])
        npt.assert_array_almost_equal((-3, 7), grid[0, -1])
        npt.assert_array_almost_equal((7, -2), grid[-1, 1])

    def test_init_circle(self):
        nx = 11
        x_axis = EquidistantAxis(nx, -3, 7, periodic=True)
        grid = Grid(x_axis)
        self.assertEqual(grid[1], grid[12])

    def test_use_meshed(self):
        nx = ny = 11
        x_axis = EquidistantAxis(nx, -3, 7, periodic=True)
        y_axis = EquidistantAxis(ny, -3, 7)
        grid = Grid(x_axis, y_axis)
        x, y = grid.coords

        X, Y = grid.meshed_coords

        f = X ** 2 + Y ** 2

        npt.assert_array_almost_equal(x_axis.coords, x)
        npt.assert_array_almost_equal(y_axis.coords, y)

        X_, Y_ = np.meshgrid(x, y, indexing="ij")
        npt.assert_array_almost_equal(X, X_)
        npt.assert_array_almost_equal(Y, Y_)

        npt.assert_array_almost_equal(f, X_ ** 2 + Y_ ** 2)

    def test_boundary(self):
        nx = ny = 11
        x_axis = EquidistantAxis(nx, -3, 7)
        y_axis = EquidistantAxis(ny, -3, 7)
        grid = Grid(x_axis, y_axis)
        X, Y = grid.meshed_coords

        bdry = np.ones_like(X, dtype=bool)
        bdry[1:-1, 1:-1] = False

        npt.assert_array_equal(grid.boundary, bdry)

    def test_boundary_cheb(self):
        nx = ny = 11
        x_axis = ChebyshevAxis(nx, -3, 7)
        y_axis = ChebyshevAxis(ny, -3, 7)
        grid = Grid(x_axis, y_axis)
        X, Y = grid.meshed_coords

        bdry = np.ones_like(X, dtype=bool)
        bdry[1:-1, 1:-1] = False

        npt.assert_array_equal(grid.boundary, bdry)

    def test_boundary_periodic(self):
        phi_axis = EquidistantAxis(11, 0, 2 * np.pi, periodic=True)
        r_axis = EquidistantAxis(11, 1.E-3, 1)
        grid = Grid(r_axis, phi_axis)
        R, Phi = grid.meshed_coords

        bdry = np.ones_like(R, dtype=bool)
        bdry[1:-1, :] = False

        npt.assert_array_equal(grid.boundary, bdry)

    def test_repr(self):
        nx = 11
        x_axis = EquidistantAxis(nx, -3, 7, periodic=True)
        y_axis = EquidistantAxis(nx, -3, 7, periodic=True)
        grid = Grid(x_axis, y_axis)
        result = repr(grid)
        self.assertIn("Grid", result)
        self.assertIn(str(grid.shape), result)

    def test_refine_grid_default(self):
        x_axis = ChebyshevAxis(10, -3, 7)
        y_axis = ChebyshevAxis(10, -4, 8)
        grid = Grid(x_axis, y_axis)

        fine_grid = grid.refine()
        x, y = fine_grid.coords

        self.assertEqual(20, len(fine_grid.axes[0]))
        self.assertEqual(20, len(fine_grid.axes[1]))
        self.assertEqual(-3, x[0])
        self.assertEqual(7, x[-1])
        self.assertEqual(-4, y[0])
        self.assertEqual(8, y[-1])
        self.assertEqual(type(fine_grid.axes[0]), type(x_axis))

    def test_coarsen_grid_default(self):
        x_axis = ChebyshevAxis(20, -3, 7)
        y_axis = ChebyshevAxis(20, -4, 8)
        grid = Grid(x_axis, y_axis)

        coarse_grid = grid.coarsen()
        x, y = coarse_grid.coords

        self.assertEqual(10, len(coarse_grid.axes[0]))
        self.assertEqual(10, len(coarse_grid.axes[1]))
        self.assertEqual(-3, x[0])
        self.assertEqual(7, x[-1])
        self.assertEqual(-4, y[0])
        self.assertEqual(8, y[-1])
        self.assertEqual(type(coarse_grid.axes[0]), type(x_axis))

    def test_meshed_indices(self):
        x_axis = ChebyshevAxis(20, -3, 7)
        y_axis = ChebyshevAxis(10, -4, 8)
        grid = Grid(x_axis, y_axis)

        I, J = grid.meshed_indices
        self.assertEqual(0, I[0, 0])
        self.assertEqual(19, I[-1, 0])
        self.assertEqual(0, J[5, 0])
        self.assertEqual(9, J[7, -1])

    def test_index_tuples(self):
        x_axis = ChebyshevAxis(20, -3, 7)
        y_axis = ChebyshevAxis(10, -4, 8)
        grid = Grid(x_axis, y_axis)

        inds = grid.index_tuples

        npt.assert_array_equal((0, 0), inds[0, 0])
        npt.assert_array_equal((19, 9), inds[-1, -1])

    def test_coord_tuples(self):
        x_axis = ChebyshevAxis(20, -3, 7)
        y_axis = ChebyshevAxis(10, -4, 8)
        grid = Grid(x_axis, y_axis)

        coord_tuples = grid.coord_tuples

        npt.assert_array_equal((-3, -4), coord_tuples[0, 0])
        npt.assert_array_equal((-3, 8), coord_tuples[0, -1])
        npt.assert_array_equal((7, 8), coord_tuples[-1, -1])


class TestMultiGrid(unittest.TestCase):

    def test_init_multigrid(self):
        axis = EquidistantAxis(10, -1, 1)
        mgrid = MultiGrid(axis, axis, min_size=3)

        self.assertEqual(3, len(mgrid.levels))
        self.assertEqual((10, 10), mgrid.levels[0].shape)
        self.assertEqual((5, 5), mgrid.levels[1].shape)
        self.assertEqual((3, 3), mgrid.levels[2].shape)

    def test_transfer_to_coarse_grid(self):
        xaxis = EquidistantAxis(120, -1, 1)
        yaxis = EquidistantAxis(100, -1, 1)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        grid_0 = mgrid.levels[0]
        X_0, Y_0 = grid_0.meshed_coords

        f_0 = X_0**2 + Y_0**2

        grid_1 = mgrid.levels[1]
        X_1, Y_1 = grid_1.meshed_coords

        f_1 = X_1**2 + Y_1**2
        actual = mgrid.transfer(f_0, 0, 1)

        npt.assert_allclose(actual, f_1, atol=1.E-3)

    def test_transfer_to_fine_grid(self):
        xaxis = EquidistantAxis(120, -1, 1)
        yaxis = EquidistantAxis(100, -1, 1)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        grid_0 = mgrid.levels[0]
        X_0, Y_0 = grid_0.meshed_coords

        f_0 = X_0**2 + Y_0**2

        grid_1 = mgrid.levels[1]
        X_1, Y_1 = grid_1.meshed_coords

        f_1 = X_1**2 + Y_1**2
        actual = mgrid.transfer(f_1, 1, 0)

        npt.assert_allclose(actual, f_0, atol=1.E-3)

    def test_transfer_wrong_shape_raises(self):
        xaxis = EquidistantAxis(20, -1, 1)
        yaxis = EquidistantAxis(20, -1, 1)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        wrong_shape = np.zeros((5, 5))
        with self.assertRaises(ValueError):
            mgrid.transfer(wrong_shape, 0, 1)

    def test_transfer_non_adjacent_raises(self):
        xaxis = EquidistantAxis(40, -1, 1)
        yaxis = EquidistantAxis(40, -1, 1)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        grid_0 = mgrid.levels[0]
        f_0 = np.ones(grid_0.shape)
        with self.assertRaises(ValueError):
            mgrid.transfer(f_0, 0, 2)


class TestMultiGridMixedAxes(unittest.TestCase):
    """Tests for MultiGrid with Chebyshev, Log, periodic, and mixed axis types."""

    # --- ChebyshevAxis MultiGrid ---

    def test_chebyshev_multigrid_creation(self):
        axis = ChebyshevAxis(20, -1, 1)
        mgrid = MultiGrid(axis, axis, min_size=3)

        self.assertGreater(len(mgrid.levels), 1)
        self.assertEqual((20, 20), mgrid.levels[0].shape)
        for level in mgrid.levels:
            for ax in level.axes:
                self.assertIsInstance(ax, ChebyshevAxis)

    def test_chebyshev_domain_preserved(self):
        axis = ChebyshevAxis(40, -2, 5)
        mgrid = MultiGrid(axis, min_size=3)

        for level in mgrid.levels:
            ax = level.axes[0]
            self.assertAlmostEqual(ax.low, -2)
            self.assertAlmostEqual(ax.high, 5)
            self.assertAlmostEqual(ax.coords[0], -2)
            self.assertAlmostEqual(ax.coords[-1], 5)

    def test_chebyshev_transfer_coarsen(self):
        axis = ChebyshevAxis(60, -1, 1)
        mgrid = MultiGrid(axis, axis, min_size=3)

        grid_0 = mgrid.levels[0]
        X_0, Y_0 = grid_0.meshed_coords
        f_0 = X_0 ** 2 + Y_0 ** 2

        grid_1 = mgrid.levels[1]
        X_1, Y_1 = grid_1.meshed_coords
        f_1 = X_1 ** 2 + Y_1 ** 2

        actual = mgrid.transfer(f_0, 0, 1)
        # Linear interpolation on clustered Chebyshev points has larger
        # errors near boundaries where the spacing is very small.
        npt.assert_allclose(actual, f_1, atol=2e-3)

    def test_chebyshev_transfer_refine(self):
        axis = ChebyshevAxis(60, -1, 1)
        mgrid = MultiGrid(axis, axis, min_size=3)

        grid_0 = mgrid.levels[0]
        X_0, Y_0 = grid_0.meshed_coords
        f_0 = X_0 ** 2 + Y_0 ** 2

        grid_1 = mgrid.levels[1]
        X_1, Y_1 = grid_1.meshed_coords
        f_1 = X_1 ** 2 + Y_1 ** 2

        actual = mgrid.transfer(f_1, 1, 0)
        npt.assert_allclose(actual, f_0, atol=1e-2)

    # --- LogAxis MultiGrid ---

    def test_log_multigrid_creation(self):
        axis = LogAxis(20, 0.1, 100)
        mgrid = MultiGrid(axis, min_size=3)

        self.assertGreater(len(mgrid.levels), 1)
        for level in mgrid.levels:
            ax = level.axes[0]
            self.assertIsInstance(ax, LogAxis)

    def test_log_domain_preserved(self):
        axis = LogAxis(40, 0.01, 10)
        mgrid = MultiGrid(axis, min_size=3)

        for level in mgrid.levels:
            ax = level.axes[0]
            self.assertAlmostEqual(ax.low, 0.01)
            self.assertAlmostEqual(ax.high, 10)

    def test_log_transfer_coarsen(self):
        xaxis = LogAxis(80, 0.1, 10)
        yaxis = LogAxis(60, 0.1, 10)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        grid_0 = mgrid.levels[0]
        X_0, Y_0 = grid_0.meshed_coords
        f_0 = np.log(X_0) + np.log(Y_0)

        grid_1 = mgrid.levels[1]
        X_1, Y_1 = grid_1.meshed_coords
        f_1 = np.log(X_1) + np.log(Y_1)

        actual = mgrid.transfer(f_0, 0, 1)
        npt.assert_allclose(actual, f_1, atol=1e-2)

    # --- Periodic EquidistantAxis MultiGrid ---

    def test_periodic_multigrid_preserves_flag(self):
        axis = EquidistantAxis(64, 0, 2 * np.pi, periodic=True)
        mgrid = MultiGrid(axis, min_size=3)

        for level in mgrid.levels:
            ax = level.axes[0]
            self.assertTrue(ax.periodic,
                            f"Periodic flag lost at level with {len(ax)} points")

    def test_periodic_multigrid_domain_preserved(self):
        axis = EquidistantAxis(64, 0, 2 * np.pi, periodic=True)
        mgrid = MultiGrid(axis, min_size=3)

        for level in mgrid.levels:
            ax = level.axes[0]
            self.assertAlmostEqual(ax.low, 0)
            self.assertAlmostEqual(ax.high, 2 * np.pi)

    def test_periodic_transfer_coarsen(self):
        axis = EquidistantAxis(128, 0, 2 * np.pi, periodic=True)
        mgrid = MultiGrid(axis, min_size=3)

        grid_0 = mgrid.levels[0]
        x_0 = grid_0.get_axis(0).coords
        f_0 = np.sin(3 * x_0)

        grid_1 = mgrid.levels[1]
        x_1 = grid_1.get_axis(0).coords
        f_1 = np.sin(3 * x_1)

        actual = mgrid.transfer(f_0, 0, 1)
        npt.assert_allclose(actual, f_1, atol=1e-3)

    # --- Mixed axis types ---

    def test_mixed_chebyshev_equidistant(self):
        xaxis = ChebyshevAxis(40, 0, 1)
        yaxis = EquidistantAxis(40, -1, 1)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        self.assertGreater(len(mgrid.levels), 1)
        for level in mgrid.levels:
            self.assertIsInstance(level.axes[0], ChebyshevAxis)
            self.assertIsInstance(level.axes[1], EquidistantAxis)

    def test_mixed_chebyshev_equidistant_transfer(self):
        xaxis = ChebyshevAxis(60, 0, 1)
        yaxis = EquidistantAxis(80, -1, 1)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        grid_0 = mgrid.levels[0]
        X_0, Y_0 = grid_0.meshed_coords
        f_0 = X_0 ** 2 + Y_0 ** 2

        grid_1 = mgrid.levels[1]
        X_1, Y_1 = grid_1.meshed_coords
        f_1 = X_1 ** 2 + Y_1 ** 2

        actual = mgrid.transfer(f_0, 0, 1)
        npt.assert_allclose(actual, f_1, atol=1e-3)

    def test_mixed_log_chebyshev(self):
        xaxis = LogAxis(40, 0.1, 10)
        yaxis = ChebyshevAxis(40, -1, 1)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        self.assertGreater(len(mgrid.levels), 1)
        for level in mgrid.levels:
            self.assertIsInstance(level.axes[0], LogAxis)
            self.assertIsInstance(level.axes[1], ChebyshevAxis)

    def test_mixed_log_chebyshev_transfer(self):
        xaxis = LogAxis(60, 0.1, 10)
        yaxis = ChebyshevAxis(60, -1, 1)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        grid_0 = mgrid.levels[0]
        X_0, Y_0 = grid_0.meshed_coords
        f_0 = np.log(X_0) + Y_0 ** 2

        grid_1 = mgrid.levels[1]
        X_1, Y_1 = grid_1.meshed_coords
        f_1 = np.log(X_1) + Y_1 ** 2

        actual = mgrid.transfer(f_0, 0, 1)
        npt.assert_allclose(actual, f_1, atol=1e-2)

    def test_mixed_periodic_chebyshev(self):
        xaxis = EquidistantAxis(64, 0, 2 * np.pi, periodic=True)
        yaxis = ChebyshevAxis(40, 0, 1)
        mgrid = MultiGrid(xaxis, yaxis, min_size=3)

        for level in mgrid.levels:
            self.assertIsInstance(level.axes[0], EquidistantAxis)
            self.assertTrue(level.axes[0].periodic)
            self.assertIsInstance(level.axes[1], ChebyshevAxis)

    # --- Axis metadata preservation ---

    def test_name_preserved_across_levels(self):
        axis = EquidistantAxis(40, 0, 1, name="x")
        mgrid = MultiGrid(axis, min_size=3)

        for level in mgrid.levels:
            self.assertEqual(level.axes[0].name, "x")

    def test_name_preserved_chebyshev(self):
        axis = ChebyshevAxis(40, 0, 1, name="radial")
        mgrid = MultiGrid(axis, min_size=3)

        for level in mgrid.levels:
            self.assertEqual(level.axes[0].name, "radial")

    # --- Grid.refine() / Grid.coarsen() with non-equidistant axes ---

    def test_refine_periodic_preserves_flag(self):
        axis = EquidistantAxis(20, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        refined = grid.refine()
        self.assertTrue(refined.axes[0].periodic)
        self.assertEqual(len(refined.axes[0]), 40)
        self.assertAlmostEqual(refined.axes[0].low, 0)
        self.assertAlmostEqual(refined.axes[0].high, 2 * np.pi)

    def test_coarsen_periodic_preserves_flag(self):
        axis = EquidistantAxis(20, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        coarsened = grid.coarsen()
        self.assertTrue(coarsened.axes[0].periodic)
        self.assertEqual(len(coarsened.axes[0]), 10)
        self.assertAlmostEqual(coarsened.axes[0].low, 0)
        self.assertAlmostEqual(coarsened.axes[0].high, 2 * np.pi)

    def test_refine_log_axis(self):
        axis = LogAxis(20, 0.01, 100)
        grid = Grid(axis)
        refined = grid.refine()
        self.assertIsInstance(refined.axes[0], LogAxis)
        self.assertEqual(len(refined.axes[0]), 40)
        self.assertAlmostEqual(refined.axes[0].low, 0.01)
        self.assertAlmostEqual(refined.axes[0].high, 100)

    def test_coarsen_chebyshev_preserves_domain(self):
        axis = ChebyshevAxis(40, -3, 7)
        grid = Grid(axis)
        coarsened = grid.coarsen()
        self.assertIsInstance(coarsened.axes[0], ChebyshevAxis)
        self.assertAlmostEqual(coarsened.axes[0].coords[0], -3)
        self.assertAlmostEqual(coarsened.axes[0].coords[-1], 7)

    def test_refine_mixed_axes(self):
        xaxis = ChebyshevAxis(20, 0, 1, name="x")
        yaxis = EquidistantAxis(20, -1, 1, periodic=True, name="y")
        grid = Grid(xaxis, yaxis)
        refined = grid.refine()

        self.assertIsInstance(refined.axes[0], ChebyshevAxis)
        self.assertEqual(refined.axes[0].name, "x")
        self.assertIsInstance(refined.axes[1], EquidistantAxis)
        self.assertTrue(refined.axes[1].periodic)
        self.assertEqual(refined.axes[1].name, "y")

    # --- Axis.resized() ---

    def test_resized_equidistant(self):
        axis = EquidistantAxis(100, -2, 3, name="x")
        resized = axis.resized(50)
        self.assertIsInstance(resized, EquidistantAxis)
        self.assertEqual(len(resized), 50)
        self.assertAlmostEqual(resized.low, -2)
        self.assertAlmostEqual(resized.high, 3)
        self.assertEqual(resized.name, "x")
        self.assertFalse(resized.periodic)

    def test_resized_equidistant_periodic(self):
        axis = EquidistantAxis(100, 0, 2 * np.pi, periodic=True, name="phi")
        resized = axis.resized(50)
        self.assertIsInstance(resized, EquidistantAxis)
        self.assertEqual(len(resized), 50)
        self.assertAlmostEqual(resized.low, 0)
        self.assertAlmostEqual(resized.high, 2 * np.pi)
        self.assertTrue(resized.periodic)
        self.assertEqual(resized.name, "phi")

    def test_resized_chebyshev(self):
        axis = ChebyshevAxis(30, -1, 1, name="r")
        resized = axis.resized(15)
        self.assertIsInstance(resized, ChebyshevAxis)
        self.assertEqual(len(resized), 15)
        self.assertAlmostEqual(resized.low, -1)
        self.assertAlmostEqual(resized.high, 1)
        self.assertEqual(resized.name, "r")

    def test_resized_log(self):
        axis = LogAxis(50, 0.01, 100, name="r")
        resized = axis.resized(25)
        self.assertIsInstance(resized, LogAxis)
        self.assertEqual(len(resized), 25)
        self.assertAlmostEqual(resized.low, 0.01)
        self.assertAlmostEqual(resized.high, 100)
        self.assertEqual(resized.name, "r")

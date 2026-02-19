import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from numgrids.axes import EquidistantAxis, ChebyshevAxis, LogAxis
from numgrids.grids import Grid
from numgrids.api import SphericalGrid, CylindricalGrid, PolarGrid, create_axis, AxisType
from numgrids.io import save_grid, load_grid


class TestSaveLoadGrid(unittest.TestCase):

    def _roundtrip(self, grid, **data):
        """Save and reload a grid, returning (loaded_grid, loaded_data)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.npz"
            save_grid(path, grid, **data)
            return load_grid(path)

    # ---- basic grid types ----

    def test_equidistant_1d(self):
        grid = Grid(EquidistantAxis(21, -3, 7))
        grid2, data = self._roundtrip(grid)
        self.assertEqual(grid.shape, grid2.shape)
        npt.assert_array_almost_equal(grid.coords, grid2.coords)
        self.assertEqual(data, {})

    def test_equidistant_2d(self):
        grid = Grid(EquidistantAxis(11, 0, 1), EquidistantAxis(15, -2, 2))
        grid2, _ = self._roundtrip(grid)
        self.assertEqual(grid.shape, grid2.shape)
        for i in range(grid.ndims):
            npt.assert_array_almost_equal(
                grid.get_axis(i).coords, grid2.get_axis(i).coords
            )

    def test_equidistant_periodic(self):
        grid = Grid(EquidistantAxis(32, 0, 2 * np.pi, periodic=True))
        grid2, _ = self._roundtrip(grid)
        self.assertEqual(grid.shape, grid2.shape)
        self.assertTrue(grid2.get_axis(0).periodic)
        npt.assert_array_almost_equal(grid.coords, grid2.coords)

    def test_chebyshev_1d(self):
        grid = Grid(ChebyshevAxis(20, -1, 1))
        grid2, _ = self._roundtrip(grid)
        self.assertEqual(grid.shape, grid2.shape)
        npt.assert_array_almost_equal(grid.coords, grid2.coords)

    def test_log_axis(self):
        grid = Grid(LogAxis(30, 0.01, 100))
        grid2, _ = self._roundtrip(grid)
        self.assertEqual(grid.shape, grid2.shape)
        npt.assert_array_almost_equal(grid.coords, grid2.coords)

    def test_mixed_axes(self):
        grid = Grid(
            ChebyshevAxis(15, 0, 1),
            EquidistantAxis(20, 0, 2 * np.pi, periodic=True),
        )
        grid2, _ = self._roundtrip(grid)
        self.assertEqual(grid.shape, grid2.shape)
        self.assertFalse(grid2.get_axis(0).periodic)
        self.assertTrue(grid2.get_axis(1).periodic)
        for i in range(grid.ndims):
            npt.assert_array_almost_equal(
                grid.get_axis(i).coords, grid2.get_axis(i).coords
            )

    def test_3d_grid(self):
        grid = Grid(
            EquidistantAxis(5, 0, 1),
            EquidistantAxis(6, 0, 1),
            EquidistantAxis(7, 0, 1),
        )
        grid2, _ = self._roundtrip(grid)
        self.assertEqual(grid.shape, grid2.shape)

    # ---- specialized grids ----

    def test_spherical_grid(self):
        grid = SphericalGrid(
            ChebyshevAxis(10, 0.1, 5),
            ChebyshevAxis(8, 0.1, np.pi - 0.1),
            EquidistantAxis(12, 0, 2 * np.pi, periodic=True),
        )
        grid2, _ = self._roundtrip(grid)
        self.assertIsInstance(grid2, SphericalGrid)
        self.assertEqual(grid.shape, grid2.shape)

    def test_cylindrical_grid(self):
        grid = CylindricalGrid(
            ChebyshevAxis(10, 0.1, 3),
            EquidistantAxis(16, 0, 2 * np.pi, periodic=True),
            ChebyshevAxis(10, -1, 1),
        )
        grid2, _ = self._roundtrip(grid)
        self.assertIsInstance(grid2, CylindricalGrid)
        self.assertEqual(grid.shape, grid2.shape)

    def test_polar_grid(self):
        grid = PolarGrid(
            ChebyshevAxis(15, 0.1, 1),
            EquidistantAxis(20, 0, 2 * np.pi, periodic=True),
        )
        grid2, _ = self._roundtrip(grid)
        self.assertIsInstance(grid2, PolarGrid)
        self.assertEqual(grid.shape, grid2.shape)

    # ---- data arrays ----

    def test_save_load_single_array(self):
        grid = Grid(EquidistantAxis(11, 0, 1), EquidistantAxis(11, 0, 1))
        X, Y = grid.meshed_coords
        f = X ** 2 + Y ** 2
        grid2, data = self._roundtrip(grid, f=f)
        npt.assert_array_almost_equal(data["f"], f)

    def test_save_load_multiple_arrays(self):
        grid = Grid(EquidistantAxis(11, 0, 1))
        x = grid.coords
        grid2, data = self._roundtrip(grid, u=x ** 2, v=np.sin(x))
        npt.assert_array_almost_equal(data["u"], x ** 2)
        npt.assert_array_almost_equal(data["v"], np.sin(x))

    def test_no_data_returns_empty_dict(self):
        grid = Grid(EquidistantAxis(5, 0, 1))
        _, data = self._roundtrip(grid)
        self.assertEqual(data, {})

    def test_wrong_shape_raises(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmp:
                save_grid(Path(tmp) / "bad.npz", grid, f=np.zeros((5,)))

    # ---- axis name preservation ----

    def test_axis_name_preserved(self):
        grid = Grid(
            EquidistantAxis(10, 0, 1, name="x"),
            ChebyshevAxis(10, 0, 1, name="y"),
        )
        grid2, _ = self._roundtrip(grid)
        self.assertEqual(grid2.get_axis(0).name, "x")
        self.assertEqual(grid2.get_axis(1).name, "y")

    def test_axis_name_none(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        grid2, _ = self._roundtrip(grid)
        self.assertIsNone(grid2.get_axis(0).name)

    # ---- boundary mask preserved ----

    def test_boundary_mask_preserved(self):
        grid = Grid(EquidistantAxis(11, 0, 1), EquidistantAxis(11, 0, 1))
        grid2, _ = self._roundtrip(grid)
        npt.assert_array_equal(grid.boundary, grid2.boundary)

    # ---- path handling ----

    def test_npz_suffix_added(self):
        grid = Grid(EquidistantAxis(5, 0, 1))
        with tempfile.TemporaryDirectory() as tmp:
            save_grid(Path(tmp) / "grid", grid)
            # np.savez adds .npz automatically
            self.assertTrue((Path(tmp) / "grid.npz").exists())

    def test_load_without_suffix(self):
        grid = Grid(EquidistantAxis(5, 0, 1))
        with tempfile.TemporaryDirectory() as tmp:
            save_grid(Path(tmp) / "grid.npz", grid)
            grid2, _ = load_grid(Path(tmp) / "grid")
            self.assertEqual(grid.shape, grid2.shape)

    def test_string_path(self):
        grid = Grid(EquidistantAxis(5, 0, 1))
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "grid.npz")
            save_grid(p, grid)
            grid2, _ = load_grid(p)
            self.assertEqual(grid.shape, grid2.shape)

    # ---- operators still work after reload ----

    def test_diff_on_loaded_grid(self):
        """Verify that derivative operators work on a reloaded grid."""
        from numgrids.api import Diff
        grid = Grid(ChebyshevAxis(30, 0, 1))
        grid2, _ = self._roundtrip(grid)
        x = grid2.coords
        f = x ** 3
        df = Diff(grid2, 1, 0)(f)
        npt.assert_array_almost_equal(df, 3 * x ** 2, decimal=5)

    def test_laplacian_on_loaded_spherical_grid(self):
        """Verify that laplacian works on a reloaded SphericalGrid."""
        grid = SphericalGrid(
            ChebyshevAxis(15, 0.5, 3),
            ChebyshevAxis(12, 0.3, np.pi - 0.3),
            EquidistantAxis(16, 0, 2 * np.pi, periodic=True),
        )
        R, Theta, Phi = grid.meshed_coords
        f = R ** 2

        grid2, _ = self._roundtrip(grid)
        R2, _, _ = grid2.meshed_coords
        f2 = R2 ** 2

        lap1 = grid.laplacian(f)
        lap2 = grid2.laplacian(f2)
        npt.assert_array_almost_equal(lap1, lap2, decimal=10)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import numpy.testing as npt
from scipy.sparse.linalg import spsolve

from numgrids.axes import EquidistantAxis, ChebyshevAxis
from numgrids.grids import Grid
from numgrids.api import Diff
from numgrids.boundary import (
    BoundaryFace,
    DirichletBC,
    NeumannBC,
    RobinBC,
    apply_bcs,
)


# -----------------------------------------------------------------------
# BoundaryFace
# -----------------------------------------------------------------------

class TestBoundaryFace(unittest.TestCase):

    def test_face_mask_1d(self):
        grid = Grid(EquidistantAxis(11, 0, 1))
        low = BoundaryFace(grid, 0, "low")
        high = BoundaryFace(grid, 0, "high")

        expected_low = np.zeros(11, dtype=bool)
        expected_low[0] = True
        npt.assert_array_equal(low.mask, expected_low)

        expected_high = np.zeros(11, dtype=bool)
        expected_high[-1] = True
        npt.assert_array_equal(high.mask, expected_high)

    def test_face_mask_2d(self):
        grid = Grid(EquidistantAxis(5, 0, 1), EquidistantAxis(7, 0, 1))

        m0_low = BoundaryFace(grid, 0, "low").mask
        self.assertEqual(m0_low.shape, (5, 7))
        self.assertTrue(np.all(m0_low[0, :]))
        self.assertFalse(np.any(m0_low[1:, :]))

        m0_high = BoundaryFace(grid, 0, "high").mask
        self.assertTrue(np.all(m0_high[-1, :]))
        self.assertFalse(np.any(m0_high[:-1, :]))

        m1_low = BoundaryFace(grid, 1, "low").mask
        self.assertTrue(np.all(m1_low[:, 0]))
        self.assertFalse(np.any(m1_low[:, 1:]))

        m1_high = BoundaryFace(grid, 1, "high").mask
        self.assertTrue(np.all(m1_high[:, -1]))
        self.assertFalse(np.any(m1_high[:, :-1]))

    def test_face_mask_3d(self):
        grid = Grid(
            EquidistantAxis(4, 0, 1),
            EquidistantAxis(5, 0, 1),
            EquidistantAxis(6, 0, 1),
        )
        face = BoundaryFace(grid, 1, "high")
        self.assertEqual(np.sum(face.mask), 4 * 6)  # 4 x 6 face

    def test_periodic_axis_excluded(self):
        with self.assertRaises(ValueError):
            BoundaryFace(
                Grid(EquidistantAxis(10, 0, 2 * np.pi, periodic=True)),
                0,
                "low",
            )

    def test_flat_indices_match_mask(self):
        grid = Grid(EquidistantAxis(5, 0, 1), EquidistantAxis(7, 0, 1))
        face = BoundaryFace(grid, 0, "high")
        npt.assert_array_equal(face.flat_indices, np.flatnonzero(face.mask))

    def test_normal_sign(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        self.assertEqual(BoundaryFace(grid, 0, "low").normal_sign, -1)
        self.assertEqual(BoundaryFace(grid, 0, "high").normal_sign, 1)

    def test_faces_union_equals_boundary(self):
        grid = Grid(EquidistantAxis(5, 0, 1), EquidistantAxis(7, 0, 1))
        union = np.zeros(grid.shape, dtype=bool)
        for face in grid.faces.values():
            union |= face.mask
        npt.assert_array_equal(union, grid.boundary)

    def test_invalid_side_raises(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        with self.assertRaises(ValueError):
            BoundaryFace(grid, 0, "middle")

    def test_invalid_axis_raises(self):
        grid = Grid(EquidistantAxis(10, 0, 1))
        with self.assertRaises(ValueError):
            BoundaryFace(grid, 1, "low")


# -----------------------------------------------------------------------
# Grid.faces property
# -----------------------------------------------------------------------

class TestGridFacesProperty(unittest.TestCase):

    def test_faces_count_nonperiodic(self):
        grid = Grid(EquidistantAxis(5, 0, 1), EquidistantAxis(7, 0, 1))
        self.assertEqual(len(grid.faces), 4)

    def test_faces_with_periodic(self):
        grid = Grid(
            EquidistantAxis(10, 0, 1),
            EquidistantAxis(20, 0, 2 * np.pi, periodic=True),
        )
        self.assertEqual(len(grid.faces), 2)
        self.assertIn("0_low", grid.faces)
        self.assertIn("0_high", grid.faces)

    def test_faces_caching(self):
        grid = Grid(EquidistantAxis(5, 0, 1))
        a = grid.faces
        b = grid.faces
        self.assertIs(a, b)


# -----------------------------------------------------------------------
# DirichletBC
# -----------------------------------------------------------------------

class TestDirichletBC(unittest.TestCase):

    def test_apply_constant_1d(self):
        grid = Grid(EquidistantAxis(11, 0, 1))
        u = np.zeros(grid.shape)
        DirichletBC(grid.faces["0_low"], value=5.0).apply(u)
        self.assertAlmostEqual(u[0], 5.0)
        self.assertAlmostEqual(u[1], 0.0)

    def test_apply_constant_2d(self):
        grid = Grid(EquidistantAxis(5, 0, 1), EquidistantAxis(7, 0, 1))
        u = np.zeros(grid.shape)
        DirichletBC(grid.faces["0_high"], value=3.0).apply(u)
        npt.assert_array_almost_equal(u[-1, :], 3.0)
        npt.assert_array_almost_equal(u[:-1, :], 0.0)

    def test_apply_callable_2d(self):
        grid = Grid(EquidistantAxis(5, 0, 1), EquidistantAxis(7, 0, 1))
        u = np.zeros(grid.shape)
        # On the 1_high face (y = 1), set u = x^2
        DirichletBC(
            grid.faces["1_high"],
            value=lambda coords: coords[0] ** 2,
        ).apply(u)
        X, Y = grid.meshed_coords
        npt.assert_array_almost_equal(u[:, -1], X[:, -1] ** 2)

    def test_apply_array_value(self):
        grid = Grid(EquidistantAxis(5, 0, 1))
        u = np.zeros(grid.shape)
        DirichletBC(grid.faces["0_low"], value=np.array([42.0])).apply(u)
        self.assertAlmostEqual(u[0], 42.0)

    def test_apply_preserves_interior(self):
        grid = Grid(EquidistantAxis(11, 0, 1))
        u = np.ones(grid.shape) * 7.0
        DirichletBC(grid.faces["0_low"], value=0.0).apply(u)
        npt.assert_array_almost_equal(u[1:], 7.0)

    def test_apply_to_system_1d_poisson(self):
        """u'' = -2, u(0) = 0, u(1) = 0  =>  u = x(1-x)."""
        grid = Grid(EquidistantAxis(101, 0, 1))
        x = grid.coords
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.full(grid.size, -2.0)

        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 0.0),
                DirichletBC(grid.faces["0_high"], 0.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        npt.assert_array_almost_equal(u, x * (1 - x), decimal=4)

    def test_apply_to_system_2d_poisson(self):
        """Laplacian(u) = -2 on [0,1]^2, u = 0 on all boundaries.

        Check that interior values are positive (the solution of the Poisson
        problem with negative constant RHS and zero Dirichlet BCs is a
        positive bump).
        """
        n = 31
        grid = Grid(EquidistantAxis(n, 0, 1), EquidistantAxis(n, 0, 1))
        Dxx = Diff(grid, 2, 0).as_matrix()
        Dyy = Diff(grid, 2, 1).as_matrix()
        L = Dxx + Dyy
        rhs = np.full(grid.size, -2.0)

        bcs = [DirichletBC(grid.faces[k], 0.0) for k in grid.faces]
        L, rhs = apply_bcs(bcs, L, rhs)
        u = spsolve(L, rhs).reshape(grid.shape)

        # Boundary should be zero
        npt.assert_array_almost_equal(u[grid.boundary], 0.0, decimal=10)
        # Interior should be positive
        self.assertTrue(np.all(u[~grid.boundary] > 0))


# -----------------------------------------------------------------------
# NeumannBC
# -----------------------------------------------------------------------

class TestNeumannBC(unittest.TestCase):

    def test_apply_to_system_constant_solution(self):
        """u'' = 0, u(0) = 1, u'(1) = 0  =>  u = 1 everywhere."""
        grid = Grid(EquidistantAxis(51, 0, 1))
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.zeros(grid.size)

        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 1.0),
                NeumannBC(grid.faces["0_high"], 0.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        npt.assert_array_almost_equal(u, 1.0, decimal=4)

    def test_apply_to_system_linear_solution(self):
        """u'' = 0, u(0) = 0, u'(1) = 1  =>  u = x."""
        grid = Grid(EquidistantAxis(51, 0, 1))
        x = grid.coords
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.zeros(grid.size)

        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 0.0),
                NeumannBC(grid.faces["0_high"], 1.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        npt.assert_array_almost_equal(u, x, decimal=3)

    def test_neumann_normal_direction(self):
        """u'' = 0, u'(0) = 1 (outward = -x, so du/dn = -du/dx = 1 => du/dx = -1),
        u(1) = 0  =>  u = 1 - x  (since du/dx = -1 and u(1) = 0)."""
        grid = Grid(EquidistantAxis(51, 0, 1))
        x = grid.coords
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.zeros(grid.size)

        L, rhs = apply_bcs(
            [
                NeumannBC(grid.faces["0_low"], 1.0),
                DirichletBC(grid.faces["0_high"], 0.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        npt.assert_array_almost_equal(u, 1.0 - x, decimal=3)

    def test_apply_function_level(self):
        """Verify function-level apply modifies the boundary."""
        grid = Grid(EquidistantAxis(11, 0, 1))
        u = np.ones(grid.shape) * 5.0
        NeumannBC(grid.faces["0_high"], 0.0).apply(u)
        # With du/dn = 0 at high end, u[-1] should be set to u[-2]
        self.assertAlmostEqual(u[-1], u[-2], places=10)


# -----------------------------------------------------------------------
# RobinBC
# -----------------------------------------------------------------------

class TestRobinBC(unittest.TestCase):

    def test_robin_reduces_to_dirichlet(self):
        """Robin with a=1, b=0 should act like Dirichlet."""
        grid = Grid(EquidistantAxis(51, 0, 1))
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.full(grid.size, -2.0)

        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 0.0),
                RobinBC(grid.faces["0_high"], a=1.0, b=0.0, value=0.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        x = grid.coords
        npt.assert_array_almost_equal(u, x * (1 - x), decimal=3)

    def test_robin_reduces_to_neumann(self):
        """Robin with a=0, b=1 should act like Neumann."""
        grid = Grid(EquidistantAxis(51, 0, 1))
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.zeros(grid.size)

        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 1.0),
                RobinBC(grid.faces["0_high"], a=0.0, b=1.0, value=0.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        npt.assert_array_almost_equal(u, 1.0, decimal=3)

    def test_apply_to_system_robin(self):
        """u'' = 0, u(0) = 0, u(1) + u'(1) = 1  =>  u = x/2."""
        grid = Grid(EquidistantAxis(101, 0, 1))
        x = grid.coords
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.zeros(grid.size)

        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 0.0),
                RobinBC(grid.faces["0_high"], a=1.0, b=1.0, value=1.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        npt.assert_array_almost_equal(u, x / 2, decimal=3)


# -----------------------------------------------------------------------
# apply_bcs convenience function
# -----------------------------------------------------------------------

class TestApplyBcs(unittest.TestCase):

    def test_apply_multiple_bcs(self):
        """Dirichlet left, Neumann right: u'' = 0, u(0)=2, u'(1)=0 => u=2."""
        grid = Grid(EquidistantAxis(51, 0, 1))
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.zeros(grid.size)

        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 2.0),
                NeumannBC(grid.faces["0_high"], 0.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        npt.assert_array_almost_equal(u, 2.0, decimal=4)

    def test_corner_point_last_bc_wins(self):
        """Two Dirichlet BCs on adjacent faces: the last one applied wins at
        the shared corner point."""
        import scipy.sparse
        grid = Grid(EquidistantAxis(5, 0, 1), EquidistantAxis(5, 0, 1))
        L = scipy.sparse.eye(grid.size, format="csc")
        rhs = np.zeros(grid.size)

        # Apply axis-0 low face first (value=1), then axis-1 low face (value=2)
        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 1.0),
                DirichletBC(grid.faces["1_low"], 2.0),
            ],
            L,
            rhs,
        )
        # Corner (0, 0) is on both faces; last BC (value=2) should win
        corner_flat_idx = 0  # (0, 0) in row-major is index 0
        self.assertAlmostEqual(rhs[corner_flat_idx], 2.0)


# -----------------------------------------------------------------------
# Chebyshev grids
# -----------------------------------------------------------------------

class TestWithChebyshevAxis(unittest.TestCase):

    def test_dirichlet_chebyshev_1d(self):
        """u'' = -2, u(0)=0, u(1)=0 on Chebyshev grid => u = x(1-x)."""
        grid = Grid(ChebyshevAxis(30, 0, 1))
        x = grid.coords
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.full(grid.size, -2.0)

        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 0.0),
                DirichletBC(grid.faces["0_high"], 0.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        npt.assert_array_almost_equal(u, x * (1 - x), decimal=5)

    def test_neumann_chebyshev_1d(self):
        """u'' = 0, u(0)=0, u'(1)=1 on Chebyshev grid => u = x."""
        grid = Grid(ChebyshevAxis(30, 0, 1))
        x = grid.coords
        L = Diff(grid, 2, 0).as_matrix()
        rhs = np.zeros(grid.size)

        L, rhs = apply_bcs(
            [
                DirichletBC(grid.faces["0_low"], 0.0),
                NeumannBC(grid.faces["0_high"], 1.0),
            ],
            L,
            rhs,
        )
        u = spsolve(L, rhs)
        npt.assert_array_almost_equal(u, x, decimal=4)

    def test_poisson_chebyshev_2d(self):
        """2D Poisson on Chebyshev grid with zero Dirichlet BCs."""
        n = 20
        grid = Grid(ChebyshevAxis(n, 0, 1), ChebyshevAxis(n, 0, 1))
        Dxx = Diff(grid, 2, 0).as_matrix()
        Dyy = Diff(grid, 2, 1).as_matrix()
        L = Dxx + Dyy
        rhs = np.full(grid.size, -2.0)

        bcs = [DirichletBC(grid.faces[k], 0.0) for k in grid.faces]
        L, rhs = apply_bcs(bcs, L, rhs)
        u = spsolve(L, rhs).reshape(grid.shape)

        npt.assert_array_almost_equal(u[grid.boundary], 0.0, decimal=10)
        self.assertTrue(np.all(u[~grid.boundary] > 0))


if __name__ == "__main__":
    unittest.main()

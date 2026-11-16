import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse

from numgrids import (
    Diff,
    Dirichlet,
    Grid,
    Neumann,
    Operator,
    PolarGrid,
    Robin,
    SphericalGrid,
    chebyshev,
    equidistant,
    periodic,
)


@pytest.fixture
def grid1d():
    return Grid(x=chebyshev(30, 0, np.pi))


@pytest.fixture
def grid2d():
    return Grid(x=chebyshev(20, 0, 1), y=chebyshev(20, 0, 1))


class TestAlgebra:

    def test_apply(self, grid1d):
        op = grid1d.diff_operator(2)
        x = grid1d.coords
        npt.assert_allclose(op(np.sin(x)), -np.sin(x), atol=1e-8)

    def test_scalar_addition_and_scaling(self, grid1d):
        x = grid1d.coords
        f = np.sin(x)
        L = 2 * grid1d.diff_operator(2) + 3
        npt.assert_allclose(L(f), -2 * f + 3 * f, atol=1e-8)
        npt.assert_allclose((3 - grid1d.diff_operator(2))(f), 4 * f, atol=1e-8)
        npt.assert_allclose((grid1d.diff_operator(2) / 2)(f), -f / 2, atol=1e-8)
        npt.assert_allclose((-grid1d.diff_operator(2))(f), f, atol=1e-8)

    def test_array_multiplication_order(self, grid1d):
        x = grid1d.coords
        D = grid1d.diff_operator(1)
        f = np.sin(x)
        # V * D: differentiate, then multiply
        npt.assert_allclose((x * D)(f), x * np.cos(x), atol=1e-8)
        # D * V: multiply, then differentiate
        npt.assert_allclose((D * x)(f), np.sin(x) + x * np.cos(x), atol=1e-8)

    def test_array_addition(self, grid1d):
        x = grid1d.coords
        f = np.sin(x)
        op = grid1d.diff_operator(2) + x ** 2
        npt.assert_allclose(op(f), -f + x ** 2 * f, atol=1e-8)

    def test_composition_and_power(self, grid1d):
        D = grid1d.diff_operator(1)
        D2 = grid1d.diff_operator(2)
        f = np.sin(grid1d.coords)
        npt.assert_allclose((D @ D)(f), D2(f), atol=1e-8)
        npt.assert_allclose((D * D)(f), D2(f), atol=1e-8)
        npt.assert_allclose((D ** 2)(f), D2(f), atol=1e-8)
        npt.assert_allclose((D ** 0)(f), f)

    def test_diff_objects_combine(self, grid2d):
        X, Y = grid2d.meshed_coords
        f = X ** 2 * Y ** 3
        L = Diff(grid2d, 2, "x") + Diff(grid2d, 2, "y")
        assert isinstance(L, Operator)
        npt.assert_allclose(L(f), 2 * Y ** 3 + 6 * X ** 2 * Y, atol=1e-8)
        npt.assert_allclose((2 * Diff(grid2d, 1, 0))(f), 4 * X * Y ** 3, atol=1e-8)
        npt.assert_allclose((Diff(grid2d, 1, 0) * X)(f), 3 * X ** 2 * Y ** 3, atol=1e-8)
        npt.assert_allclose((X * Diff(grid2d, 1, 0))(f), 2 * X ** 2 * Y ** 3, atol=1e-8)
        npt.assert_allclose((-Diff(grid2d, 1, 0))(f), -2 * X * Y ** 3, atol=1e-8)
        npt.assert_allclose((Diff(grid2d, 1, 0) ** 2)(f), 2 * Y ** 3, atol=1e-8)

    def test_laplacian_operator_cartesian(self, grid2d):
        X, Y = grid2d.meshed_coords
        f = np.sin(X) * np.cosh(Y)
        npt.assert_allclose(grid2d.laplacian_operator()(f), 0, atol=1e-8)
        assert grid2d.laplacian_operator() is grid2d.laplacian_operator()

    def test_identity_and_multiplication(self, grid2d):
        X, _ = grid2d.meshed_coords
        npt.assert_array_equal(Operator.identity(grid2d)(X), X)
        npt.assert_array_equal(Operator.multiplication(grid2d, X)(X), X ** 2)

    def test_shape_mismatch(self, grid2d):
        with pytest.raises(ValueError):
            Operator(grid2d, scipy.sparse.identity(5))
        with pytest.raises(ValueError):
            grid2d.laplacian_operator() + np.ones(3)
        with pytest.raises(ValueError):
            grid2d.laplacian_operator()(np.ones(3))

    def test_incompatible_grids(self, grid2d):
        other = Grid(chebyshev(10), chebyshev(10))
        with pytest.raises(ValueError):
            grid2d.laplacian_operator() + other.laplacian_operator()

    def test_unsupported_operand(self, grid2d):
        with pytest.raises(TypeError):
            grid2d.laplacian_operator() + "a"

    def test_repr(self, grid2d):
        assert "Operator" in repr(grid2d.laplacian_operator())


class TestSolve:

    def test_dirichlet_scalar(self, grid1d):
        x = grid1d.coords
        u = grid1d.diff_operator(2).solve(-np.sin(x), bcs=0)
        npt.assert_allclose(u, np.sin(x), atol=1e-10)

    def test_callable_rhs_and_dict_bcs(self, grid2d):
        # u = x^2 + y^2 -> Laplacian 4
        L = grid2d.laplacian_operator()
        exact = lambda x, y: x ** 2 + y ** 2  # noqa: E731
        u = L.solve(lambda x, y: 4 + 0 * x, bcs={"all": lambda c: c[0] ** 2 + c[1] ** 2})
        npt.assert_allclose(u, grid2d.evaluate(exact), atol=1e-9)

    def test_neumann_and_robin(self):
        grid = Grid(x=chebyshev(20, 0, 1))
        x = grid.coords
        L = grid.diff_operator(2)
        # u = x^2 : u'' = 2, u(0) = 0, u'(1) = 2
        u = L.solve(2, bcs={"x_low": Dirichlet(0), "x_high": Neumann(2)})
        npt.assert_allclose(u, x ** 2, atol=1e-10)
        # Robin at x=0: u - u_n = u + u' = 0 + 0, outward normal is -x so u_n = -u'
        # choose a=1, b=1: u(0) + (-u'(0)) = 0
        u = L.solve(2, bcs={"x_low": Robin(1, 1, 0), "x_high": 1})
        npt.assert_allclose(u, x ** 2, atol=1e-10)

    def test_periodic_helmholtz_without_bcs(self):
        grid = Grid(periodic(32))
        x = grid.coords
        L = grid.diff_operator(2) - 1
        u = L.solve(-2 * np.sin(x))
        npt.assert_allclose(u, np.sin(x), atol=1e-10)

    def test_polar_poisson(self):
        grid = PolarGrid(chebyshev(25, 0.2, 1), periodic(24))
        R, Phi = grid.meshed_coords
        exact = R ** 2 * np.cos(2 * Phi) + R ** 2
        L = grid.laplacian_operator()
        u = L.solve(4.0, bcs=lambda c: c[0] ** 2 * np.cos(2 * c[1]) + c[0] ** 2)
        npt.assert_allclose(u, exact, atol=1e-8)

    def test_list_bcs(self, grid1d):
        from numgrids import DirichletBC
        bcs = [DirichletBC(f, 0) for f in grid1d.faces.values()]
        u = grid1d.diff_operator(2).solve(-np.sin(grid1d.coords), bcs=bcs)
        npt.assert_allclose(u, np.sin(grid1d.coords), atol=1e-10)

    def test_with_bcs(self, grid1d):
        L, b = grid1d.diff_operator(2).with_bcs(1.0, bcs=0)
        assert L.shape == (grid1d.size, grid1d.size)
        assert b[0] == 0 and b[-1] == 0 and b[1] == 1


class TestEig:

    def test_dirichlet_1d(self, grid1d):
        vals, vecs = (-grid1d.diff_operator(2)).eig(k=4, bcs=0)
        npt.assert_allclose(vals, [1, 4, 9, 16], rtol=1e-8)
        assert vecs.shape == (4,) + grid1d.shape
        npt.assert_allclose(vecs[0], np.sin(grid1d.coords) / np.linalg.norm(np.sin(grid1d.coords)),
                            atol=1e-8)
        assert np.isrealobj(vals)

    def test_neumann_1d_no_spurious_eigenvalues(self):
        grid = Grid(chebyshev(30, 0, np.pi))
        vals, _ = (-grid.diff_operator(2)).eig(k=4, bcs=Neumann(0))
        npt.assert_allclose(vals, [0, 1, 4, 9], atol=1e-8)

    def test_sigma(self, grid1d):
        vals, _ = (-grid1d.diff_operator(2)).eig(k=2, bcs=0, sigma=12)
        npt.assert_allclose(sorted(vals), [9, 16], rtol=1e-8)

    def test_which(self, grid1d):
        vals, _ = grid1d.diff_operator(2).eig(k=3, bcs=0, which="LR")
        npt.assert_allclose(vals, [-1, -4, -9], rtol=1e-8)
        with pytest.raises(ValueError):
            grid1d.diff_operator(2).eig(which="XX")

    def test_quantum_harmonic_oscillator(self):
        grid = Grid(x=chebyshev(60, -10, 10))
        x = grid.coords
        H = -0.5 * grid.diff_operator(2) + 0.5 * x ** 2
        vals, _ = H.eig(k=5, bcs=0)
        npt.assert_allclose(vals, np.arange(5) + 0.5, atol=1e-8)

    def test_sparse_path_matches_dense(self):
        grid = Grid(equidistant(26, 0, np.pi), equidistant(26, 0, np.pi))
        L = -grid.laplacian_operator()
        dense_vals, _ = L.eig(k=3, bcs=0)
        sparse_vals, _ = L.eig(k=3, bcs=0, dense_threshold=10)
        npt.assert_allclose(sparse_vals, dense_vals, rtol=1e-8)
        npt.assert_allclose(dense_vals, [2, 5, 5], rtol=1e-3)

    def test_sparse_generalized_needs_sigma_for_lm(self):
        grid = Grid(equidistant(40, 0, 1), equidistant(40, 0, 1))
        L = grid.laplacian_operator()
        with pytest.raises(NotImplementedError):
            L.eig(k=2, bcs=Neumann(0), which="LM", dense_threshold=10)
        vals, _ = L.eig(k=2, bcs=Neumann(0), sigma=-1.0, dense_threshold=10)
        assert np.all(np.isfinite(vals))

    def test_sphere_eigenvalues(self):
        # Dirichlet Laplacian in a spherical shell, l = 0 modes: k = n pi / (b - a)
        grid = SphericalGrid(chebyshev(24, 1, 2), chebyshev(6, 0.3, np.pi - 0.3), periodic(4))
        L = -grid.laplacian_operator()
        vals, vecs = L.eig(k=1, bcs={"r_low": 0, "r_high": 0, "theta_low": Neumann(0),
                                     "theta_high": Neumann(0)})
        assert vals[0] == pytest.approx(np.pi ** 2, rel=1e-6)


def test_remaining_operator_dunders():
    grid = Grid(x=chebyshev(15, 0, 1))
    x = grid.coords
    f = x ** 3
    D = Diff(grid, 1)
    Dop = D.to_operator()
    npt.assert_allclose((1 + D)(f), f + 3 * x ** 2, atol=1e-9)
    npt.assert_allclose((1 - D)(f), f - 3 * x ** 2, atol=1e-9)
    npt.assert_allclose((D - 1)(f), 3 * x ** 2 - f, atol=1e-9)
    npt.assert_allclose((D @ D)(f), 6 * x, atol=1e-9)
    npt.assert_allclose((Dop @ D)(f), 6 * x, atol=1e-9)
    npt.assert_allclose((D @ Dop)(f), 6 * x, atol=1e-9)
    npt.assert_allclose((D / 3)(f), x ** 2, atol=1e-9)
    npt.assert_allclose((+Dop)(f), 3 * x ** 2, atol=1e-9)
    npt.assert_allclose((np.float64(2) * Dop)(f), 6 * x ** 2, atol=1e-9)
    npt.assert_allclose((Dop + np.array(2.0))(f), 3 * x ** 2 + 2 * f, atol=1e-9)
    assert "Diff(order=1" in repr(D)
    assert Dop.shape == (15, 15)
    with pytest.raises(ValueError):
        Dop ** -1
    with pytest.raises(TypeError):
        Dop @ x
    with pytest.raises(TypeError):
        Dop / x
    with pytest.raises(TypeError):
        Diff(grid, 1.5)

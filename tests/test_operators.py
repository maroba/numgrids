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
    SphericalSurfaceGrid,
    chebyshev,
    colatitude,
    equidistant,
    periodic,
    radial,
)
from numgrids.boundary import resolve_bcs
from numgrids.operators import _real_eigenbasis


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


class TestGeneralizedEig:
    """L u = lambda M u."""

    @pytest.fixture
    def string(self):
        # -u'' = lambda u / x^2 on [1, e] with u = 0 at the ends:
        # u = sqrt(x) sin(n pi ln x), lambda = 1/4 + n^2 pi^2
        grid = Grid(x=chebyshev(40, 1, np.e))
        return grid, 0.25 + np.pi ** 2 * np.arange(1, 5) ** 2

    @pytest.mark.parametrize("kind", ["array", "callable", "operator", "sparse", "dense"])
    def test_density_forms(self, string, kind):
        grid, exact = string
        rho = 1 / grid.coords ** 2
        M = {
            "array": rho,
            "callable": lambda x: 1 / x ** 2,
            "operator": Operator.multiplication(grid, rho),
            "sparse": scipy.sparse.diags(rho),
            "dense": np.diag(rho),
        }[kind]
        vals, vecs = (-grid.diff_operator(2)).eig(k=4, bcs=0, M=M)
        npt.assert_allclose(vals, exact, rtol=1e-10)
        assert np.isrealobj(vecs)
        x = grid.coords
        u = np.sqrt(x) * np.sin(np.pi * np.log(x))
        npt.assert_allclose(vecs[0], u / np.linalg.norm(u), atol=1e-8)

    def test_sparse_path(self, string):
        grid, exact = string
        vals, _ = (-grid.diff_operator(2)).eig(k=4, bcs=0, M=1 / grid.coords ** 2,
                                               dense_threshold=10)
        npt.assert_allclose(vals, exact, rtol=1e-10)
        vals, _ = (-grid.diff_operator(2)).eig(k=1, bcs=0, M=1 / grid.coords ** 2,
                                               sigma=40, dense_threshold=10)
        npt.assert_allclose(vals, exact[1], rtol=1e-10)

    def test_scalar_mass_with_neumann(self):
        grid = Grid(chebyshev(30, 0, np.pi))
        L = -grid.diff_operator(2)
        vals, _ = L.eig(k=3, bcs=Neumann(0), M=4.0)
        npt.assert_allclose(vals, [0, 0.25, 1], atol=1e-10)
        vals, _ = L.eig(k=2, bcs=Neumann(0), M=4.0, sigma=0.3, dense_threshold=5)
        npt.assert_allclose(sorted(vals), [0, 0.25], atol=1e-10)   # closest to 0.3

    def test_membrane_with_variable_density_2d(self):
        # rho(x, y) = 1 / x^2 in x only: separable, lambda = 1/4 + m^2 pi^2 + n^2
        grid = Grid(x=chebyshev(24, 1, np.e), y=chebyshev(16, 0, np.pi))
        X, Y = grid.meshed_coords
        L = -grid.diff_operator(2, "x") - (1 / X ** 2) * grid.diff_operator(2, "y")
        vals, _ = L.eig(k=2, bcs=0, M=1 / X ** 2)
        npt.assert_allclose(vals, [0.25 + np.pi ** 2 + 1, 0.25 + np.pi ** 2 + 4], rtol=1e-8)

    def test_drum_with_hemisphere_density(self):
        # Stereographic projection maps the unit disk with density 4 / (1 + r^2)^2 onto a
        # hemisphere; the clamped rim is the equator: eigenvalues l (l + 1) with l + m odd.
        grid = PolarGrid(radial(24, 1.0), periodic(32))
        R, _ = grid.meshed_coords
        vals, vecs = (-grid.laplacian_operator()).eig(k=10, bcs=0, M=4 / (1 + R ** 2) ** 2)
        npt.assert_allclose(vals, [2, 6, 6, 12, 12, 12, 20, 20, 20, 20], rtol=1e-9)
        assert np.isrealobj(vecs)

    def test_invalid_mass(self, string):
        grid, _ = string
        L = grid.diff_operator(2)
        with pytest.raises(ValueError):
            L.eig(M=scipy.sparse.identity(3))
        with pytest.raises(TypeError):
            L.eig(M="heavy")
        with pytest.raises(NotImplementedError, match="generalized"):
            L.eig(k=1, bcs=0, M=2.0, which="LM", dense_threshold=5)


def _assert_eigenpairs(L, vals, vecs, bcs, atol=1e-7):
    """Check L v = lambda v away from the boundary points."""
    interior = np.ones(L.grid.size, dtype=bool)
    for bc in resolve_bcs(L.grid, bcs):
        interior[bc.face.flat_indices] = False
    for lam, v in zip(vals, vecs):
        residual = L.matrix @ v.ravel() - lam * v.ravel()
        assert np.max(np.abs(residual[interior])) <= atol * max(1.0, abs(lam))


class TestRealEigenvectors:
    """Degenerate eigenvalues of real operators come with real eigenvectors."""

    @pytest.mark.parametrize("dense_threshold", [2000, 10])
    def test_square(self, dense_threshold):
        grid = Grid(chebyshev(20, 0, 1), chebyshev(20, 0, 1))
        L = -grid.laplacian_operator()
        vals, vecs = L.eig(k=4, bcs=0, dense_threshold=dense_threshold)
        assert np.isrealobj(vals) and np.isrealobj(vecs)
        npt.assert_allclose(vals, np.pi ** 2 * np.array([2, 5, 5, 8]), rtol=1e-6)
        _assert_eigenpairs(L, vals, vecs, 0)
        # The degenerate pair is an orthonormal basis of its eigenspace.
        pair = vecs[1:3].reshape(2, -1)
        npt.assert_allclose(pair @ pair.T, np.eye(2), atol=1e-10)

    @pytest.mark.parametrize("dense_threshold", [2000, 10])
    def test_drum(self, dense_threshold):
        L = -PolarGrid(radial(20, 1.0), periodic(32)).laplacian_operator()
        vals, vecs = L.eig(k=6, bcs=0, dense_threshold=dense_threshold)
        assert np.isrealobj(vals) and np.isrealobj(vecs)
        npt.assert_allclose(np.sqrt(vals), [2.404825557695773, 3.831705970207512, 3.831705970207512,
                                            5.135622301840683, 5.135622301840683, 5.520078110286311],
                            rtol=1e-8)
        _assert_eigenpairs(L, vals, vecs, 0)

    def test_sphere_surface(self):
        L = -SphericalSurfaceGrid(colatitude(16), periodic(32)).laplacian_operator()
        vals, vecs = L.eig(k=9, dense_threshold=10)
        assert np.isrealobj(vals) and np.isrealobj(vecs)
        npt.assert_allclose(vals, [0, 2, 2, 2, 6, 6, 6, 6, 6], atol=1e-8)
        _assert_eigenpairs(L, vals, vecs, None)
        block = vecs[4:9].reshape(5, -1)
        npt.assert_allclose(block @ block.T, np.eye(5), atol=1e-8)

    def test_neumann_generalized_problem(self):
        grid = Grid(chebyshev(16, 0, np.pi), chebyshev(16, 0, np.pi))
        L = -grid.laplacian_operator()
        vals, vecs = L.eig(k=3, bcs=Neumann(0))
        assert np.isrealobj(vals) and np.isrealobj(vecs)
        npt.assert_allclose(vals, [0, 1, 1], atol=1e-8)

    def test_complex_eigenvalues_stay_complex(self):
        # d/dx on a periodic axis has eigenvalues i m with eigenvectors exp(i m x).
        grid = Grid(periodic(15, 0, 2 * np.pi))     # odd: no Nyquist mode
        D = grid.diff_operator(1)
        vals, vecs = D.eig(k=3)
        assert np.iscomplexobj(vals) and np.iscomplexobj(vecs)
        npt.assert_allclose(np.sort(vals.imag), [-1, 0, 1], atol=1e-10)
        _assert_eigenpairs(D, vals, vecs, None)

    def test_conjugate_pair_is_made_real(self):
        rng = np.random.default_rng(0)
        a, b = rng.standard_normal((2, 10))
        v = (a + 1j * b) / np.linalg.norm(a + 1j * b)
        vals = np.array([3.0 + 1e-12j, 3.0 - 1e-12j, 5.0])
        vecs = np.column_stack([v, v.conj(), rng.standard_normal(10)])
        new_vals, new_vecs = _real_eigenbasis(vals, vecs)
        npt.assert_array_equal(new_vals.imag, 0)
        assert not np.any(new_vecs.imag)
        # Same span as the real and imaginary parts of the input pair:
        basis = new_vecs[:, :2].real
        npt.assert_allclose(basis @ (basis.T @ np.column_stack([a, b])), np.column_stack([a, b]),
                            atol=1e-12)
        npt.assert_allclose(new_vecs[:, 2], vecs[:, 2])

    def test_genuinely_complex_pair_is_unchanged(self):
        vals = np.array([1 + 2j, 1 - 2j])
        vecs = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
        new_vals, new_vecs = _real_eigenbasis(vals, vecs)
        npt.assert_array_equal(new_vals, vals)
        npt.assert_array_equal(new_vecs, vecs)


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


def test_singular_rows_are_reported():
    grid = PolarGrid(chebyshev(10, 0.0, 1.0), periodic(8))
    L = grid.laplacian_operator()
    with pytest.raises(ValueError, match="singular"):
        L.solve(1.0, bcs={"r_high": 0})
    with pytest.warns(RuntimeWarning, match="spurious"):
        L.eig(k=2, bcs={"r_high": 0})
    # Covering the singular points with a condition removes the problem:
    u = L.solve(1.0, bcs={"r_high": 0, "r_low": 0})
    assert np.all(np.isfinite(u))


def test_no_spurious_nyquist_modes_on_periodic_axes():
    # The m = 0 modes of an annulus are non-degenerate; a Laplacian built from
    # squared first-derivative matrices would duplicate them (Nyquist mode).
    grid = PolarGrid(chebyshev(24, 0.4, 1.0), periodic(32))
    vals, _ = (-grid.laplacian_operator()).eig(k=3, bcs=0)
    npt.assert_allclose(np.sqrt(vals), [5.183072122637354, 5.391181199567447, 5.391181199567447],
                        rtol=1e-8)


class TestTransposeAndAdjoint:
    def test_transpose_and_conjugate_transpose(self):
        grid = Grid(x=chebyshev(8, 0, 1))
        L = grid.diff_operator() + 1j * grid.coords
        npt.assert_allclose(L.T.toarray(), L.toarray().T)
        npt.assert_allclose(L.H.toarray(), L.toarray().conj().T)
        assert isinstance(L.toarray(), np.ndarray)

    def test_adjoint_satisfies_its_definition(self):
        grid = PolarGrid(chebyshev(12, 1, 2), periodic(12))
        R, P = grid.meshed_coords
        L = grid.diff_operator(1, "r") + 1j * grid.diff_operator(1, "phi") + R
        f = np.exp(R) * np.cos(P) + 1j * np.sin(2 * P)
        g = R ** 2 * np.sin(P) + 1j * R
        assert grid.inner(f, L(g)) == pytest.approx(grid.inner(L.adjoint()(f), g), abs=1e-10)

    def test_derivative_is_anti_self_adjoint_on_periodic_grids(self):
        D = Grid(x=periodic(16)).diff_operator()
        npt.assert_allclose(D.adjoint().toarray(), -D.toarray(), atol=1e-12)

    def test_laplacian_is_self_adjoint_on_functions_vanishing_at_the_boundary(self):
        grid = Grid(x=chebyshev(20, 0, 1))
        x = grid.coords
        L = grid.laplacian_operator()
        f, g = np.sin(np.pi * x), x * (1 - x) * np.exp(x)
        assert grid.inner(f, L(g)) == pytest.approx(grid.inner(L(f), g), abs=1e-10)


class TestEigNormalization:
    def test_l2_normalization(self):
        grid = Grid(x=chebyshev(60, -8, 8))
        x = grid.coords
        H = -0.5 * grid.diff_operator(2) + 0.5 * x ** 2
        vals, psi = H.eig(k=3, bcs=0, normalize="l2")
        npt.assert_allclose(vals, [0.5, 1.5, 2.5], atol=1e-8)
        for i in range(3):
            assert grid.norm(psi[i]) == pytest.approx(1.0, abs=1e-12)
        # The ground state is the normalized Gaussian
        npt.assert_allclose(psi[0], np.pi ** -0.25 * np.exp(-x ** 2 / 2), atol=1e-6)

    def test_other_normalizations(self):
        grid = Grid(x=chebyshev(20, 0, 1))
        L = -grid.diff_operator(2)
        _, v = L.eig(k=2, bcs=0, normalize="max")
        npt.assert_allclose(np.abs(v).reshape(2, -1).max(axis=1), 1.0)
        _, v = L.eig(k=2, bcs=0)
        npt.assert_allclose(np.linalg.norm(v.reshape(2, -1), axis=1), 1.0)
        with pytest.raises(ValueError, match="normalization"):
            L.eig(k=2, bcs=0, normalize="l3")
        with pytest.raises(ValueError, match="method"):
            L.eig(k=2, bcs=0, method="magic")

    def test_l2_normalization_on_the_disk(self):
        disk = PolarGrid(radial(16, 1.0), periodic(24))
        _, modes = (-disk.laplacian_operator()).eig(k=4, bcs=0, normalize="l2")
        for mode in modes:
            assert disk.norm(mode) == pytest.approx(1.0, abs=1e-12)

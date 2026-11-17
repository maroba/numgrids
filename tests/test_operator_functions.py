"""Functions of operators: f(L), fractional powers, square roots, exponentials, inverses."""

import warnings

import numpy as np
import numpy.testing as npt
import pytest
import scipy.linalg
import scipy.signal
import scipy.special

from numgrids import (
    Grid,
    Neumann,
    Operator,
    PolarGrid,
    Robin,
    SphericalGrid,
    SphericalSurfaceGrid,
    TimeDependent,
    chebyshev,
    colatitude,
    equidistant,
    evolve,
    funm,
    periodic,
    radial,
)
from numgrids.funm import FourierOperator


@pytest.fixture(scope="module")
def disk():
    return PolarGrid(radial(24, 1.0), periodic(32))


@pytest.fixture(scope="module")
def ball():
    return SphericalGrid(radial(12, 1.0), colatitude(12), periodic(16))


def interior(grid):
    """Mask of the grid points that do not lie on a boundary face."""
    mask = np.ones(grid.shape, dtype=bool)
    for face in grid.faces.values():
        mask &= ~face.mask
    return mask


class TestPeriodic:
    """Operators with constant coefficients on periodic grids: f acts on the Fourier symbol."""

    @pytest.fixture
    def grid(self):
        return Grid(x=periodic(32))

    @pytest.mark.parametrize("s", [0.5, 0.3, 1.0, 1.7, -0.5, -1])
    def test_fractional_laplacian_1d(self, grid, s):
        x = grid.coords
        Ls = (-grid.laplacian_operator()).power(s)
        assert isinstance(Ls, FourierOperator)
        for k in (1, 3, 7):
            u = np.exp(1j * k * x)
            npt.assert_allclose(Ls(u), k ** (2 * s) * u, rtol=1e-9, atol=1e-9)
            npt.assert_allclose(Ls(np.cos(k * x)), k ** (2 * s) * np.cos(k * x), rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("s", [0.5, 0.25, 1.5, -1])
    def test_fractional_laplacian_2d(self, s):
        grid = Grid(periodic(16), periodic(12, 0, 1))
        X, Y = grid.meshed_coords
        Ls = (-grid.laplacian_operator()).power(s)
        assert isinstance(Ls, FourierOperator)
        u = np.exp(1j * (3 * X - 2 * np.pi * 2 * Y))
        npt.assert_allclose(Ls(u), (9 + (4 * np.pi) ** 2) ** s * u, rtol=1e-10, atol=1e-10)

    def test_square_root_is_hilbert_transform_of_derivative(self, grid):
        # (-d^2/dx^2)^(1/2) = H d/dx with the Hilbert transform H; both have the symbol |k|.
        x = grid.coords
        u = np.sin(2 * x) + 0.5 * np.cos(5 * x + 1) + np.exp(np.sin(x))
        root = (-grid.laplacian_operator()).sqrt()
        du = grid.diff_operator(1)(u)
        npt.assert_allclose(root(u), np.imag(scipy.signal.hilbert(du)), atol=1e-10)
        k = np.fft.fftfreq(len(x), 1 / len(x))
        npt.assert_allclose(root(u), np.fft.ifft(np.abs(k) * np.fft.fft(u)).real, atol=1e-10)

    def test_composition_of_powers(self, grid):
        L = -grid.laplacian_operator()
        product = L.power(0.3) @ L.power(0.4)
        assert isinstance(product, FourierOperator)
        npt.assert_allclose(product.toarray(), L.power(0.7).toarray(), atol=1e-10)
        root = L.sqrt()
        npt.assert_allclose((root @ root).toarray(), L.toarray(), atol=1e-10)
        npt.assert_allclose((root * root).toarray(), L.toarray(), atol=1e-10)

    def test_zero_mode_of_negative_powers(self, grid):
        # The Laplacian annihilates constants: negative powers act as the pseudo-inverse.
        x = grid.coords
        G = (-grid.laplacian_operator()).inverse()
        u = G(np.sin(x) + np.cos(3 * x) + 2.0)
        npt.assert_allclose(u, np.sin(x) + np.cos(3 * x) / 9, atol=1e-12)
        npt.assert_allclose(G(np.ones(grid.shape)), 0, atol=1e-12)
        half = (-grid.laplacian_operator()).power(-0.5)
        npt.assert_allclose(half(np.cos(3 * x) + 1), np.cos(3 * x) / 3, atol=1e-12)

    def test_powers_of_the_derivative(self, grid):
        # Symbol ik: a real operator with a complex spectrum.
        x = grid.coords
        D = grid.diff_operator(1)
        half = D.power(0.5)
        u = np.sin(2 * x) + np.cos(5 * x)
        assert half(u).dtype == float
        npt.assert_allclose(half(half(u)), D(u), atol=1e-10)
        npt.assert_allclose(D.inverse()(np.cos(3 * x)), np.sin(3 * x) / 3, atol=1e-12)

    def test_general_function_gets_real_eigenvalues(self, grid):
        x = grid.coords
        seen = []

        def lowpass(lam):
            seen.append(lam.dtype.kind)
            return lam <= 16

        P = (-grid.laplacian_operator()).function(lowpass)
        assert set(seen) == {"f"}
        npt.assert_allclose(P(np.sin(2 * x) + np.sin(9 * x)), np.sin(2 * x), atol=1e-12)
        npt.assert_allclose((P @ P).toarray(), P.toarray(), atol=1e-12)     # a projection

    def test_finite_differences(self):
        # Any shift-invariant operator has a symbol, also a finite difference stencil.
        grid = Grid(x=periodic(40))
        h = grid.coords[1] - grid.coords[0]
        shift = np.roll(np.eye(40), 1, axis=1)
        L = Operator(grid, (2 * np.eye(40) - shift - shift.T) / h ** 2)
        root = L.sqrt()
        assert isinstance(root, FourierOperator)
        u = np.sin(3 * grid.coords)
        npt.assert_allclose(root(u), 2 / h * np.sin(3 * h / 2) * u, atol=1e-10)

    def test_large_grid_needs_no_matrix(self):
        grid = Grid(periodic(128), periodic(128))
        X, Y = grid.meshed_coords
        root = (-grid.laplacian_operator()).sqrt()
        u = np.sin(3 * X) * np.cos(4 * Y)
        npt.assert_allclose(root(u), 5 * u, atol=1e-9)
        npt.assert_allclose(root.exp(-0.1)(u), np.exp(-0.5) * u, atol=1e-10)
        assert root._matrix is None
        with pytest.raises(ValueError, match="dense"):
            root.toarray()


class TestInterval:
    """Chebyshev grid on an interval with Dirichlet, Neumann and Robin conditions."""

    @pytest.fixture
    def grid(self):
        return Grid(x=chebyshev(40, 0, 1))

    @pytest.mark.parametrize("s", [0.5, 0.3, 1.0, -0.5, -1])
    def test_dirichlet(self, grid, s):
        x = grid.coords
        Ls = (-grid.laplacian_operator()).power(s, bcs=0)
        for n in (1, 2, 5):
            u = np.sin(n * np.pi * x)
            npt.assert_allclose(Ls(u), (n * np.pi) ** (2 * s) * u, rtol=1e-7, atol=1e-7)

    @pytest.mark.parametrize("s", [0.5, 0.3, 1.0, -0.5])
    def test_neumann(self, grid, s):
        x = grid.coords
        Ls = (-grid.laplacian_operator()).power(s, bcs=Neumann(0))
        for n in (1, 2, 5):
            u = np.cos(n * np.pi * x)
            npt.assert_allclose(Ls(u), (n * np.pi) ** (2 * s) * u, rtol=1e-7, atol=1e-7)
        # Constants are the zero mode, also for negative powers (pseudo-inverse).
        npt.assert_allclose(Ls(np.ones(grid.shape)), 0, atol=1e-7)

    def test_mixed_dirichlet_neumann(self, grid):
        x = grid.coords
        bcs = {"x_low": 0, "x_high": Neumann(0)}
        Ls = (-grid.laplacian_operator()).power(0.7, bcs=bcs)
        u = np.sin(1.5 * np.pi * x)
        npt.assert_allclose(Ls(u), (1.5 * np.pi) ** 1.4 * u, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("s", [0.5, -0.3, 1.0])
    def test_robin_against_eigenpairs(self, grid, s):
        L = -grid.laplacian_operator()
        bcs = {"x_low": Robin(2, 1), "x_high": Robin(1, 1)}
        vals, vecs = L.eig(k=5, bcs=bcs)
        Ls = L.power(s, bcs=bcs)
        for lam, v in zip(vals, vecs):
            npt.assert_allclose(Ls(v), lam ** s * v, rtol=1e-6, atol=1e-6)

    def test_fractional_poisson_problem(self, grid):
        # (-d^2/dx^2)^s u = f with u(0) = u(1) = 0.
        x = grid.coords
        s = 0.6
        L = -grid.laplacian_operator()
        f = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
        exact = np.sin(np.pi * x) / np.pi ** (2 * s) + 0.5 * np.sin(3 * np.pi * x) / (3 * np.pi) ** (2 * s)
        npt.assert_allclose(L.power(-s, bcs=0)(f), exact, atol=1e-10)
        npt.assert_allclose(L.power(s, bcs=0).solve(f, bcs=0), exact, atol=1e-10)
        npt.assert_allclose(L.power(s, bcs=0).factorize(bcs=0)(f), exact, atol=1e-10)

    def test_fractional_poisson_problem_neumann(self, grid):
        x = grid.coords
        bcs = {"x_low": 0, "x_high": Neumann(0)}
        L = -grid.laplacian_operator()
        f = np.sin(0.5 * np.pi * x)
        exact = f / (0.5 * np.pi) ** 1.2
        npt.assert_allclose(L.power(-0.6, bcs=bcs)(f), exact, atol=1e-10)
        npt.assert_allclose(L.power(0.6, bcs=bcs).solve(f, bcs=bcs), exact, atol=1e-10)

    @pytest.mark.parametrize("bcs", [0, Neumann(0), {"x_low": 0, "x_high": Robin(1, 2)}],
                             ids=["dirichlet", "neumann", "robin"])
    def test_result_satisfies_conditions(self, grid, bcs):
        from numgrids.boundary import resolve_bcs

        x = grid.coords
        L = -grid.laplacian_operator()
        u = L.power(0.4, bcs=bcs)(np.exp(x) * np.cos(3 * x))
        for bc in resolve_bcs(grid, bcs):
            rows, _ = bc.apply_to_system(0 * L.matrix, np.zeros(grid.size))
            npt.assert_allclose((rows @ u)[bc.face.flat_indices], 0, atol=1e-7)

    @pytest.mark.parametrize("bcs", [0, Neumann(0), {"x_low": 0, "x_high": Robin(1, 2)}],
                             ids=["dirichlet", "neumann", "robin"])
    def test_identity_function_reproduces_operator(self, grid, bcs):
        x = grid.coords
        L = -grid.laplacian_operator() + Operator.multiplication(grid, 1 + x)
        same = L.function(lambda z: z, bcs=bcs)
        # A function that satisfies the conditions: project an arbitrary one.
        projection = L.function(np.ones_like, bcs=bcs)
        u = projection(np.exp(x) * np.cos(3 * x))
        npt.assert_allclose(projection(u), u, atol=1e-12)
        inside = interior(grid)
        npt.assert_allclose(same(u)[inside], L(u)[inside], rtol=1e-8, atol=1e-8)
        # Boundary values of the argument do not matter, those of the result obey the conditions.
        v = u.copy()
        v[~inside] += 1.0
        npt.assert_allclose(same(v), same(u), atol=1e-12)
        npt.assert_allclose(projection(same(u)), same(u), atol=1e-8)

    def test_dirichlet_rows_and_columns_vanish(self, grid):
        M = (-grid.laplacian_operator()).power(0.5, bcs=0).toarray()
        assert not M[[0, -1], :].any()
        assert not M[:, [0, -1]].any()

    def test_functional_calculus(self, grid):
        L = -grid.laplacian_operator()
        bcs = {"x_low": 0, "x_high": Robin(1, 2)}
        a, b = L.power(0.3, bcs=bcs), L.power(-0.8, bcs=bcs)
        scale = np.abs(L.power(-0.5, bcs=bcs).toarray()).max()
        npt.assert_allclose((a @ b).toarray(), L.power(-0.5, bcs=bcs).toarray(), atol=1e-9 * scale)
        root = L.sqrt(bcs=bcs)
        inside = interior(grid)
        u = L.function(np.ones_like, bcs=bcs)(np.sin(3 * grid.coords))
        npt.assert_allclose((root @ root)(u)[inside], L(u)[inside], rtol=1e-8, atol=1e-8)

    def test_eigenvalues_of_the_result(self, grid):
        root = (-grid.laplacian_operator()).sqrt(bcs=0)
        vals, _ = root.eig(k=4, bcs=0)
        npt.assert_allclose(vals, np.pi * np.arange(1, 5), rtol=1e-9)

    def test_finite_differences(self):
        grid = Grid(x=equidistant(81, 0, 1))
        x = grid.coords
        root = (-grid.laplacian_operator()).sqrt(bcs=0)
        npt.assert_allclose(root(np.sin(np.pi * x)), np.pi * np.sin(np.pi * x), atol=1e-5)

    def test_two_dimensions(self):
        grid = Grid(x=chebyshev(16, 0, 1), y=chebyshev(14, 0, 1))
        X, Y = grid.meshed_coords
        L = -grid.laplacian_operator()
        bcs = {"all": 0, "y_high": Neumann(0)}
        u = np.sin(np.pi * X) * np.sin(1.5 * np.pi * Y)
        npt.assert_allclose(L.power(0.5, bcs=bcs)(u), np.sqrt(3.25) * np.pi * u, atol=1e-6)
        f = 1 + X * Y
        npt.assert_allclose(L.inverse(bcs=bcs)(f), L.solve(f, bcs=bcs), atol=1e-10)


class TestExponential:

    @pytest.fixture
    def grid(self):
        return Grid(x=chebyshev(30, 0, 1))

    @pytest.mark.parametrize("bcs", [0, Neumann(0), {"x_low": 0, "x_high": Robin(1, 2)}],
                             ids=["dirichlet", "neumann", "robin"])
    def test_heat_equation_against_evolve(self, grid, bcs):
        x = grid.coords
        L = grid.laplacian_operator()
        u0 = np.exp(-30 * (x - 0.4) ** 2)
        result = evolve(grid, L, u0, (0, 0.02), bcs=bcs, method="expm")
        npt.assert_allclose(L.exp(0.02, bcs=bcs)(u0), result.final, atol=1e-10)

    def test_heat_equation_exact(self, grid):
        x = grid.coords
        L = grid.laplacian_operator()
        npt.assert_allclose(L.exp(0.1, bcs=0)(np.sin(np.pi * x)),
                            np.exp(-0.1 * np.pi ** 2) * np.sin(np.pi * x), atol=1e-12)
        npt.assert_allclose(L.exp(0.1, bcs=Neumann(0))(1 + np.cos(np.pi * x)),
                            1 + np.exp(-0.1 * np.pi ** 2) * np.cos(np.pi * x), atol=1e-12)

    def test_semigroup(self, grid):
        L = grid.laplacian_operator()
        product = L.exp(0.01, bcs=0) @ L.exp(0.02, bcs=0)
        npt.assert_allclose(product.toarray(), L.exp(0.03, bcs=0).toarray(), atol=1e-12)

    def test_schroedinger_propagator(self):
        grid = Grid(x=chebyshev(60, -8, 8))
        x = grid.coords
        H = -0.5 * grid.diff_operator(2) + 0.5 * x ** 2
        t = 0.7
        U = H.exp(-1j * t, bcs=0)
        assert np.iscomplexobj(U.toarray())
        psi0 = np.exp(-(x - 1) ** 2 / 2) / np.pi ** 0.25
        result = evolve(grid, -1j * H, psi0.astype(complex), (0, t), bcs=0, method="expm")
        npt.assert_allclose(U(psi0), result.final, atol=1e-10)
        # A coherent state keeps its shape and oscillates: |psi|^2 is centered at cos(t).
        npt.assert_allclose(np.abs(U(psi0)) ** 2, np.exp(-(x - np.cos(t)) ** 2) / np.sqrt(np.pi), atol=1e-8)
        assert grid.norm(U(psi0)) == pytest.approx(1.0, abs=1e-9)

    def test_fractional_diffusion_with_evolve(self, grid):
        # u_t = -(-d^2/dx^2)^(1/2) u: the mode sin(n pi x) decays with the rate n pi.
        x = grid.coords
        Ls = (-grid.laplacian_operator()).power(0.5, bcs=0)
        u0 = np.sin(np.pi * x) + np.sin(2 * np.pi * x)
        exact = np.exp(-0.3 * np.pi) * np.sin(np.pi * x) + np.exp(-0.6 * np.pi) * np.sin(2 * np.pi * x)
        result = evolve(grid, -Ls, u0, (0, 0.3), bcs=0, method="expm")
        npt.assert_allclose(result.final, exact, atol=1e-10)
        result = evolve(grid, -Ls, u0, (0, 0.3), bcs=0, method="BDF", rtol=1e-8, atol=1e-10)
        npt.assert_allclose(result.final, exact, atol=1e-6)
        npt.assert_allclose(Ls.exp(-0.3, bcs=0)(u0), exact, atol=1e-10)

    def test_fractional_diffusion_periodic(self):
        grid = Grid(periodic(16), periodic(16))
        X, Y = grid.meshed_coords
        Ls = (-grid.laplacian_operator()).power(0.75)
        u0 = np.sin(X) * np.cos(2 * Y) + 1
        exact = np.exp(-0.5 * 5 ** 0.75) * (u0 - 1) + 1
        result = evolve(grid, -Ls, u0, (0, 0.5), method="expm")
        npt.assert_allclose(result.final, exact, atol=1e-10)
        npt.assert_allclose((-Ls).exp(0.5)(u0), exact, atol=1e-12)

    def test_overflow_is_reported(self, grid):
        with pytest.raises(ValueError, match="not finite on the spectrum"):
            (-grid.laplacian_operator()).exp(1.0, bcs=0)


class TestInverse:

    def test_against_solve(self):
        grid = Grid(x=chebyshev(30, 0, 1))
        x = grid.coords
        L = grid.laplacian_operator() - Operator.multiplication(grid, 1 + x ** 2)
        f = np.exp(x)
        for bcs in (0, Neumann(0), {"x_low": Neumann(0), "x_high": Robin(1, 1)}):
            npt.assert_allclose(L.inverse(bcs=bcs)(f), L.solve(f, bcs=bcs), atol=1e-11)

    def test_greens_function(self):
        # The Green's function of u'' = f, u(0) = u(1) = 0 is G(x, y) = x (y - 1) for x < y.
        grid = Grid(x=chebyshev(30, 0, 1))
        x = grid.coords
        G = grid.laplacian_operator().inverse(bcs=0)
        npt.assert_allclose(G(np.ones(grid.shape)), x * (x - 1) / 2, atol=1e-12)
        npt.assert_allclose(G(x), x * (x ** 2 - 1) / 6, atol=1e-12)

    def test_inverse_is_inverse(self):
        grid = Grid(x=chebyshev(20, 0, 1))
        L = grid.laplacian_operator()
        product = (L.inverse(bcs=0) @ L.function(lambda z: z, bcs=0)).toarray()
        npt.assert_allclose(product, L.function(np.ones_like, bcs=0).toarray(), atol=1e-9)


class TestDiskAndBall:
    """Problems that split into Fourier modes of the angle."""

    @pytest.mark.parametrize("s", [0.5, 0.25, -0.5])
    def test_disk_bessel_modes(self, disk, s):
        R, Phi = disk.meshed_coords
        Ls = (-disk.laplacian_operator()).power(s, bcs=0)
        assert isinstance(Ls, FourierOperator)
        for m, k in ((0, 0), (0, 1), (1, 0), (3, 1)):
            j = scipy.special.jn_zeros(m, k + 1)[k]
            u = scipy.special.jv(m, j * R) * np.cos(m * Phi)
            npt.assert_allclose(Ls(u), j ** (2 * s) * u, atol=1e-8)

    def test_disk_against_eigenpairs(self, disk):
        L = -disk.laplacian_operator()
        vals, vecs = L.eig(k=8, bcs=0)
        Ls = L.power(0.3, bcs=0)
        for lam, v in zip(vals, vecs):
            npt.assert_allclose(Ls(v), lam ** 0.3 * v, atol=1e-10)

    def test_disk_robin_against_eigenpairs(self, disk):
        L = -disk.laplacian_operator()
        bcs = Robin(1, 1)
        vals, vecs = L.eig(k=6, bcs=bcs)
        Ls = L.power(0.5, bcs=bcs)
        for lam, v in zip(vals, vecs):
            npt.assert_allclose(Ls(v), lam ** 0.5 * v, atol=1e-9)

    def test_disk_inverse_against_solve(self, disk):
        R, Phi = disk.meshed_coords
        L = -disk.laplacian_operator()
        f = np.exp(R * np.cos(Phi)) * (1 + R * np.sin(2 * Phi))
        npt.assert_allclose(L.inverse(bcs=0)(f), L.solve(f, bcs=0), atol=1e-11)
        npt.assert_allclose(L.inverse(bcs=Robin(1, 1))(f), L.solve(f, bcs=Robin(1, 1)), atol=1e-11)

    def test_disk_matrix(self, disk):
        # The matrix (built on demand) does what the Fourier form does.
        R, Phi = disk.meshed_coords
        Ls = (-disk.laplacian_operator()).power(0.5, bcs=0)
        M = Ls.toarray()
        assert M.dtype == float
        u = np.exp(R * np.cos(Phi)) * (1 - R ** 2)
        npt.assert_allclose(M @ u.ravel(), Ls(u).ravel(), atol=1e-10)
        rim = disk.faces["0_high"].flat_indices
        assert not M[rim].any() and not M[:, rim].any()

    def test_disk_heat_equation(self, disk):
        R, _ = disk.meshed_coords
        j = scipy.special.jn_zeros(0, 1)[0]
        u0 = scipy.special.jv(0, j * R)
        u = disk.laplacian_operator().exp(0.1, bcs=0)(u0)
        npt.assert_allclose(u, np.exp(-0.1 * j ** 2) * u0, atol=1e-10)

    @pytest.mark.parametrize("s", [0.5, -0.25])
    def test_ball_spherical_modes(self, ball, s):
        R, Theta, _ = ball.meshed_coords
        Ls = (-ball.laplacian_operator()).power(s, bcs=0)
        assert isinstance(Ls, FourierOperator)
        u = np.sinc(R)                       # j_0(pi r)
        npt.assert_allclose(Ls(u), np.pi ** (2 * s) * u, atol=1e-8)
        k = 4.493409457909064                # first zero of j_1
        u = scipy.special.spherical_jn(1, k * R) * np.cos(Theta)
        npt.assert_allclose(Ls(u), k ** (2 * s) * u, atol=1e-8)

    def test_ball_against_eigenpairs_and_solve(self, ball):
        L = -ball.laplacian_operator()
        vals, vecs = L.eig(k=5, bcs=0)
        Ls = L.power(0.5, bcs=0)
        for lam, v in zip(vals, vecs):
            npt.assert_allclose(Ls(v), lam ** 0.5 * v, atol=1e-9)
        X, Y, Z = ball.cartesian_coords
        f = 1 + X + Y * Z
        npt.assert_allclose(L.inverse(bcs=0)(f), L.solve(f, bcs=0), atol=1e-10)

    def test_sphere_surface(self):
        # Laplace-Beltrami operator: eigenvalues l (l + 1), with the zero mode l = 0.
        sphere = SphericalSurfaceGrid(colatitude(16), periodic(32))
        Theta, Phi = sphere.meshed_coords
        L = -sphere.laplacian_operator()
        Y21 = np.sin(Theta) * np.cos(Theta) * np.cos(Phi)
        npt.assert_allclose(L.sqrt()(Y21 + 3), np.sqrt(6) * Y21, atol=1e-9)
        npt.assert_allclose(L.inverse()(Y21 + 3), Y21 / 6, atol=1e-9)
        npt.assert_allclose((-L).exp(0.1)(Y21 + 3), np.exp(-0.6) * Y21 + 3, atol=1e-10)

    def test_periodic_and_bounded_axis(self):
        grid = Grid(x=periodic(16), y=chebyshev(20, 0, 1))
        X, Y = grid.meshed_coords
        L = -grid.laplacian_operator()
        Ls = L.power(0.4, bcs=0)
        assert isinstance(Ls, FourierOperator)
        u = np.cos(2 * X) * np.sin(np.pi * Y)
        npt.assert_allclose(Ls(u), (4 + np.pi ** 2) ** 0.4 * u, atol=1e-10)
        # Against a dense reference: eigendecomposition of the operator on the interior points.
        inside = interior(grid).ravel()
        lam, V = np.linalg.eig(L.toarray()[np.ix_(inside, inside)])
        reference = np.zeros((grid.size, grid.size))
        reference[np.ix_(inside, inside)] = ((V * lam ** 0.4) @ np.linalg.inv(V)).real
        npt.assert_allclose(Ls.toarray(), reference, atol=1e-8)

    def test_coefficients_that_depend_on_the_angle(self, disk):
        # No shift invariance: the dense path gives an ordinary operator.
        R, Phi = disk.meshed_coords
        L = -disk.laplacian_operator() + Operator.multiplication(disk, 1 + R * np.cos(Phi))
        Ls = L.power(0.5, bcs=0)
        assert type(Ls) is Operator
        vals, vecs = L.eig(k=3, bcs=0)
        for lam, v in zip(vals, vecs):
            npt.assert_allclose(Ls(v), lam ** 0.5 * v, atol=1e-9)


class TestFourierOperator:

    @pytest.fixture
    def grid(self):
        return Grid(x=periodic(8), y=chebyshev(16, 0, 1))

    @pytest.fixture
    def op(self, grid):
        return (-grid.laplacian_operator()).power(0.5, bcs=0)

    def test_repr_and_shape(self, grid, op):
        assert repr(op) == "FourierOperator(grid=Grid(8, 16), modes=8, block size=16)"
        assert op.shape == (128, 128)
        assert op._matrix is None

    def test_algebra_keeps_fourier_form(self, grid, op):
        X, Y = grid.meshed_coords
        u = np.cos(X) * np.sin(np.pi * Y) + Y * (1 - Y)
        other = (-grid.laplacian_operator()).power(-0.5, bcs=0)
        cases = [
            (-op, -op(u)),
            (2 * op, 2 * op(u)),
            (op * 2, 2 * op(u)),
            (op / 2, op(u) / 2),
            (op + other, op(u) + other(u)),
            (op - other, op(u) - other(u)),
            (op + 3, op(u) + 3 * u),
            (3 + op, op(u) + 3 * u),
            (op - 3, op(u) - 3 * u),
            (3 - op, 3 * u - op(u)),
            (op @ other, op(other(u))),
            (op * other, op(other(u))),
            (op ** 2, op(op(u))),
            (op ** 0, u),
        ]
        for result, expected in cases:
            assert isinstance(result, FourierOperator)
            assert result(u).dtype == float
            npt.assert_allclose(result(u), expected, atol=1e-10)
        assert op._matrix is None
        assert np.iscomplexobj((1j * op)(u))
        with pytest.raises(ValueError):
            op ** -1

    def test_mixed_algebra_uses_matrix(self, grid, op):
        X, Y = grid.meshed_coords
        u = np.cos(X) * np.sin(np.pi * Y)
        V = 1 + Y * np.cos(X)
        L = grid.laplacian_operator()
        for result, expected in [
            (op + V, op(u) + V * u),
            (V + op, op(u) + V * u),
            (V * op, V * op(u)),
            (op * V, op(V * u)),
            (op + L, op(u) + L(u)),
            (L - op, L(u) - op(u)),
            (op @ L, op(L(u))),
            (L @ op, L(op(u))),
        ]:
            assert type(result) is Operator
            npt.assert_allclose(result(u), expected, atol=1e-7)
        npt.assert_allclose(op.T.toarray(), op.toarray().T)
        npt.assert_allclose(op.as_matrix().toarray(), op.toarray())

    def test_functions_of_fourier_operators(self, grid, op):
        X, Y = grid.meshed_coords
        u = np.cos(2 * X) * np.sin(np.pi * Y)
        lam = 4 + np.pi ** 2
        for result in (op.power(2, bcs=0), op.power(2)):
            assert isinstance(result, FourierOperator)
            npt.assert_allclose(result(u), lam * u, atol=1e-9)
        npt.assert_allclose(op.exp(-0.1, bcs=0)(u), np.exp(-0.1 * np.sqrt(lam)) * u, atol=1e-10)

    def test_solve_and_eig(self, grid, op):
        X, Y = grid.meshed_coords
        u = np.cos(2 * X) * np.sin(np.pi * Y)
        npt.assert_allclose(op.solve(u, bcs=0), u / np.sqrt(4 + np.pi ** 2), atol=1e-10)
        vals, _ = op.eig(k=3, bcs=0)
        npt.assert_allclose(vals, np.sqrt(np.pi ** 2 + np.array([0, 1, 1])), rtol=1e-9)

    def test_complex_fields_and_wrong_shapes(self, grid, op):
        X, Y = grid.meshed_coords
        u = np.exp(2j * X) * np.sin(np.pi * Y)
        npt.assert_allclose(op(u), np.sqrt(4 + np.pi ** 2) * u, atol=1e-10)
        with pytest.raises(ValueError, match="does not match grid shape"):
            op(np.zeros(5))
        with pytest.raises(ValueError, match="do not match the Fourier decomposition"):
            FourierOperator(grid, op._blocks, np.zeros((3, 3, 3)))

    def test_matrix_two_periodic_axes(self):
        # Shifts along two axes at once.
        grid = Grid(periodic(6), periodic(4), chebyshev(5, 0, 1))
        X, Y, Z = grid.meshed_coords
        op = (-grid.laplacian_operator()).power(0.5, bcs=0)
        assert op._blocks.mode_shape == (6, 4)
        u = np.sin(X + Z) * np.cos(Y) * Z * (1 - Z)
        npt.assert_allclose(op.toarray() @ u.ravel(), op(u).ravel(), atol=1e-10)

    def test_matrix_size_limit(self, grid, op, monkeypatch):
        monkeypatch.setattr(funm, "_MAX_MATRIX_BYTES", 1000.0)
        with pytest.raises(ValueError, match="dense"):
            op.toarray()


class TestRealAndComplex:

    def test_real_in_real_out(self):
        grid = Grid(x=chebyshev(20, 0, 1))
        L = -grid.laplacian_operator()
        for op in (L.power(0.5, bcs=0), L.sqrt(bcs=Neumann(0)), L.inverse(bcs=0), (-L).exp(0.1, bcs=0),
                   L.function(np.log1p, bcs=0)):
            assert op.toarray().dtype == float
            assert op(np.sin(grid.coords)).dtype == float

    def test_real_operator_with_complex_eigenvalues(self):
        # Convection-diffusion on a periodic grid: eigenvalues k^2 + ik.
        grid = Grid(x=periodic(24))
        x = grid.coords
        L = -grid.laplacian_operator() + grid.diff_operator(1)
        seen = []

        def root(z):
            seen.append(z.dtype.kind)
            return np.sqrt(z)

        op = L.function(root)
        assert set(seen) == {"c"}
        assert op(np.sin(x)).dtype == float
        npt.assert_allclose(op(op(np.sin(2 * x))), L(np.sin(2 * x)), atol=1e-10)
        assert op.toarray().dtype == float

    def test_complex_function_of_real_operator(self):
        grid = Grid(x=chebyshev(20, 0, 1))
        x = grid.coords
        L = grid.laplacian_operator()
        op = L.function(lambda z: np.emath.sqrt(z), bcs=0)       # principal branch: i sqrt(-L)
        assert np.iscomplexobj(op.toarray())
        npt.assert_allclose(op(np.sin(np.pi * x)), 1j * np.pi * np.sin(np.pi * x), atol=1e-10)

    def test_complex_operator(self):
        grid = Grid(x=periodic(16))
        x = grid.coords
        L = 1j * grid.diff_operator(1)               # symbol -k
        op = L.function(np.abs)
        npt.assert_allclose(op(np.exp(3j * x)), 3 * np.exp(3j * x), atol=1e-12)
        dense = (L + Operator.multiplication(grid, 0 * x)).power(2)
        npt.assert_allclose(dense(np.exp(3j * x)), 9 * np.exp(3j * x), atol=1e-10)

    def test_function_returning_scalars(self):
        grid = Grid(x=chebyshev(12, 0, 1))
        L = grid.laplacian_operator()
        op = L.function(lambda z: 2.0, bcs=0)
        u = np.sin(np.pi * grid.coords)
        npt.assert_allclose(op(u), 2 * u, atol=1e-12)


class TestNonNormal:
    """Convection-dominated operators: the eigenvectors are ill conditioned."""

    eps = 0.02

    @pytest.fixture
    def grid(self):
        return Grid(x=chebyshev(80, 0, 1))

    @pytest.fixture
    def L(self, grid):
        # -eps u'' + u' has the eigenfunctions exp(x / (2 eps)) sin(n pi x).
        return -self.eps * grid.laplacian_operator() + grid.diff_operator(1)

    def eigenpair(self, grid, n=1):
        x = grid.coords
        u = np.exp((x - 1) / (2 * self.eps)) * np.sin(n * np.pi * x)
        return self.eps * (n * np.pi) ** 2 + 1 / (4 * self.eps), u

    def test_eigenvectors_are_ill_conditioned(self, L):
        _, V = np.linalg.eig(L.toarray()[1:-1, 1:-1])
        assert np.linalg.cond(V) > 1e7

    def test_square_root(self, grid, L):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            root = L.sqrt(bcs=0)
        M = root.toarray()
        assert M.dtype == float
        A = L.toarray()[1:-1, 1:-1]
        # The eigendecomposition would only reach about 1e-7 here.
        residual = np.linalg.norm(M[1:-1, 1:-1] @ M[1:-1, 1:-1] - A) / np.linalg.norm(A)
        assert residual < 1e-11
        lam, u = self.eigenpair(grid)
        npt.assert_allclose(root(u), np.sqrt(lam) * u, atol=1e-10)

    def test_fractional_power_and_inverse(self, grid, L):
        lam, u = self.eigenpair(grid)
        npt.assert_allclose(L.power(0.3, bcs=0)(u), lam ** 0.3 * u, atol=1e-10)
        npt.assert_allclose(L.power(-1.5, bcs=0)(u), lam ** -1.5 * u, atol=1e-10)
        npt.assert_allclose(L.inverse(bcs=0)(u), u / lam, atol=1e-10)
        f = np.cos(3 * grid.coords)
        npt.assert_allclose(L.inverse(bcs=0)(f), L.solve(f, bcs=0), atol=1e-10)

    def test_exponential(self, grid, L):
        lam, u = self.eigenpair(grid)
        E = L.exp(-0.05, bcs=0)
        npt.assert_allclose(E(u), np.exp(-0.05 * lam) * u, atol=1e-10)
        reference = scipy.linalg.expm(-0.05 * L.toarray()[1:-1, 1:-1])
        npt.assert_allclose(E.toarray()[1:-1, 1:-1], reference, atol=1e-11)

    def test_general_function(self, grid, L):
        # No dedicated algorithm: Schur-Parlett, which is still far better than the eigenvectors.
        op = L.function(np.sqrt, bcs=0)
        assert op.toarray().dtype == float
        reference = L.sqrt(bcs=0).toarray()
        assert np.linalg.norm(op.toarray() - reference) < 1e-6 * np.linalg.norm(reference)
        lam, u = self.eigenpair(grid)
        npt.assert_allclose(op(u), np.sqrt(lam) * u, atol=1e-8)

    def test_warning_if_inaccurate(self, grid, L, monkeypatch):
        monkeypatch.setattr(funm, "_WARN_ERROR", 0.0)
        with pytest.warns(RuntimeWarning, match="far from normal"):
            L.function(np.sqrt, bcs=0)


class TestErrors:

    @pytest.fixture
    def grid(self):
        return Grid(x=chebyshev(20, 0, 1))

    def test_negative_eigenvalues_with_fractional_power(self, grid):
        L = grid.laplacian_operator()
        with pytest.raises(ValueError, match=r"negative eigenvalue.*\(-L\)\.power\(0\.5, bcs\)"):
            L.power(0.5, bcs=0)
        with pytest.raises(ValueError, match="negative eigenvalue"):
            L.sqrt(bcs=0)
        with pytest.raises(ValueError, match="negative eigenvalue"):
            Grid(x=periodic(16)).laplacian_operator().power(-0.5)
        # Integer powers are fine.
        x = grid.coords
        npt.assert_allclose(L.power(2, bcs=0)(np.sin(np.pi * x)), np.pi ** 4 * np.sin(np.pi * x), atol=1e-4)
        npt.assert_allclose(L.power(-1, bcs=0)(np.sin(np.pi * x)), -np.sin(np.pi * x) / np.pi ** 2, atol=1e-12)

    def test_function_not_finite(self, grid):
        L = grid.laplacian_operator()
        with pytest.raises(ValueError, match="not finite on the spectrum"):
            L.function(np.log, bcs=0)
        with pytest.raises(ValueError, match="not finite on the spectrum"):
            L.function(lambda z: 1 / z, bcs=Neumann(0))      # zero eigenvalue

    def test_bad_arguments(self, grid):
        L = -grid.laplacian_operator()
        with pytest.raises(TypeError, match="f must be a function"):
            L.function(2.0, bcs=0)
        with pytest.raises(ValueError, match="one value per eigenvalue"):
            L.function(lambda z: np.ones(3), bcs=0)
        with pytest.raises(TypeError, match="real number"):
            L.power(1j, bcs=0)
        with pytest.raises(TypeError, match="real number"):
            L.power("1/2", bcs=0)
        with pytest.raises(ValueError, match="finite"):
            L.power(np.inf, bcs=0)
        with pytest.raises(TypeError, match="t must be a number"):
            L.exp("now", bcs=0)
        with pytest.raises(ValueError, match="Time-dependent"):
            L.power(0.5, bcs={"x_low": TimeDependent(lambda t: t)})

    def test_too_large(self, grid):
        L = -grid.laplacian_operator()
        with pytest.raises(ValueError, match="max_size = 10"):
            L.power(0.5, bcs=0, max_size=10)
        assert L.power(0.5, bcs=0, max_size=18).shape == (20, 20)
        disk = PolarGrid(radial(12, 1.0), periodic(16))
        with pytest.raises(ValueError, match="16 dense matrices of size 11 x 11"):
            (-disk.laplacian_operator()).sqrt(bcs=0, max_size=10)

    def test_too_much_memory(self, monkeypatch):
        disk = PolarGrid(radial(12, 1.0), periodic(16))
        monkeypatch.setattr(funm, "_MAX_BLOCK_BYTES", 1000.0)
        with pytest.raises(ValueError, match="GB"):
            (-disk.laplacian_operator()).sqrt(bcs=0)

    def test_conditions_that_do_not_determine_boundary_values(self, grid):
        L = -grid.laplacian_operator()
        with pytest.raises(ValueError, match="do not determine the boundary values"):
            L.power(0.5, bcs=Robin(0, 0))

    def test_coordinate_singularity(self):
        # Grid points on r = 0 give empty rows of the Laplacian.
        grid = PolarGrid(chebyshev(8, 0.0, 1.0), periodic(8))
        with pytest.warns(RuntimeWarning, match="empty row"):
            (-grid.laplacian_operator()).exp(-0.1, bcs={"0_high": 0})

"""evolve with linear=: exponential time differencing and IMEX schemes for stiff semilinear problems."""

import math
from fractions import Fraction

import numpy as np
import numpy.testing as npt
import pytest
import scipy.linalg

import numgrids.etd
from numgrids import (
    Grid,
    Neumann,
    PolarGrid,
    Robin,
    TimeDependent,
    chebyshev,
    equidistant,
    evolve,
    periodic,
    radial,
)
from numgrids.etd import _phi_dense, _phi_double, _phi_scalar

TOL = dict(rtol=1e-10, atol=1e-12)


def observed_order(errors, dts):
    """Convergence order from a least-squares fit of log(error) against log(dt)."""
    return np.polyfit(np.log(dts), np.log(errors), 1)[0]


def series(z, k, terms=80):
    """phi_k(z) from its Taylor series, in exact rational arithmetic."""
    z = Fraction(z)
    return float(sum(z ** j / math.factorial(j + k) for j in range(terms)))


class TestPhiFunctions:
    def test_scalar_values(self):
        z = np.array([0.0, 1e-12, -1e-6, 3e-3, 0.4, -1.3, 1.999, 2.0, -2.001, 4.0, -5.0])
        phi = _phi_scalar(z, 3)
        assert phi.shape == (4, len(z)) and phi.dtype == float
        for k in range(4):
            npt.assert_allclose(phi[k], [series(x, k) for x in z], rtol=1e-12)
        npt.assert_array_equal(phi[:, 0], [1, 1, 0.5, 1 / 6])

    def test_scalar_large_arguments_follow_the_recurrence(self):
        z = np.array([-30.0, -1e4, 25.0])
        phi = _phi_scalar(z, 3)
        npt.assert_allclose(phi[1], np.expm1(z) / z, rtol=1e-13)
        npt.assert_allclose(phi[2], (np.expm1(z) / z - 1) / z, rtol=1e-13)
        npt.assert_allclose(phi[3], ((np.expm1(z) / z - 1) / z - 0.5) / z, rtol=1e-13)

    def test_scalar_complex_arguments(self):
        z = np.array([1e-7j, 0.3 - 0.4j, 2.5j, -3 + 10j])
        phi = _phi_scalar(z, 2)
        assert np.iscomplexobj(phi)
        npt.assert_allclose(phi[1], np.expm1(z) / z, rtol=1e-12)
        # phi_2 on the imaginary axis: (exp(iy) - 1 - iy) / (iy)^2
        y = 2.5
        npt.assert_allclose(phi[2, 2], (1 - np.cos(y)) / y ** 2 + 1j * (y - np.sin(y)) / y ** 2, rtol=1e-12)
        npt.assert_allclose(phi[2, 0], 0.5 + 1e-7j / 6, rtol=1e-12)

    def test_matrix_functions_of_a_stiff_symmetric_matrix(self):
        rng = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
        lam = -np.logspace(-8, 5, 12)                    # slow and very fast modes
        Z = (Q * lam) @ Q.T
        phi = _phi_dense(Z, 3)
        for k in range(4):
            npt.assert_allclose(phi[k], (Q * _phi_scalar(lam, 3)[k]) @ Q.T, atol=1e-10)

    def test_matrix_functions_agree_with_the_augmented_exponential(self):
        # exp of [[Z, 1, 0], [0, 0, 1], [0, 0, 0]] has phi_1(Z) and phi_2(Z) in its first row of blocks
        grid = Grid(x=chebyshev(12, 0, 1))
        Z = 0.05 * grid.diff_operator(1).toarray() + 0.01 * grid.diff_operator(2).toarray()   # not normal
        n = len(Z)
        W = np.zeros((3 * n, 3 * n))
        W[:n, :n] = Z
        W[:n, n:2 * n] = W[n:2 * n, 2 * n:] = np.eye(n)
        E = scipy.linalg.expm(W)
        phi = _phi_dense(Z, 2)
        scale = np.abs(E).max()
        npt.assert_allclose(phi[0], E[:n, :n], atol=1e-11 * scale)
        npt.assert_allclose(phi[1], E[:n, n:2 * n], atol=1e-11 * scale)
        npt.assert_allclose(phi[2], E[:n, 2 * n:], atol=1e-11 * scale)

    def test_singular_and_defective_matrices(self):
        Z = np.array([[0.0, 1.0], [0.0, 0.0]])           # nilpotent: phi_k = 1/k! + Z/(k+1)!
        for scale in (1.0, 1e5):
            phi = _phi_dense(scale * Z, 3)
            for k in range(4):
                npt.assert_allclose(phi[k], np.eye(2) / math.factorial(k) + scale * Z / math.factorial(k + 1),
                                    rtol=1e-13)
        npt.assert_array_equal(_phi_dense(np.zeros((3, 3)), 2)[2], 0.5 * np.eye(3))

    def test_stack_of_matrices_and_doubling(self):
        rng = np.random.default_rng(2)
        stack = rng.standard_normal((5, 3, 3)) * np.array([1e-9, 1e-2, 1, 50, 1e3])[:, None, None]
        stack = stack - 1.2 * np.abs(stack).sum(axis=-1)[:, :, None] * np.eye(3)       # decaying
        phi = _phi_dense(stack, 3)
        assert phi.shape == (4, 5, 3, 3)
        for m in range(5):
            npt.assert_allclose(phi[:, m], _phi_dense(stack[m], 3), rtol=1e-14, atol=1e-300)
            npt.assert_allclose(phi[0, m], scipy.linalg.expm(stack[m]), atol=1e-13)
        doubled = _phi_double(list(_phi_dense(stack[2], 3)))
        npt.assert_allclose(doubled, _phi_dense(2 * stack[2], 3), atol=1e-13)
        assert np.iscomplexobj(_phi_dense(1j * stack[1], 1))


class TestConvergence:
    """Manufactured solutions of u_t = u_xx + u^2 + f(t, x)."""

    dts = np.array([0.1, 0.05, 0.025])
    orders = {"ETD1": 1, "ETDRK2": 2, "ETDRK4": 4, "IMEX2": 2}

    def errors(self, grid, exact, exact_t, method, bcs=None, linear=None, dts=None):
        L = grid.diff_operator(2) if linear is None else linear

        def rhs(t, u):
            return u ** 2 + exact_t(t) - L(exact(t)) - exact(t) ** 2

        return np.array([
            np.abs(evolve(grid, rhs, exact(0), (0, 1), linear=L, method=method, dt=dt, bcs=bcs).final
                   - exact(1)).max() for dt in (self.dts if dts is None else dts)])

    def check(self, errors, method, dts=None):
        order = observed_order(errors, self.dts if dts is None else dts)
        expected = self.orders[method]
        assert expected - 0.5 < order < expected + 0.7, (method, order, errors)

    @pytest.mark.parametrize("method", ["ETD1", "ETDRK2", "ETDRK4", "IMEX2"])
    def test_periodic_grid(self, method):
        grid = Grid(x=periodic(32, 0, 2 * np.pi))
        x = grid.coords
        exact = lambda t: 0.3 * np.exp(np.sin(x - t)) * np.cos(2 * t)
        exact_t = lambda t: -0.3 * (np.cos(x - t) * np.cos(2 * t) + 2 * np.sin(2 * t)) * np.exp(np.sin(x - t))
        errors = self.errors(grid, exact, exact_t, method, linear=0.5 * grid.diff_operator(2))
        self.check(errors, method)
        assert errors[-1] < {"ETD1": 0.1, "ETDRK2": 3e-3, "ETDRK4": 5e-7, "IMEX2": 1e-3}[method]

    @pytest.mark.parametrize("method", ["ETDRK2", "ETDRK4", "IMEX2"])
    def test_chebyshev_grid_with_dirichlet_conditions(self, method):
        grid = Grid(x=chebyshev(16, 0, 1))
        x = grid.coords
        exact = lambda t: np.exp(-t) * np.sin(np.pi * x)
        errors = self.errors(grid, exact, lambda t: -exact(t), method, bcs=0)
        self.check(errors, method)

    @pytest.mark.parametrize("method", ["ETDRK2", "ETDRK4", "IMEX2"])
    def test_chebyshev_grid_with_neumann_conditions(self, method):
        grid = Grid(x=chebyshev(16, 0, 1))
        x = grid.coords
        exact = lambda t: np.exp(-t) * np.cos(np.pi * x)
        errors = self.errors(grid, exact, lambda t: -exact(t), method, bcs=Neumann(0))
        self.check(errors, method)

    def test_inhomogeneous_dirichlet_and_robin_conditions(self):
        # u = exp(-t) cos(x) + 1 + x: constant boundary values 2 at x = 0, Robin u + u_n at x = 1
        grid = Grid(x=chebyshev(16, 0, 1))
        x = grid.coords
        exact = lambda t: np.exp(-t) * np.cos(x)
        bcs = {"x_low": TimeDependent(lambda t: np.exp(-t)),
               "x_high": Robin(1, 1, TimeDependent(lambda t: np.exp(-t) * (np.cos(1) - np.sin(1))))}
        for method, bound in (("ETDRK4", 1e-6), ("IMEX2", 1e-3)):
            errors = self.errors(grid, exact, lambda t: -exact(t), method, bcs=bcs)
            assert errors[-1] < bound
            # Boundary values that change in time cost ETDRK4 one order
            order = observed_order(errors, self.dts)
            assert (2.5 < order < 4.5) if method == "ETDRK4" else (1.5 < order < 2.5)


SPEED = 4.0


@pytest.fixture(scope="module")
def kdv():
    """A soliton of the Korteweg-de Vries equation, integrated with ETDRK4."""
    grid = Grid(x=periodic(128, -12, 12))
    x = grid.coords
    soliton = lambda t: 0.5 * SPEED / np.cosh(0.5 * np.sqrt(SPEED) * (x - SPEED * t + 4)) ** 2
    D = grid.diff_operator()
    times = np.linspace(0, 2, 5)
    res = evolve(grid, lambda t, u: -3 * D(u ** 2), soliton(0), (0, 2), t_eval=times,
                 linear=-grid.diff_operator(3), method="ETDRK4", dt=0.005)
    return grid, soliton, res


class TestKortewegDeVries:
    """u_t + 6 u u_x + u_xxx = 0 with the soliton u = c/2 sech^2(sqrt(c) (x - c t) / 2)."""

    c = SPEED

    def test_soliton_keeps_its_shape(self, kdv):
        grid, soliton, res = kdv
        assert res.success and res.sol is None and res.u.dtype == float
        for t, u in zip(res.t, res.u):
            npt.assert_allclose(u, soliton(t), atol=2e-4)

    def test_soliton_moves_with_speed_c(self, kdv):
        grid, soliton, res = kdv
        x = grid.coords
        center = np.array([grid.integrate(x * u) / grid.integrate(u) for u in res.u])
        npt.assert_allclose(np.diff(center) / np.diff(res.t), self.c, rtol=1e-4)

    def test_conserved_quantities(self, kdv):
        grid, soliton, res = kdv
        D = grid.diff_operator()
        mass = np.array([grid.integrate(u) for u in res.u])
        momentum = np.array([grid.integrate(u ** 2) for u in res.u])
        energy = np.array([grid.integrate(u ** 3 - 0.5 * D(u) ** 2) for u in res.u])
        # The values of the soliton on the real line (the periodic box cuts off its tails)
        assert mass[0] == pytest.approx(2 * np.sqrt(self.c), rel=1e-6)
        assert momentum[0] == pytest.approx(2 / 3 * self.c ** 1.5, rel=1e-6)
        assert energy[0] == pytest.approx(self.c ** 2.5 / 5, rel=1e-6)
        npt.assert_allclose(mass, mass[0], rtol=1e-12)        # exactly conserved by the scheme
        npt.assert_allclose(momentum, momentum[0], rtol=5e-5)
        npt.assert_allclose(energy, energy[0], rtol=1e-4)

    @pytest.mark.parametrize("method, dt, tol", [("ETDRK2", 0.001, 2e-3), ("IMEX2", 0.001, 2e-3)])
    def test_second_order_methods(self, kdv, method, dt, tol):
        grid, soliton, _ = kdv
        D = grid.diff_operator()
        res = evolve(grid, lambda t, u: -3 * D(u ** 2), soliton(0), (0, 0.5),
                     linear=-grid.diff_operator(3), method=method, dt=dt)
        npt.assert_allclose(res.final, soliton(0.5), atol=tol)


class TestNonlinearSchroedinger:
    """i psi_t + psi_xx / 2 + |psi|^2 psi = 0 with the bright soliton sech(x) exp(i t / 2)."""

    @pytest.fixture
    def setup(self):
        grid = Grid(x=periodic(128, -16, 16))
        x = grid.coords
        soliton = lambda t: np.exp(0.5j * t) / np.cosh(x)
        return grid, soliton, 0.5j * grid.diff_operator(2), (lambda t, psi: 1j * np.abs(psi) ** 2 * psi)

    def test_bright_soliton(self, setup):
        grid, soliton, L, rhs = setup
        res = evolve(grid, rhs, soliton(0), (0, 5), t_eval=[0, 2.5, 5], linear=L, method="ETDRK4", dt=0.05)
        assert np.iscomplexobj(res.u)
        for t, psi in zip(res.t, res.u):
            npt.assert_allclose(psi, soliton(t), atol=2e-5)
            assert grid.norm(psi) == pytest.approx(np.sqrt(2), rel=1e-6)       # integral of sech^2 is 2

    def test_real_initial_state_becomes_complex(self, setup):
        grid, soliton, L, rhs = setup
        res = evolve(grid, rhs, 1 / np.cosh(grid.coords), (0, 1), linear=L, method="ETDRK4", dt=0.05)
        npt.assert_allclose(res.final, soliton(1), atol=1e-5)

    def test_backward_in_time(self, setup):
        grid, soliton, L, rhs = setup
        res = evolve(grid, rhs, soliton(1), (1, 0), linear=L, method="ETDRK4", dt=0.05)
        npt.assert_allclose(res.t, [1, 0])
        npt.assert_allclose(res.final, soliton(0), atol=1e-5)

    @pytest.mark.parametrize("method", ["ETDRK2", "IMEX2"])
    def test_second_order_methods(self, setup, method):
        grid, soliton, L, rhs = setup
        res = evolve(grid, rhs, soliton(0), (0, 1), linear=L, method=method, dt=0.005)
        npt.assert_allclose(res.final, soliton(1), atol=1e-4)

    def test_potential_makes_the_linear_part_a_dense_normal_matrix(self):
        # Gross-Pitaevskii equation in a trap: i psi_t = -psi_xx / 2 + x^2 psi / 2 + |psi|^2 psi
        grid = Grid(x=periodic(48, -8, 8))
        x = grid.coords
        H = -0.5 * grid.diff_operator(2) + 0.5 * x ** 2
        psi0 = np.exp(-(x - 1) ** 2 / 2).astype(complex)
        ref = evolve(grid, lambda t, psi: -1j * (H(psi) + np.abs(psi) ** 2 * psi), psi0, (0, 2),
                     method="DOP853", **TOL)
        res = evolve(grid, lambda t, psi: -1j * np.abs(psi) ** 2 * psi, psi0, (0, 2), linear=-1j * H,
                     method="ETDRK4", dt=0.02)
        npt.assert_allclose(res.final, ref.final, atol=1e-6)
        assert grid.norm(res.final) == pytest.approx(grid.norm(psi0), rel=1e-7)


class TestReactionDiffusion:
    def test_fisher_front(self):
        # u_t = u_xx + u (1 - u) has the front u = (1 + exp((x - c t) / sqrt(6)))^-2 with c = 5 / sqrt(6)
        grid = Grid(x=chebyshev(72, -20, 30))
        x = grid.coords
        c = 5 / np.sqrt(6)
        front = lambda x, t: (1 + np.exp((x - c * t) / np.sqrt(6))) ** -2
        bcs = {"x_low": TimeDependent(lambda t: front(-20, t)), "x_high": TimeDependent(lambda t: front(30, t))}
        times = np.linspace(0, 4, 5)
        res = evolve(grid, lambda t, u: u * (1 - u), front(x, 0), (0, 4), t_eval=times, bcs=bcs,
                     linear=grid.diff_operator(2), method="ETDRK4", dt=0.05)
        for t, u in zip(res.t, res.u):
            npt.assert_allclose(u, front(x, t), atol=1e-6)
        # The point where u = 1/4 moves with the speed c
        fine = np.linspace(-5, 15, 20001)
        position = [fine[np.argmin(np.abs(grid.interpolate(u, fine) - 0.25))] for u in res.u]
        npt.assert_allclose(np.diff(position) / np.diff(times), c, atol=0.01)

    def test_allen_cahn_kink_is_stationary_and_stable(self):
        # u_t = u_xx + u - u^3: tanh(x / sqrt(2)) is a steady state that attracts nearby fronts
        grid = Grid(x=chebyshev(64, -8, 8))
        x = grid.coords
        kink = np.tanh(x / np.sqrt(2))
        res = evolve(grid, lambda t, u: u - u ** 3, np.tanh(x), (0, 20), bcs={"x_low": kink[0], "x_high": kink[-1]},
                     linear=grid.diff_operator(2), method="ETDRK4", dt=0.25)
        npt.assert_allclose(res.final, kink, atol=1e-6)
        assert res.final.dtype == float

    @pytest.mark.parametrize("method, dt, tol", [("ETDRK4", 0.02, 1e-7), ("ETDRK2", 0.005, 1e-5),
                                                 ("IMEX2", 0.005, 1e-4)])
    def test_two_species_with_per_field_conditions(self, method, dt, tol):
        # Predator-prey system with diffusion; fixed prey density at the ends, no flux of predators
        grid = Grid(x=chebyshev(20, 0, 5))
        x = grid.coords
        L = grid.diff_operator(2)
        a0, b0 = 1 + 0.5 * np.sin(np.pi * x / 5), 1 + 0.3 * np.cos(np.pi * x / 5)
        bcs = ({"x_low": 1, "x_high": 1}, Neumann(0))
        ref = evolve(grid, lambda t, a, b: (L(a) + a - a * b, 0.5 * L(b) + a * b - b), (a0, b0), (0, 2),
                     bcs=bcs, method="BDF", **TOL)
        res = evolve(grid, lambda t, a, b: (a - a * b, a * b - b), (a0, b0), (0, 2), bcs=bcs,
                     linear=[[L, None], [None, 0.5 * L]], method=method, dt=dt)
        assert isinstance(res.u, tuple) and res.u[0].shape == (2, 20)
        npt.assert_allclose(res.final[0], ref.final[0], atol=tol)
        npt.assert_allclose(res.final[1], ref.final[1], atol=tol)
        # The same with the linear reaction terms in the linear part, which then couples nothing new
        res = evolve(grid, lambda t, a, b: (-a * b, a * b), (a0, b0), (0, 2), bcs=bcs,
                     linear=[[L + 1, None], [None, 0.5 * L - 1]], method=method, dt=dt)
        npt.assert_allclose(res.final[0], ref.final[0], atol=tol)
        npt.assert_allclose(res.final[1], ref.final[1], atol=tol)

    @pytest.mark.parametrize("variation", [0.5, 1e-4])
    def test_variable_coefficients_on_a_periodic_grid(self, variation):
        # u_t = u_xx - a(x) u - u^3: not diagonal in Fourier space, however small the variation of a
        grid = Grid(x=periodic(32, 0, 2 * np.pi))
        x = grid.coords
        L = grid.diff_operator(2) - (1 + variation * np.cos(x))
        u0 = 1 + 0.5 * np.sin(x)
        ref = evolve(grid, lambda t, u: L(u) - u ** 3, u0, (0, 1), method="DOP853", **TOL)
        res = evolve(grid, lambda t, u: -u ** 3, u0, (0, 1), linear=L, method="ETDRK4", dt=0.02)
        assert res.final.dtype == float
        npt.assert_allclose(res.final, ref.final, atol=1e-7)
        res = evolve(grid, lambda t, u: -u ** 3, u0, (0, 1), linear=L, method="IMEX2", dt=0.005)
        npt.assert_allclose(res.final, ref.final, atol=1e-4)

    def test_nonlinear_wave_equation_with_a_single_condition(self):
        # u_tt = u_xx - u^3 as u_t = v, v_t = u_xx - u^3; the ends of the string are fixed
        grid = Grid(x=chebyshev(24, 0, np.pi))
        x = grid.coords
        L = grid.diff_operator(2)
        u0, v0 = 2 * np.sin(x), np.zeros_like(x)
        ref = evolve(grid, lambda t, u, v: (v, L(u) - u ** 3), (u0, v0), (0, 3), bcs=0, method="DOP853", **TOL)
        zero = np.zeros_like(x)
        for method, dt, tol in (("ETDRK4", 0.02, 1e-6), ("IMEX2", 0.002, 2e-4)):
            res = evolve(grid, lambda t, u, v: (zero, -u ** 3), (u0, v0), (0, 3), bcs=0,
                         linear=[[None, 1], [L, None]], method=method, dt=dt)
            npt.assert_allclose(res.final[0], ref.final[0], atol=tol)
            npt.assert_allclose(res.final[1], ref.final[1], atol=tol)
            npt.assert_array_equal(res.final[1][[0, -1]], 0)


class TestTwoDimensions:
    def test_periodic_grid_uses_fourier_space(self, monkeypatch):
        monkeypatch.setattr(numgrids.etd, "_ETD_DENSE_LIMIT", 0)       # no dense matrices allowed
        grid = Grid(x=periodic(24, 0, 2 * np.pi), y=periodic(16, 0, 2 * np.pi))
        X, Y = grid.meshed_coords
        exact = lambda t: np.exp(-t) * np.sin(X) * np.cos(Y)           # u_t = -u, Laplacian: -2 u
        rhs = lambda t, u: u + u ** 2 - exact(t) ** 2
        errors = []
        for dt in (0.1, 0.05):
            res = evolve(grid, rhs, exact(0), (0, 1), linear=grid.laplacian_operator(), method="ETDRK4", dt=dt)
            errors.append(np.abs(res.final - exact(1)).max())
        assert res.final.dtype == float and res.final.shape == (24, 16)
        assert errors[1] < 2e-6 and 3.5 < np.log2(errors[0] / errors[1]) < 4.7

    @pytest.mark.parametrize("method, dt, tol", [("ETDRK4", 0.025, 1e-7), ("ETDRK2", 0.0125, 1e-4),
                                                 ("IMEX2", 0.0125, 1e-3)])
    def test_system_with_coupled_linear_part(self, monkeypatch, method, dt, tol):
        grid = Grid(x=periodic(16, 0, 2 * np.pi), y=periodic(16, 0, 2 * np.pi))
        X, Y = grid.meshed_coords
        Lap = grid.laplacian_operator()
        u0, v0 = np.sin(X) * np.cos(2 * Y), np.cos(X + Y)
        ref = evolve(grid, lambda t, u, v: (Lap(u) + u * v - u, 0.5 * Lap(v) + u - v ** 3), (u0, v0), (0, 0.5),
                     method="DOP853", **TOL)
        monkeypatch.setattr(numgrids.etd, "_ETD_DENSE_LIMIT", 0)
        # one operator per field ...
        res = evolve(grid, lambda t, u, v: (u * v - u, u - v ** 3), (u0, v0), (0, 0.5),
                     linear=[[Lap, None], [None, 0.5 * Lap]], method=method, dt=dt)
        npt.assert_allclose(res.final, ref.final, atol=tol)
        # ... or a linear part that couples the fields: a small matrix per Fourier mode
        res = evolve(grid, lambda t, u, v: (u * v, -v ** 3), (u0, v0), (0, 0.5),
                     linear=[[Lap - 1, None], [1, 0.5 * Lap]], method=method, dt=dt)
        npt.assert_allclose(res.final, ref.final, atol=tol)
        assert res.final[0].dtype == float

    def test_wave_operator_in_fourier_space(self, monkeypatch):
        # Klein-Gordon equation u_tt = u_xx - u - u^3; the mode k = 0 of [[0, 1], [d_xx, 0]] is defective
        monkeypatch.setattr(numgrids.etd, "_ETD_DENSE_LIMIT", 0)
        grid = Grid(x=periodic(32, 0, 2 * np.pi))
        x = grid.coords
        L = grid.diff_operator(2)
        u0, v0 = 1 + 0.5 * np.cos(x), 0.2 * np.sin(2 * x)
        ref = evolve(grid, lambda t, u, v: (v, L(u) - u - u ** 3), (u0, v0), (0, 2), method="DOP853", **TOL)
        res = evolve(grid, lambda t, u, v: (0 * u, -u - u ** 3), (u0, v0), (0, 2),
                     linear=[[None, 1], [L, None]], method="ETDRK4", dt=0.02)
        npt.assert_allclose(res.final, ref.final, atol=1e-7)

    def test_square_with_boundaries(self):
        # Allen-Cahn equation on a square with no-flux boundaries, against an implicit solver
        grid = Grid(x=chebyshev(12, -1, 1), y=chebyshev(12, -1, 1))
        X, Y = grid.meshed_coords
        L = 0.05 * grid.laplacian_operator()
        u0 = np.tanh(3 * X) * np.cos(np.pi * Y / 2)
        ref = evolve(grid, lambda t, u: L(u) + u - u ** 3, u0, (0, 2), bcs=Neumann(0), method="BDF", **TOL)
        res = evolve(grid, lambda t, u: u - u ** 3, u0, (0, 2), bcs=Neumann(0), linear=L, method="ETDRK4", dt=0.05)
        npt.assert_allclose(res.final, ref.final, atol=1e-6)


class TestFourierSymbols:
    """On periodic grids, the linear part can be given as a function of the wave numbers."""

    @pytest.mark.parametrize("n", [64, 63])
    def test_agreement_with_the_operator(self, n):
        grid = Grid(x=periodic(n, -12, 12))
        x = grid.coords
        u0 = 2 / np.cosh(x + 4) ** 2
        D = grid.diff_operator()
        rhs = lambda t, u: -3 * D(u ** 2)
        for method in ("ETDRK4", "IMEX2"):
            a = evolve(grid, rhs, u0, (0, 0.2), linear=-grid.diff_operator(3), method=method, dt=0.005)
            b = evolve(grid, rhs, u0, (0, 0.2), linear=lambda k: 1j * k ** 3, method=method, dt=0.005)
            assert b.final.dtype == float
            npt.assert_allclose(b.final, a.final, atol=1e-10)

    def test_symbol_of_a_complex_operator(self):
        grid = Grid(x=periodic(64, -16, 16))
        x = grid.coords
        rhs = lambda t, psi: 1j * np.abs(psi) ** 2 * psi
        a = evolve(grid, rhs, 1 / np.cosh(x), (0, 1), linear=0.5j * grid.diff_operator(2), method="ETDRK4", dt=0.05)
        b = evolve(grid, rhs, 1 / np.cosh(x), (0, 1), linear=lambda k: -0.5j * k ** 2, method="ETDRK4", dt=0.05)
        assert np.iscomplexobj(b.final)
        npt.assert_allclose(b.final, a.final, atol=1e-11)

    def test_symbols_and_operators_in_a_system(self):
        grid = Grid(x=periodic(16, 0, 2 * np.pi), y=periodic(12, 0, 4 * np.pi))
        X, Y = grid.meshed_coords
        Lap = grid.laplacian_operator()
        u0, v0 = np.sin(X) * np.cos(Y), np.cos(X + 0.5 * Y)
        rhs = lambda t, u, v: (u * v, -v ** 3)
        a = evolve(grid, rhs, (u0, v0), (0, 0.5), method="ETDRK4", dt=0.05,
                   linear=[[0.3 - (1 + Lap) ** 2, grid.diff_operator(1, 0)], [1, 0.5 * Lap]])
        b = evolve(grid, rhs, (u0, v0), (0, 0.5), method="ETDRK4", dt=0.05,
                   linear=[[lambda kx, ky: 0.3 - (1 - kx ** 2 - ky ** 2) ** 2, lambda kx, ky: 1j * kx],
                           [1, 0.5 * Lap]])
        assert b.final[0].dtype == float
        npt.assert_allclose(b.final, a.final, atol=1e-10)

    def test_swift_hohenberg_stripes(self):
        # u_t = r u - (1 + Laplacian)^2 u - u^3: the mode cos(x) grows with the rate r and
        # saturates near the amplitude sqrt(4 r / 3); the eigenvalues of the linear part reach -3700
        grid = Grid(x=periodic(32, 0, 8 * np.pi), y=periodic(32, 0, 8 * np.pi))
        X, Y = grid.meshed_coords
        r = 0.1
        res = evolve(grid, lambda t, u: -u ** 3, 1e-4 * np.cos(X), (0, 300), t_eval=[0, 10, 300],
                     linear=lambda kx, ky: r - (1 - kx ** 2 - ky ** 2) ** 2, method="ETDRK4", dt=0.5)
        assert res.success
        npt.assert_allclose(res.u[1], 1e-4 * np.exp(r * 10) * np.cos(X), rtol=1e-5, atol=1e-9)
        assert np.abs(res.final).max() == pytest.approx(np.sqrt(4 * r / 3), rel=0.02)
        npt.assert_allclose(res.final, res.final[:, :1] * np.ones(32), atol=1e-9)      # still stripes

    def test_growing_modes_do_not_amplify_round_off(self):
        # The half spectrum of a real field in 2D holds some coefficients twice. Their difference is
        # invisible on the grid; a growth rate of 0.5 would amplify it by exp(150) here.
        grid = Grid(x=periodic(20, 0, 4 * np.pi), y=periodic(20, 0, 4 * np.pi))
        u0 = 0.1 * np.random.default_rng(3).standard_normal(grid.shape)
        for method, dt in (("ETDRK4", 0.5), ("IMEX2", 0.25)):
            res = evolve(grid, lambda t, u: -u ** 3, u0, (0, 300), method=method, dt=dt,
                         linear=lambda kx, ky: 0.5 - (1 - kx ** 2 - ky ** 2) ** 2)
            assert res.success, res.message
            assert res.final.dtype == float and 0.5 < np.abs(res.final).max() < 1.5
        # ... also with operators, and for a system
        L = 0.5 - (1 + grid.laplacian_operator()) ** 2
        res = evolve(grid, lambda t, u, v: (-u ** 3, -v ** 3), (u0, u0), (0, 300), method="ETDRK4", dt=0.5,
                     linear=[[L, None], [0.1, L]])
        assert res.success, res.message
        assert 0.5 < np.abs(res.final[0]).max() < 1.5 and 0.5 < np.abs(res.final[1]).max() < 1.5

    def test_errors(self):
        symbol = lambda k: -k ** 2
        rod = Grid(x=chebyshev(8, 0, 1))
        with pytest.raises(ValueError, match="periodic along all axes"):
            evolve(rod, None, np.zeros(8), (0, 1), linear=symbol, method="ETDRK4", dt=0.1)
        grid = Grid(x=periodic(8))
        x = grid.coords
        u0 = np.sin(x)
        with pytest.raises(TypeError, match="single field"):
            evolve(grid, None, (u0, u0), (0, 1), linear=symbol, method="ETDRK4", dt=0.1)
        with pytest.raises(ValueError, match="2 x 2"):
            evolve(grid, None, (u0, u0), (0, 1), linear=[[symbol, None]], method="ETDRK4", dt=0.1)
        with pytest.raises(ValueError, match="meshed wave numbers"):
            evolve(grid, None, u0, (0, 1), linear=lambda k: k[1:], method="ETDRK4", dt=0.1)
        with pytest.raises(ValueError, match="not finite"):
            evolve(grid, None, u0, (0, 1), linear=lambda k: np.where(k == 0, np.nan, k), method="ETDRK4", dt=0.1)
        with pytest.raises(ValueError, match="constant coefficients"):
            evolve(grid, None, (u0, u0), (0, 1), linear=[[symbol, np.cos(x)], [None, symbol]],
                   method="ETDRK4", dt=0.1)


class TestStiffness:
    def test_large_steps_where_explicit_schemes_blow_up(self):
        # The eigenvalues of the Chebyshev second derivative reach -5e5 here: far beyond explicit schemes
        grid = Grid(x=chebyshev(40, 0, 1))
        x = grid.coords
        L = grid.diff_operator(2)
        exact = lambda t: np.exp(-t) * np.sin(np.pi * x)
        rhs = lambda t, u: u ** 2 + (np.pi ** 2 - 1) * exact(t) - exact(t) ** 2
        for method, tol in (("ETDRK4", 1e-5), ("ETDRK2", 5e-3), ("IMEX2", 1e-2)):
            res = evolve(grid, rhs, exact(0), (0, 1), bcs=0, linear=L, method=method, dt=0.1)
            assert res.success
            npt.assert_allclose(res.final, exact(1), atol=tol)
        # Without a linear part, ETDRK4 is the classical Runge-Kutta scheme, which is unstable at this step
        res = evolve(grid, lambda t, u: L(u) + rhs(t, u), exact(0), (0, 1), t_eval=[0, 0.5, 1], bcs=0,
                     linear=0 * L, method="ETDRK4", dt=0.1)
        assert not res.success and "smaller time step" in res.message
        npt.assert_allclose(res.t, [0])
        assert res.u.shape == (1, 40)

    def test_classical_runge_kutta_limit(self):
        # u' = -u^2, u(0) = 1: u = 1 / (1 + t); with L = 0 the scheme has the classical order 4
        grid = Grid(x=periodic(4))
        ones = np.ones(grid.shape)
        errors = [abs(evolve(grid, lambda t, u: -u ** 2, ones, (0, 1), linear=0 * grid.diff_operator(),
                             method="ETDRK4", dt=dt).final[0] - 0.5) for dt in (0.1, 0.05)]
        assert 3.7 < np.log2(errors[0] / errors[1]) < 4.3


class TestLinearProblems:
    """Without a nonlinear part, the ETD methods are exact, like method="expm"."""

    @pytest.mark.parametrize("method", ["ETD1", "ETDRK2", "ETDRK4"])
    def test_agreement_with_expm(self, method):
        grid = Grid(x=chebyshev(20, 0, 1))
        x = grid.coords
        L = grid.diff_operator(2)
        u0 = np.sin(3 * x) + x ** 2
        cases = [
            dict(bcs={"x_low": 1, "x_high": 3}),
            dict(bcs={"x_low": Neumann(0.5), "x_high": Robin(2.0, 1.0, 0.5)}, source=np.exp(x)),
        ]
        for case in cases:
            ref = evolve(grid, L, u0, (0, 0.3), t_eval=[0, 0.1, 0.3], method="expm", **case)
            res = evolve(grid, None, u0, (0, 0.3), t_eval=[0, 0.1, 0.3], linear=L, method=method, dt=0.05, **case)
            npt.assert_allclose(res.u, ref.u, atol=1e-9)
            res = evolve(grid, lambda t, u: 0.0, u0, (0, 0.3), linear=L, method=method, dt=0.1, **case)
            npt.assert_allclose(res.final, ref.final, atol=1e-9)

    def test_schroedinger_equation_and_wave_equation(self):
        grid = Grid(x=periodic(32, 0, 2 * np.pi))
        x = grid.coords
        H = -0.5 * grid.diff_operator(2)
        psi0 = np.exp(-4 * (x - np.pi) ** 2 + 2j * x)
        ref = evolve(grid, -1j * H, psi0, (0, 5), method="expm")
        res = evolve(grid, None, psi0, (0, 5), linear=-1j * H, method="ETDRK4", dt=1.0)
        npt.assert_allclose(res.final, ref.final, atol=1e-11)
        assert grid.norm(res.final) == pytest.approx(grid.norm(psi0), rel=1e-12)
        # Drum: the blocks couple the fields, and the grid has a boundary
        disk = PolarGrid(radial(8, 1.0), periodic(8))
        Lap = disk.laplacian_operator()
        X, Y = disk.cartesian_coords
        u0 = (1 - X ** 2 - Y ** 2) * (1 + X)
        ref = evolve(disk, [[None, 1], [Lap, None]], (u0, 0 * u0), (0, 2), bcs=0, method="expm")
        res = evolve(disk, None, (u0, 0 * u0), (0, 2), bcs=0, linear=[[None, 1], [Lap, None]],
                     method="ETDRK4", dt=0.5)
        npt.assert_allclose(res.final, ref.final, atol=1e-9)

    def test_time_dependent_source(self):
        # u = t sin(pi x):  u_t = u_xx + (1 + pi^2 t) sin(pi x)
        grid = Grid(x=chebyshev(16, 0, 1))
        x = grid.coords
        source = lambda t: (1 + np.pi ** 2 * t) * np.sin(np.pi * x)
        for method in ("ETDRK4", "ETDRK2", "IMEX2"):           # linear in time: integrated exactly
            res = evolve(grid, None, np.zeros(grid.shape), (0, 1), bcs=0, linear=grid.diff_operator(2),
                         method=method, dt=0.1, source=source)
            npt.assert_allclose(res.final, np.sin(np.pi * x), atol=1e-9)
        # ... also in Fourier space, here with a source that is not a polynomial in time
        grid = Grid(x=periodic(16, 0, 2 * np.pi))
        x = grid.coords
        res = evolve(grid, None, np.zeros(grid.shape), (0, 1), linear=grid.diff_operator(2), method="ETDRK4",
                     dt=0.05, source=lambda t: (np.cos(t) + np.sin(t)) * np.sin(x))
        npt.assert_allclose(res.final, np.sin(1) * np.sin(x), atol=1e-8)

    def test_imex_converges_to_expm(self):
        grid = Grid(x=chebyshev(20, 0, 1))
        x = grid.coords
        L = grid.diff_operator(2)
        u0 = np.sin(3 * x) + x ** 2
        bcs = {"x_low": 1, "x_high": Neumann(0.5)}
        ref = evolve(grid, L, u0, (0, 0.3), bcs=bcs, method="expm")
        dts = np.array([0.03, 0.015, 0.0075])
        errors = [np.abs(evolve(grid, None, u0, (0, 0.3), bcs=bcs, linear=L, method="IMEX2", dt=dt).final
                         - ref.final).max() for dt in dts]
        assert errors[-1] < 1e-4 and 1.6 < observed_order(errors, dts) < 2.4


class TestLargeProblems:
    def test_imex_on_a_finite_difference_grid(self, monkeypatch):
        grid = Grid(x=equidistant(30, 0, 1), y=equidistant(30, 0, 1))
        X, Y = grid.meshed_coords
        L = 0.02 * grid.laplacian_operator()
        u0 = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
        rhs = lambda t, u: u - u ** 3
        ref = evolve(grid, lambda t, u: L(u) + rhs(t, u), u0, (0, 1), bcs=0, method="BDF", **TOL)
        res = evolve(grid, rhs, u0, (0, 1), bcs=0, linear=L, method="IMEX2", dt=0.01)
        npt.assert_allclose(res.final, ref.final, atol=1e-4)
        monkeypatch.setattr(numgrids.etd, "_ETD_DENSE_LIMIT", 500)
        with pytest.raises(ValueError, match="IMEX2"):
            evolve(grid, rhs, u0, (0, 1), bcs=0, linear=L, method="ETDRK4", dt=0.01)

    def test_imex_decouples_fourier_modes_on_a_disk(self, monkeypatch):
        from numgrids.operators import Operator
        methods = []
        factorize = Operator.factorize

        def spy(self, *args, **kwargs):
            solver = factorize(self, *args, **kwargs)
            methods.append(solver.method)
            return solver

        monkeypatch.setattr(Operator, "factorize", spy)
        disk = PolarGrid(radial(12, 1.0), periodic(16))
        X, Y = disk.cartesian_coords
        L = 0.05 * disk.laplacian_operator()
        u0 = np.exp(-10 * ((X - 0.3) ** 2 + Y ** 2))
        rhs = lambda t, u: u * (1 - u)
        res = evolve(disk, rhs, u0, (0, 1), bcs=Neumann(0), linear=L, method="IMEX2", dt=0.01)
        assert methods == ["fourier"]
        ref = evolve(disk, lambda t, u: L(u) + rhs(t, u), u0, (0, 1), bcs=Neumann(0), method="BDF", **TOL)
        npt.assert_allclose(res.final, ref.final, atol=1e-4)
        etd = evolve(disk, rhs, u0, (0, 1), bcs=Neumann(0), linear=L, method="ETDRK4", dt=0.05)
        npt.assert_allclose(etd.final, ref.final, atol=1e-6)

    def test_imex_for_a_large_system(self, monkeypatch):
        monkeypatch.setattr(numgrids.etd, "_SOLVE_DENSE_LIMIT", 0)      # sparse LU
        grid = Grid(x=equidistant(60, 0, 5))
        x = grid.coords
        L = grid.diff_operator(2)
        a0, b0 = 1 + 0.5 * np.sin(np.pi * x / 5), 1 + 0.3 * np.cos(np.pi * x / 5)
        bcs = ({"x_low": 1, "x_high": 1}, Neumann(0))
        ref = evolve(grid, lambda t, a, b: (L(a) + a - a * b, 0.5 * L(b) + a * b - b), (a0, b0), (0, 1),
                     bcs=bcs, method="BDF", **TOL)
        res = evolve(grid, lambda t, a, b: (a - a * b, a * b - b), (a0, b0), (0, 1), bcs=bcs,
                     linear=[[L, None], [None, 0.5 * L]], method="IMEX2", dt=0.005)
        npt.assert_allclose(res.final, ref.final, atol=1e-4)


class TestOutputTimes:
    @pytest.fixture
    def problem(self):
        grid = Grid(x=periodic(16, 0, 2 * np.pi))
        x = grid.coords
        return grid, dict(rhs=lambda t, u: -u ** 3, u0=1 + 0.5 * np.sin(x), linear=grid.diff_operator(2))

    def test_default_is_start_and_end(self, problem):
        grid, kwargs = problem
        res = evolve(grid, t_span=(0.5, 1.5), method="ETDRK4", dt=0.1, **kwargs)
        npt.assert_array_equal(res.t, [0.5, 1.5])
        assert res.u.shape == (2, 16)
        npt.assert_array_equal(res.u[0], kwargs["u0"])                 # the initial state, to the last bit
        assert "10 steps of ETDRK4" in res.message

    def test_uneven_output_times_are_hit_exactly(self, problem):
        grid, kwargs = problem
        t_eval = [0.1, 0.25, 0.3, 1.0]
        res = evolve(grid, t_span=(0, 1), t_eval=t_eval, method="ETDRK4", dt=0.02, **kwargs)
        npt.assert_array_equal(res.t, t_eval)
        # 5 + 8 + 3 + 35 steps: the intervals 0.15 and 0.05 are no multiples of dt
        assert "51 steps" in res.message
        ref = evolve(grid, lambda t, u: kwargs["linear"](u) - u ** 3, kwargs["u0"], (0, 1), t_eval=t_eval,
                     method="DOP853", **TOL)
        npt.assert_allclose(res.u, ref.u, atol=1e-6)

    def test_step_is_shortened_to_divide_the_interval(self, problem):
        grid, kwargs = problem
        a = evolve(grid, t_span=(0, 1), method="ETDRK2", dt=0.3, **kwargs)       # 4 steps of 0.25
        b = evolve(grid, t_span=(0, 1), method="ETDRK2", dt=0.25, **kwargs)
        assert "4 steps" in a.message and "4 steps" in b.message
        npt.assert_array_equal(a.final, b.final)
        c = evolve(grid, t_span=(0, 1), method="ETDRK2", dt=5.0, **kwargs)        # a single step
        assert "1 steps" in c.message

    def test_many_step_sizes(self, problem):
        grid, kwargs = problem
        t_eval = np.cumsum(0.01 * 1.5 ** np.arange(8))                            # 8 different intervals
        ref = evolve(grid, t_span=(0, 1), t_eval=t_eval[-1:], method="ETDRK4", dt=0.01, **kwargs)
        for method, tol in (("ETDRK4", 1e-6), ("IMEX2", 1e-3)):
            res = evolve(grid, t_span=(0, 1), t_eval=t_eval, method=method, dt=0.02, **kwargs)
            npt.assert_allclose(res.final, ref.final, atol=tol)

    def test_invalid_output_times(self, problem):
        grid, kwargs = problem
        with pytest.raises(ValueError, match="t_span"):
            evolve(grid, t_span=(0, 1), t_eval=[0, 2], method="ETDRK4", dt=0.1, **kwargs)
        with pytest.raises(ValueError, match="sorted"):
            evolve(grid, t_span=(0, 1), t_eval=[0.5, 0.2], method="ETDRK4", dt=0.1, **kwargs)


class TestErrors:
    @pytest.fixture
    def rod(self):
        return Grid(x=chebyshev(12, 0, 1))

    def test_time_step(self, rod):
        L = rod.diff_operator(2)
        u0 = np.zeros(rod.shape)
        with pytest.raises(ValueError, match="fixed time step"):
            evolve(rod, lambda t, u: u, u0, (0, 1), bcs=0, linear=L, method="ETDRK4")
        for dt in (0, -0.1, np.inf):
            with pytest.raises(ValueError, match="positive"):
                evolve(rod, lambda t, u: u, u0, (0, 1), bcs=0, linear=L, method="IMEX2", dt=dt)
        for method in ("RK45", "BDF", "expm"):
            with pytest.raises(ValueError, match="fixed-step methods"):
                evolve(rod, L, u0, (0, 1), bcs=0, method=method, dt=0.1)

    def test_linear_part(self, rod):
        L = rod.diff_operator(2)
        u0 = np.zeros(rod.shape)
        for method in ("ETDRK4", "ETDRK2", "ETD1", "IMEX2"):
            with pytest.raises(ValueError, match="linear="):
                evolve(rod, lambda t, u: L(u), u0, (0, 1), bcs=0, method=method, dt=0.1)
        for method in ("RK45", "expm"):
            with pytest.raises(ValueError, match="only used by the methods ETDRK4"):
                evolve(rod, lambda t, u: u, u0, (0, 1), bcs=0, linear=L, method=method)
        with pytest.raises(TypeError, match="nested list"):
            evolve(rod, lambda t, u, v: (u, v), (u0, u0), (0, 1), linear=L, method="ETDRK4", dt=0.1)
        with pytest.raises(TypeError, match="tuple of initial fields"):
            evolve(rod, lambda t, u: u, u0, (0, 1), linear=[[L]], method="ETDRK4", dt=0.1)
        with pytest.raises(ValueError, match="2 x 2"):
            evolve(rod, lambda t, u, v: (u, v), (u0, u0), (0, 1), linear=[[L, None]], method="ETDRK4", dt=0.1)
        with pytest.raises(TypeError, match="Operator or a nested list"):
            evolve(rod, lambda t, u: u, u0, (0, 1), linear=L.toarray(), method="ETDRK4", dt=0.1)

    def test_nonlinear_part(self, rod):
        L = rod.diff_operator(2)
        u0 = np.zeros(rod.shape)
        with pytest.raises(TypeError, match="nonlinear part"):
            evolve(rod, L, u0, (0, 1), bcs=0, linear=L, method="ETDRK4", dt=0.1)
        with pytest.raises(ValueError, match="wrong shape"):
            evolve(rod, lambda t, u: u[1:], u0, (0, 1), bcs=0, linear=L, method="ETDRK4", dt=0.1)
        with pytest.raises(ValueError, match="expected 2 arrays"):
            evolve(rod, lambda t, u, v: u, (u0, u0), (0, 1), linear=[[L, None], [None, L]], method="ETDRK4", dt=0.1)
        with pytest.raises(ValueError, match="expected 2 arrays"):
            evolve(rod, lambda t, u, v: 0.0, (u0, u0), (0, 1), linear=[[L, None], [None, L]], method="IMEX2", dt=0.1)

    def test_solver_options(self, rod):
        with pytest.raises(TypeError, match="solver options"):
            evolve(rod, lambda t, u: u, np.zeros(rod.shape), (0, 1), bcs=0, linear=rod.diff_operator(2),
                   method="ETDRK4", dt=0.1, rtol=1e-8)

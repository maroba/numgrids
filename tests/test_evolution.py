"""evolve: time-dependent boundary values, sources, operator systems, exponential integrator."""

import numpy as np
import numpy.testing as npt
import pytest

import numgrids.evolution
from numgrids import (
    Grid,
    Neumann,
    PolarGrid,
    Robin,
    TimeDependent,
    chebyshev,
    evolve,
    periodic,
    radial,
)

TOL = dict(rtol=1e-10, atol=1e-12)


@pytest.fixture
def rod():
    return Grid(x=chebyshev(24, 0, 1))


class TestTimeDependentBoundaryValues:
    """u = exp(-t) cos(x) solves u_t = u_xx."""

    def exact(self, x, t):
        return np.exp(-t) * np.cos(x)

    @pytest.mark.parametrize("as_operator", [True, False])
    def test_dirichlet(self, rod, as_operator):
        x = rod.coords
        L = rod.laplacian_operator()
        rhs = L if as_operator else (lambda t, u: L(u))
        bcs = {"x_low": TimeDependent(lambda t: np.exp(-t)),
               "x_high": TimeDependent(lambda t: np.exp(-t) * np.cos(1))}
        res = evolve(rod, rhs, np.cos(x), (0, 1), t_eval=[0, 0.3, 1], bcs=bcs, method="BDF", **TOL)
        for t, u in zip(res.t, res.u):
            npt.assert_allclose(u, self.exact(x, t), atol=1e-8)

    def test_neumann_and_robin(self, rod):
        x = rod.coords
        # Outward normal derivative at x = 1 is u_x; Robin at x = 0: u - u_n = u + u_x = exp(-t)
        bcs = {"x_low": Robin(1, 1, TimeDependent(lambda t: np.exp(-t))),
               "x_high": Neumann(TimeDependent(lambda t: -np.exp(-t) * np.sin(1)))}
        res = evolve(rod, rod.laplacian_operator(), np.cos(x), (0, 1), bcs=bcs, method="BDF", **TOL)
        npt.assert_allclose(res.final, self.exact(x, 1), atol=1e-8)

    def test_explicit_method(self):
        grid = Grid(x=chebyshev(12, 0, 1))
        x = grid.coords
        bcs = {"x_low": TimeDependent(lambda t: np.exp(-t)), "x_high": Neumann(TimeDependent(
            lambda t: -np.exp(-t) * np.sin(1)))}
        res = evolve(grid, lambda t, u: grid.laplacian(u), np.cos(x), (0, 0.2), bcs=bcs, rtol=1e-8, atol=1e-10)
        npt.assert_allclose(res.final, self.exact(x, 0.2), atol=1e-6)

    def test_values_varying_along_the_face(self):
        # u = exp(-2t) sin(x) sin(y) on the unit square; boundary values depend on position and time
        grid = Grid(x=chebyshev(14, 0, 1), y=chebyshev(14, 0, 1))
        X, Y = grid.meshed_coords
        exact = lambda t: np.exp(-2 * t) * np.sin(X) * np.sin(Y)
        g = TimeDependent(lambda t: (lambda c: np.exp(-2 * t) * np.sin(c[0]) * np.sin(c[1])))
        res = evolve(grid, grid.laplacian_operator(), exact(0), (0, 0.5), bcs=g, method="BDF", **TOL)
        npt.assert_allclose(res.final, exact(0.5), atol=1e-8)

    def test_initial_state_is_adjusted_at_start_time(self, rod):
        res = evolve(rod, rod.laplacian_operator(), np.zeros(rod.shape), (2.0, 2.1),
                     bcs={"x_low": TimeDependent(np.sin), "x_high": 0}, method="BDF")
        assert res.u[0][0] == pytest.approx(np.sin(2.0))
        assert res.final[0] == pytest.approx(np.sin(2.1))

    def test_per_field_conditions_of_a_system(self, rod):
        x = rod.coords
        g = TimeDependent(lambda t: np.exp(-t))
        res = evolve(rod, lambda t, a, b: (rod.laplacian(a), rod.laplacian(b)),
                     (np.cos(x), 2 * np.cos(x)), (0, 0.5),
                     bcs=({"x_low": g, "x_high": Neumann(TimeDependent(lambda t: -np.exp(-t) * np.sin(1)))},
                          {"x_low": TimeDependent(lambda t: 2 * np.exp(-t)),
                           "x_high": TimeDependent(lambda t: 2 * np.exp(-t) * np.cos(1))}),
                     method="BDF", **TOL)
        a, b = res.final
        npt.assert_allclose(a, self.exact(x, 0.5), atol=1e-7)
        npt.assert_allclose(b, 2 * self.exact(x, 0.5), atol=1e-7)

    def test_only_meaningful_in_evolve(self, rod):
        with pytest.raises(ValueError, match="evolve"):
            rod.laplacian_operator().solve(0, bcs=TimeDependent(np.sin))
        with pytest.raises(TypeError):
            TimeDependent(3.0)
        assert "TimeDependent" in repr(TimeDependent(np.sin))


class TestSource:
    @pytest.mark.parametrize("as_operator", [True, False])
    def test_time_dependent_source(self, rod, as_operator):
        # u = t sin(pi x):  u_t = u_xx + (1 + pi^2 t) sin(pi x)
        x = rod.coords
        L = rod.laplacian_operator()
        rhs = L if as_operator else (lambda t, u: L(u))
        res = evolve(rod, rhs, np.zeros(rod.shape), (0, 1), bcs=0, method="BDF",
                     source=lambda t: (1 + np.pi ** 2 * t) * np.sin(np.pi * x), **TOL)
        npt.assert_allclose(res.final, np.sin(np.pi * x), atol=1e-8)

    @pytest.mark.parametrize("method", ["BDF", "expm"])
    def test_constant_source(self, rod, method):
        # u_t = u_xx + pi^2 sin(pi x), u(0) = 0:  u = (1 - exp(-pi^2 t)) sin(pi x)
        x = rod.coords
        kwargs = TOL if method == "BDF" else {}
        res = evolve(rod, rod.laplacian_operator(), np.zeros(rod.shape), (0, 0.2), bcs=0, method=method,
                     source=np.pi ** 2 * np.sin(np.pi * x), **kwargs)
        npt.assert_allclose(res.final, (1 - np.exp(-np.pi ** 2 * 0.2)) * np.sin(np.pi * x), atol=1e-8)

    def test_source_of_a_system(self, rod):
        # u_t = v, v_t = u_xx + s with u = t^2 sin(pi x): s = (2 + pi^2 t^2) sin(pi x)
        x = rod.coords
        L = rod.laplacian_operator()
        s = lambda t: (None, (2 + np.pi ** 2 * t ** 2) * np.sin(np.pi * x))
        zero = np.zeros(rod.shape)
        for rhs in ([[None, 1], [L, None]], lambda t, u, v: (v, L(u))):
            res = evolve(rod, rhs, (zero, zero), (0, 1), bcs=0, method="DOP853", source=s, **TOL)
            npt.assert_allclose(res.final[0], np.sin(np.pi * x), atol=1e-7)
        with pytest.raises(ValueError, match="tuple of 2"):
            evolve(rod, [[None, 1], [L, None]], (zero, zero), (0, 1), bcs=0, source=zero)


class TestOperatorSystems:
    def test_wave_equation_as_block_operator(self):
        grid = Grid(x=chebyshev(24, 0, np.pi))
        x = grid.coords
        L = grid.laplacian_operator()
        u0, v0 = np.sin(x), np.zeros_like(x)
        t_eval = [0, 0.7, 2.0]
        blocks = evolve(grid, [[None, 1], [L, None]], (u0, v0), (0, 2), t_eval=t_eval, bcs=0,
                        method="DOP853", **TOL)
        for t, u, v in zip(blocks.t, *blocks.u):
            npt.assert_allclose(u, np.cos(t) * np.sin(x), atol=1e-8)
            npt.assert_allclose(v, -np.sin(t) * np.sin(x), atol=1e-8)

    def test_implicit_method_uses_the_block_jacobian(self):
        # Damped wave equation; blocks given as operator, array, scalar and None
        grid = Grid(x=chebyshev(20, 0, np.pi))
        x = grid.coords
        L = grid.laplacian_operator()
        gamma = np.full(grid.shape, 0.5)
        res = evolve(grid, [[None, 1.0], [L, -gamma]], (np.sin(x), np.zeros_like(x)), (0, 3), bcs=0,
                     method="Radau", **TOL)
        omega = np.sqrt(1 - 0.25 ** 2)
        exact = np.exp(-0.25 * 3) * (np.cos(omega * 3) + 0.25 / omega * np.sin(omega * 3)) * np.sin(x)
        npt.assert_allclose(res.final[0], exact, atol=1e-7)

    def test_reaction_diffusion_with_per_field_conditions(self):
        # Linear system a_t = a_xx - a + b, b_t = b_xx with no-flux boundaries conserves b
        grid = Grid(x=chebyshev(20, 0, 1))
        x = grid.coords
        L = grid.laplacian_operator()
        res = evolve(grid, [[L - 1, 1], [None, L]], (np.zeros_like(x), 1 + np.cos(np.pi * x)), (0, 2),
                     bcs=(Neumann(0), Neumann(0)), method="BDF", **TOL)
        a, b = res.final
        assert grid.integrate(b) == pytest.approx(1.0, abs=1e-8)
        assert grid.integrate(a) == pytest.approx(1 - np.exp(-2), abs=1e-7)

    def test_errors(self, rod):
        L = rod.laplacian_operator()
        zero = np.zeros(rod.shape)
        with pytest.raises(TypeError, match="nested list"):
            evolve(rod, L, (zero, zero), (0, 1))
        with pytest.raises(TypeError, match="tuple of initial fields"):
            evolve(rod, [[L]], zero, (0, 1))
        with pytest.raises(ValueError, match="2 x 2"):
            evolve(rod, [[L, None]], (zero, zero), (0, 1))


class TestExponentialIntegrator:
    def test_heat_equation_is_exact_at_any_time(self, rod):
        x = rod.coords
        u0 = np.sin(np.pi * x)
        t_eval = [0, 0.01, 0.5, 0.6, 3.0]                 # uneven steps, long times
        res = evolve(rod, 0.1 * rod.laplacian_operator(), u0, (0, 3), t_eval=t_eval, bcs=0, method="expm")
        assert res.success and res.sol is None
        npt.assert_allclose(res.t, t_eval)
        for t, u in zip(res.t, res.u):
            npt.assert_allclose(u, np.exp(-0.1 * np.pi ** 2 * t) * u0, atol=1e-10)

    def test_default_output_times(self, rod):
        res = evolve(rod, rod.laplacian_operator(), np.sin(np.pi * rod.coords), (0.5, 1.0), bcs=0, method="expm")
        npt.assert_allclose(res.t, [0.5, 1.0])
        npt.assert_allclose(res.final, np.exp(-np.pi ** 2 * 0.5) * np.sin(np.pi * rod.coords), atol=1e-10)

    def test_inhomogeneous_conditions(self, rod):
        x = rod.coords
        # Dirichlet values 1 and 3: u = 1 + 2x + exp(-pi^2 t) sin(pi x)
        res = evolve(rod, rod.laplacian_operator(), 1 + 2 * x + np.sin(np.pi * x), (0, 0.3),
                     bcs={"x_low": 1, "x_high": 3}, method="expm")
        npt.assert_allclose(res.final, 1 + 2 * x + np.exp(-np.pi ** 2 * 0.3) * np.sin(np.pi * x), atol=1e-9)
        # Insulated ends: u = 1 + exp(-pi^2 t) cos(pi x), heat is conserved
        res = evolve(rod, rod.laplacian_operator(), 1 + np.cos(np.pi * x), (0, 0.3), bcs=Neumann(0), method="expm")
        npt.assert_allclose(res.final, 1 + np.exp(-np.pi ** 2 * 0.3) * np.cos(np.pi * x), atol=1e-9)
        # Robin condition: agrees with an implicit method
        bcs = {"x_low": Neumann(0), "x_high": Robin(2.0, 1.0, 0.5)}
        ref = evolve(rod, rod.laplacian_operator(), np.ones(rod.shape), (0, 0.5), bcs=bcs, method="BDF", **TOL)
        res = evolve(rod, rod.laplacian_operator(), np.ones(rod.shape), (0, 0.5), bcs=bcs, method="expm")
        npt.assert_allclose(res.final, ref.final, atol=1e-8)

    def test_schroedinger_equation_is_unitary(self):
        grid = Grid(x=periodic(64, 0, 2 * np.pi))
        x = grid.coords
        H = -0.5 * grid.diff_operator(2)
        res = evolve(grid, -1j * H, np.exp(3j * x), (0, 10), method="expm")
        npt.assert_allclose(res.final, np.exp(-4.5j * 10) * np.exp(3j * x), atol=1e-10)
        # A wave packet keeps its norm to round-off, however long it runs
        psi0 = np.exp(-4 * (x - np.pi) ** 2 + 2j * x)
        res = evolve(grid, -1j * H, psi0, (0, 50), t_eval=np.linspace(0, 50, 6), method="expm")
        for psi in res.u:
            assert grid.norm(psi) == pytest.approx(grid.norm(psi0), rel=1e-12)

    def test_wave_equation_on_a_drum(self):
        # A Bessel mode of the clamped disk oscillates with frequency j_01
        disk = PolarGrid(radial(14, 1.0), periodic(8))
        L = disk.laplacian_operator()
        vals, modes = (-L).eig(k=1, bcs=0)
        omega = np.sqrt(vals[0])
        res = evolve(disk, [[None, 1], [L, None]], (modes[0], np.zeros(disk.shape)), (0, 5),
                     bcs=0, method="expm")
        assert omega == pytest.approx(2.404825557695773)
        npt.assert_allclose(res.final[0], np.cos(omega * 5) * modes[0], atol=1e-9)
        npt.assert_allclose(res.final[1], -omega * np.sin(omega * 5) * modes[0], atol=1e-8)

    def test_large_problems_use_expm_multiply(self, monkeypatch):
        monkeypatch.setattr(numgrids.evolution, "_EXPM_DENSE_LIMIT", 0)
        grid = Grid(x=periodic(32, 0, 2 * np.pi))
        x = grid.coords
        res = evolve(grid, grid.diff_operator(2), np.sin(2 * x), (0, 0.5), t_eval=[0, 0.25, 0.5],
                     method="expm", source=np.cos(x))
        exact = np.exp(-2) * np.sin(2 * x) + (1 - np.exp(-0.5)) * np.cos(x)
        npt.assert_allclose(res.final, exact, atol=1e-10)

    def test_backward_in_time(self):
        grid = Grid(x=periodic(32, 0, 2 * np.pi))
        x = grid.coords
        H = -0.5 * grid.diff_operator(2)
        psi0 = np.exp(-4 * (x - np.pi) ** 2 + 2j * x)
        forward = evolve(grid, -1j * H, psi0, (0, 2), method="expm")
        back = evolve(grid, -1j * H, forward.final, (2, 0), method="expm")
        npt.assert_allclose(back.final, psi0, atol=1e-11)

    def test_errors(self, rod):
        L = rod.laplacian_operator()
        u0 = np.zeros(rod.shape)
        with pytest.raises(ValueError, match="linear right-hand side"):
            evolve(rod, lambda t, u: L(u), u0, (0, 1), bcs=0, method="expm")
        with pytest.raises(ValueError, match="constant in time"):
            evolve(rod, L, u0, (0, 1), bcs=TimeDependent(np.sin), method="expm")
        with pytest.raises(ValueError, match="constant in time"):
            evolve(rod, L, u0, (0, 1), bcs=0, method="expm", source=lambda t: u0)
        with pytest.raises(TypeError, match="solver options"):
            evolve(rod, L, u0, (0, 1), bcs=0, method="expm", rtol=1e-8)
        with pytest.raises(ValueError, match="t_span"):
            evolve(rod, L, u0, (0, 1), bcs=0, method="expm", t_eval=[0, 2])
        with pytest.raises(ValueError, match="sorted"):
            evolve(rod, L, u0, (0, 1), bcs=0, method="expm", t_eval=[0.5, 0.2])

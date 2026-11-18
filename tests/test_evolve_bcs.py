"""Neumann and Robin boundary conditions in time evolution."""
import numpy as np
import numpy.testing as npt
import pytest

from numgrids import Grid, Neumann, PolarGrid, Robin, chebyshev, evolve, periodic, radial

TOL = dict(rtol=1e-10, atol=1e-12)


@pytest.fixture
def rod():
    return Grid(x=chebyshev(30, 0, 1))


@pytest.mark.parametrize("operator_rhs", [True, False])
def test_neumann_and_robin_heat_equation(rod, operator_rhs):
    # u_t = u_xx, u'(0) = 0, u(1) + u'(1) = 0: u = cos(mu x) exp(-mu^2 t), mu tan(mu) = 1
    mu = 0.8603335890193798
    x = rod.coords
    u0 = np.cos(mu * x)
    rhs = rod.laplacian_operator() if operator_rhs else (lambda t, u: rod.laplacian(u))
    res = evolve(rod, rhs, u0, (0, 1), t_eval=[0, 0.5, 1],
                 bcs={"x_low": Neumann(0), "x_high": Robin(1, 1, 0)}, method="BDF", **TOL)
    for t, u in zip(res.t, res.u):
        npt.assert_allclose(u, np.exp(-mu ** 2 * t) * u0, atol=1e-8)


def test_conditions_hold_at_all_times(rod):
    x = rod.coords
    res = evolve(rod, rod.laplacian_operator(), np.zeros_like(x), (0, 2),
                 t_eval=np.linspace(0, 2, 5), bcs={"x_low": 0, "x_high": Neumann(1)},
                 method="BDF", **TOL)
    for u in res.u:
        assert u[0] == pytest.approx(0, abs=1e-12)
        assert rod.diff(u, 1)[-1] == pytest.approx(1, abs=1e-9)


def test_inhomogeneous_neumann_steady_state(rod):
    x = rod.coords
    res = evolve(rod, rod.laplacian_operator(), np.zeros_like(x), (0, 20),
                 bcs={"x_low": 0, "x_high": Neumann(1)}, method="BDF", **TOL)
    npt.assert_allclose(res.final, x, atol=1e-10)


def test_insulated_square_conserves_heat():
    grid = Grid(x=chebyshev(20, 0, 1), y=chebyshev(20, 0, 1))
    X, Y = grid.meshed_coords
    u0 = np.exp(-20 * ((X - 0.3) ** 2 + (Y - 0.6) ** 2))
    res = evolve(grid, grid.laplacian_operator(), u0, (0, 3), t_eval=[0.05, 3],
                 bcs=Neumann(0), method="BDF")
    total = grid.integrate(u0)
    assert grid.integrate(res.u[0]) == pytest.approx(total, rel=1e-5)
    # Equilibrium: the mean temperature everywhere.
    npt.assert_allclose(res.final, total, rtol=1e-4)


def test_insulated_disk_conserves_heat():
    grid = PolarGrid(radial(16, 1.0), periodic(24))
    X, Y = grid.cartesian_coords
    u0 = np.exp(-10 * ((X - 0.4) ** 2 + Y ** 2))
    res = evolve(grid, grid.laplacian_operator(), u0, (0, 0.1), bcs=Neumann(0), method="BDF",
                 rtol=1e-9, atol=1e-11)
    assert grid.integrate(res.final) == pytest.approx(grid.integrate(u0), rel=1e-7)


def test_explicit_method(rod):
    x = rod.coords
    res = evolve(rod, lambda t, u: rod.laplacian(u), np.cos(np.pi * x), (0, 0.1),
                 bcs=Neumann(0), method="RK45", rtol=1e-8)
    npt.assert_allclose(res.final, np.exp(-np.pi ** 2 * 0.1) * np.cos(np.pi * x), atol=1e-5)


def test_per_field_conditions(rod):
    x = rod.coords
    u0, v0 = np.cos(np.pi * x), np.sin(np.pi * x)
    res = evolve(rod, lambda t, u, v: (rod.laplacian(u), rod.laplacian(v)), (u0, v0), (0, 0.5),
                 bcs=(Neumann(0), 0), method="BDF", **TOL)
    decay = np.exp(-np.pi ** 2 * 0.5)
    npt.assert_allclose(res.final[0], decay * u0, atol=1e-9)
    npt.assert_allclose(res.final[1], decay * v0, atol=1e-9)


def test_per_field_none(rod):
    # The second field has no conditions: v_t = -v decays everywhere.
    x = rod.coords
    res = evolve(rod, lambda t, u, v: (rod.laplacian(u), -v), (np.cos(np.pi * x), 1 + x), (0, 1),
                 bcs=(Neumann(0), None), **TOL)
    npt.assert_allclose(res.final[1], np.exp(-1) * (1 + x), rtol=1e-8)


def test_wave_equation_with_neumann_on_first_field(rod):
    x = rod.coords
    res = evolve(rod, lambda t, u, v: (v, rod.laplacian(u)), (np.cos(np.pi * x), 0 * x), (0, 1),
                 bcs=Neumann(0), **TOL)
    npt.assert_allclose(res.final[0], -np.cos(np.pi * x), atol=1e-8)


def test_dirichlet_unchanged(rod):
    # Dirichlet conditions behave as before: values imposed and held fixed.
    x = rod.coords
    res = evolve(rod, rod.laplacian_operator(), np.zeros_like(x), (0, 5),
                 bcs={"x_low": 1, "x_high": 3}, method="Radau", **TOL)
    npt.assert_allclose(res.final, 1 + 2 * x, atol=1e-8)
    assert res.u[0][0] == 1 and res.u[0][-1] == 3


def test_singular_boundary_system(rod):
    with pytest.raises(ValueError, match="singular boundary system"):
        evolve(rod, lambda t, u: u, np.zeros(rod.shape), (0, 1), bcs=Robin(0, 0))

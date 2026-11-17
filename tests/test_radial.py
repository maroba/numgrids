"""Tests for radial axes through the origin (full disks, cylinders and balls)."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import j0, jn_zeros

from numgrids import (
    AxisType,
    CylindricalGrid,
    Diff,
    Grid,
    Interpolator,
    Neumann,
    PolarGrid,
    RadialAxis,
    SphericalGrid,
    chebyshev,
    create_axis,
    evolve,
    load_grid,
    periodic,
    radial,
    save_grid,
)
from numgrids.boundary import BoundaryFace


def disk(nr=20, nphi=32, radius=1.0):
    return PolarGrid(radial(nr, radius), periodic(nphi))


def bessel_zeros(count):
    zeros = sorted(z for m in range(8) for z in jn_zeros(m, 4) for _ in range(1 if m == 0 else 2))
    return np.array(zeros[:count])


# ── the axis ──────────────────────────────────────────────────────────

class TestRadialAxis:

    def test_points(self):
        ax = radial(6, 2.0)
        r = ax.coords
        assert len(r) == 6
        assert np.all(r > 0) and np.all(np.diff(r) > 0)
        assert r[-1] == 2.0
        # The points are the positive half of 12 Chebyshev nodes on [-2, 2]:
        npt.assert_allclose(ax.diameter_coords, chebyshev(12, -2, 2).coords, atol=1e-14)

    def test_boundary(self):
        ax = radial(6)
        assert ax.boundary_sides == ("high",)
        assert ax.boundary == slice(None, -1)
        assert not ax.periodic

    def test_factories(self):
        assert isinstance(create_axis(AxisType.RADIAL, 5, 0, 1), RadialAxis)
        assert isinstance(create_axis("radial", 5, 0, 1), RadialAxis)
        assert radial(5, name="rho").name == "rho"

    def test_must_start_at_origin(self):
        with pytest.raises(ValueError, match="origin"):
            RadialAxis(5, 0.5, 1.0)

    def test_resized_and_eq(self):
        ax = radial(5, 3.0)
        assert ax.resized(9) == radial(9, 3.0)

    @pytest.mark.parametrize("power", [1, 2])
    def test_radial_quadrature_exact_for_even_polynomials(self, power):
        ax = radial(8, 1.5)
        w = ax.radial_quadrature_weights(power)
        r = ax.coords
        for k in range(0, 14, 2):
            exact = 1.5 ** (k + power + 1) / (k + power + 1)
            assert w @ r ** k == pytest.approx(exact, rel=1e-12)


# ── polar grid: the full disk ─────────────────────────────────────────

class TestDisk:

    def test_only_rim_is_boundary(self):
        g = disk()
        assert list(g.faces) == ["0_high"]
        assert "r_high" in g.faces
        assert not g.boundary[:-1].any() and g.boundary[-1].all()
        with pytest.raises(ValueError, match="'high'"):
            BoundaryFace(g, 0, "low")

    def test_radial_derivatives(self):
        g = disk()
        X, Y = g.cartesian_coords
        _, Phi = g.meshed_coords
        f = np.exp(X) * np.sin(2 * Y) + X ** 3
        fx = np.exp(X) * np.sin(2 * Y) + 3 * X ** 2
        fy = 2 * np.exp(X) * np.cos(2 * Y)
        df_dr = np.cos(Phi) * fx + np.sin(Phi) * fy
        npt.assert_allclose(g.diff(f, axis="r"), df_dr, atol=1e-11)
        d = Diff(g, 1, 0)
        npt.assert_allclose((d.as_matrix() @ f.ravel()).reshape(g.shape), df_dr, atol=1e-11)

    def test_second_radial_derivative_matrix_matches_operator(self):
        g = disk(12, 16)
        X, Y = g.cartesian_coords
        f = np.cos(X + 2 * Y)
        d = Diff(g, 2, 0)
        npt.assert_allclose((d.as_matrix() @ f.ravel()).reshape(g.shape), d(f), atol=1e-10)

    def test_laplacian_through_center(self):
        g = disk()
        X, Y = g.cartesian_coords
        for f, lap in [(X ** 2 + Y ** 2, 4.0), (X ** 3 - 3 * X * Y ** 2, 0.0), (np.exp(X), np.exp(X))]:
            npt.assert_allclose(g.laplacian(f), lap * np.ones(g.shape), atol=1e-9)
            npt.assert_allclose(g.laplacian_operator()(f), lap * np.ones(g.shape), atol=1e-9)

    def test_vector_calculus(self):
        g = disk()
        X, Y = g.cartesian_coords
        f = X ** 2 * Y + np.sin(Y)
        npt.assert_allclose(g.divergence(*g.gradient(f)), g.laplacian(f), atol=1e-9)
        npt.assert_allclose(g.curl(*g.gradient(f)), 0, atol=1e-9)

    def test_integration(self):
        g = disk(radius=2.0)
        X, Y = g.cartesian_coords
        assert g.integrate(np.ones(g.shape)) == pytest.approx(4 * np.pi, rel=1e-13)
        assert g.integrate(X ** 2) == pytest.approx(np.pi * 2.0 ** 4 / 4, rel=1e-13)
        assert g.integrate(np.exp(X)) == pytest.approx(2 * np.pi * 2 * 1.590636854637329, rel=1e-12)

    def test_eigenvalues_are_bessel_zeros(self):
        vals, modes = (-disk().laplacian_operator()).eig(k=9, bcs=0)
        npt.assert_allclose(np.sqrt(vals), bessel_zeros(9), rtol=1e-11)
        R = disk().meshed_coords[0]
        exact = j0(bessel_zeros(1)[0] * R)
        npt.assert_allclose(modes[0] / modes[0].max(), exact / exact.max(), atol=1e-10)

    def test_spectrum_is_real_and_negative(self):
        # This is what makes time integration stable.
        g = disk(14, 16)
        L, _ = g.laplacian_operator().with_bcs(0.0, bcs=0)
        interior = ~g.boundary.ravel()
        ev = np.linalg.eigvals(L.toarray()[np.ix_(interior, interior)])
        assert np.abs(ev.imag).max() < 1e-8 * np.abs(ev).max()
        assert ev.real.max() < 0

    def test_poisson(self):
        g = disk()
        X, Y = g.cartesian_coords
        u = g.laplacian_operator().solve(4.0, bcs=1.0)
        npt.assert_allclose(u, X ** 2 + Y ** 2, atol=1e-9)
        # Neumann at the rim: u = r^2 + c, du/dn = 2
        u = g.laplacian_operator().solve(4.0, bcs=Neumann(2.0))
        npt.assert_allclose(u - u.mean(), X ** 2 + Y ** 2 - (X ** 2 + Y ** 2).mean(), atol=1e-8)

    def test_wave_equation_conserves_energy(self):
        g = disk(16, 24)
        X, Y = g.cartesian_coords
        u0 = np.exp(-8 * ((X - 0.2) ** 2 + Y ** 2)) * (1 - X ** 2 - Y ** 2)
        res = evolve(g, lambda t, u, v: (v, g.laplacian(u)), (u0, 0 * u0), (0, 4),
                     t_eval=[0, 2, 4], bcs={"r_high": 0}, rtol=1e-9, atol=1e-11)

        def energy(u, v):
            gr, gp = g.gradient(u)
            return 0.5 * g.integrate(v ** 2 + gr ** 2 + gp ** 2)

        E = [energy(u, v) for u, v in zip(*res.u)]
        npt.assert_allclose(E, E[0], rtol=1e-4)

    def test_heat_equation_bessel_decay(self):
        g = disk()
        R = g.meshed_coords[0]
        k = bessel_zeros(1)[0]
        res = evolve(g, g.laplacian_operator(), j0(k * R), (0, 0.1), t_eval=[0.1], bcs=0,
                     method="BDF", rtol=1e-10, atol=1e-12)
        npt.assert_allclose(res.final, np.exp(-k ** 2 * 0.1) * j0(k * R), atol=1e-8)

    @pytest.mark.parametrize("method", ["spectral", "cubic"])
    def test_interpolation_through_center(self, method):
        g = disk(16, 32)
        X, Y = g.cartesian_coords
        f = np.exp(X) * np.cos(2 * Y)
        pts = np.array([[0.0, 0.3], [0.01, 2.0], [0.5, 4.0], [1.0, 6.2]])
        exact = np.exp(pts[:, 0] * np.cos(pts[:, 1])) * np.cos(2 * pts[:, 0] * np.sin(pts[:, 1]))
        tol = 1e-11 if method == "spectral" else 1e-2
        npt.assert_allclose(Interpolator(g, f, method=method)(pts), exact, atol=tol)

    def test_interpolation_onto_other_disk(self):
        g = disk(16, 32)
        fine = disk(25, 40)
        X, Y = g.cartesian_coords
        Xf, Yf = fine.cartesian_coords
        npt.assert_allclose(g.interpolate(np.sin(X + Y), fine, method="spectral"), np.sin(Xf + Yf), atol=1e-10)

    def test_refine_keeps_folding(self):
        g = disk(10, 16).refine()
        X, Y = g.cartesian_coords
        assert g.shape == (20, 32)
        npt.assert_allclose(g.laplacian(X ** 2 + Y ** 2), 4.0, atol=1e-8)

    def test_save_load(self, tmp_path):
        g = disk(8, 12, 2.5)
        save_grid(tmp_path / "d.npz", g)
        loaded, _ = load_grid(tmp_path / "d.npz")
        assert loaded == g and isinstance(loaded.axes[0], RadialAxis)

    def test_plot_fills_center(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = disk(8, 12)
        ax = g.plot(g.cartesian_coords[0])
        mesh = ax.collections[0]
        assert mesh.get_array().size == (8 + 1) * (12 + 1)
        plt.close("all")


# ── requirements ───────────────────────────────────────────────────────

class TestRequirements:

    def test_odd_number_of_azimuthal_points(self):
        with pytest.raises(ValueError, match="even number"):
            PolarGrid(radial(8), periodic(15))

    def test_non_periodic_azimuth(self):
        with pytest.raises(ValueError, match="periodic"):
            PolarGrid(radial(8), chebyshev(16, 0, 2 * np.pi))

    def test_plain_grid_cannot_differentiate(self):
        g = Grid(radial(8), periodic(16))
        with pytest.raises(ValueError, match="PolarGrid"):
            g.diff(np.zeros(g.shape))

    def test_asymmetric_polar_axis_on_sphere(self):
        with pytest.raises(ValueError, match="symmetric"):
            SphericalGrid(radial(6), chebyshev(8, 0.1, 3.0), periodic(8))

    def test_radial_axis_must_be_first(self):
        with pytest.raises(ValueError, match="first"):
            CylindricalGrid(chebyshev(6, 0.1, 1), periodic(8), radial(6))


# ── full cylinder and ball ─────────────────────────────────────────────

def test_full_cylinder():
    g = CylindricalGrid(radial(10, 2.0), periodic(16), chebyshev(8, -1, 1))
    X, Y, Z = g.cartesian_coords
    assert sorted(g.faces) == ["0_high", "2_high", "2_low"]
    assert g.integrate(np.ones(g.shape)) == pytest.approx(8 * np.pi, rel=1e-13)
    npt.assert_allclose(g.laplacian(X ** 2 + Y ** 2 + Z ** 2), 6.0, atol=1e-9)
    # The end caps are full disks:
    assert g.faces["z_high"].integrate(np.ones(g.shape)) == pytest.approx(4 * np.pi, rel=1e-13)


def test_full_ball():
    eps = 1e-3
    g = SphericalGrid(radial(12, 1.0), chebyshev(14, eps, np.pi - eps), periodic(16))
    X, Y, Z = g.cartesian_coords
    npt.assert_allclose(g.laplacian(X ** 2 + Y ** 2 + Z ** 2), 6.0, atol=1e-9)
    # Only the polar caps theta < eps are missing from the volume:
    assert g.integrate(np.ones(g.shape)) == pytest.approx(4 / 3 * np.pi, rel=1e-5)
    vals, _ = (-g.laplacian_operator()).eig(k=9, bcs={"r_high": 0})
    expected = [np.pi] + [4.493409457909064] * 3 + [5.763459196894550] * 5
    npt.assert_allclose(np.sqrt(vals), expected, rtol=1e-9)

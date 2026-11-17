"""Tests for colatitude axes through the poles (full spheres and balls)."""

import numpy as np
import numpy.testing as npt
import pytest

try:  # SciPy >= 1.15
    from scipy.special import sph_harm_y
except ImportError:  # older SciPy: sph_harm(m, l, azimuth, polar)
    from scipy.special import sph_harm

    def sph_harm_y(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)

from numgrids import (
    AxisType,
    ColatitudeAxis,
    Diff,
    Interpolator,
    SphericalGrid,
    SphericalSurfaceGrid,
    chebyshev,
    colatitude,
    create_axis,
    evolve,
    load_grid,
    periodic,
    radial,
    save_grid,
)


def sphere(nt=16, nphi=32, radius=1.0):
    return SphericalSurfaceGrid(colatitude(nt), periodic(nphi), radius=radius)


def ball(nr=12, nt=14, nphi=16):
    return SphericalGrid(radial(nr, 1.0), colatitude(nt), periodic(nphi))


def real_harmonic(l, m):
    factor = 1.0 if m == 0 else np.sqrt(2)
    return lambda theta, phi: factor * np.real(sph_harm_y(l, m, theta, phi))


# ── the axis ──────────────────────────────────────────────────────────

class TestColatitudeAxis:

    def test_points(self):
        ax = colatitude(6)
        th = ax.coords
        npt.assert_allclose(th, (np.arange(6) + 0.5) * np.pi / 6)
        npt.assert_allclose(th + th[::-1], np.pi)          # symmetric about the equator
        assert th[0] > 0 and th[-1] < np.pi                # no point on a pole
        assert len(ax.great_circle_coords) == 12

    def test_no_boundary(self):
        ax = colatitude(6)
        assert ax.boundary_sides == ()
        assert ax.boundary == slice(None, None)
        assert not ax.periodic

    def test_factories(self):
        assert isinstance(create_axis(AxisType.COLATITUDE, 5, 0, np.pi), ColatitudeAxis)
        assert isinstance(create_axis("colatitude", 5, 0, np.pi), ColatitudeAxis)
        assert colatitude(5, name="t").name == "t"
        assert colatitude(5).resized(8) == colatitude(8)

    def test_must_span_pole_to_pole(self):
        with pytest.raises(ValueError, match="pole to pole"):
            ColatitudeAxis(5, 0.1, np.pi)

    def test_fejer_weights_exact(self):
        ax = colatitude(10)
        w = ax.sine_quadrature_weights()
        th = ax.coords
        for k in range(10):
            exact = 0.0 if k % 2 else 2.0 / (k + 1)        # int_{-1}^{1} x^k dx, x = cos(theta)
            assert w @ np.cos(th) ** k == pytest.approx(exact, abs=1e-14)


# ── sphere surface ─────────────────────────────────────────────────────

class TestSphereSurface:

    def test_no_faces(self):
        assert list(sphere().faces) == []
        assert not sphere().boundary.any()

    def test_derivative_across_poles(self):
        s = sphere()
        T, P = s.meshed_coords
        X, Y, Z = s.cartesian_coords
        f = X + Z ** 3 + np.exp(Y)
        # d/dtheta of x, z^3, exp(y)
        df = np.cos(T) * np.cos(P) - 3 * Z ** 2 * np.sin(T) + np.exp(Y) * np.cos(T) * np.sin(P)
        npt.assert_allclose(s.diff(f, axis="theta"), df, atol=1e-11)
        d = Diff(s, 1, 0)
        npt.assert_allclose((d.as_matrix() @ f.ravel()).reshape(s.shape), df, atol=1e-11)
        d2 = Diff(s, 2, 0)
        npt.assert_allclose((d2.as_matrix() @ f.ravel()).reshape(s.shape), d2(f), atol=1e-10)

    @pytest.mark.parametrize("l, m", [(0, 0), (1, 0), (2, 1), (3, 3), (5, 2)])
    def test_spherical_harmonics_are_eigenfunctions(self, l, m):
        s = sphere()
        Y = s.evaluate(real_harmonic(l, m))
        npt.assert_allclose(s.laplacian(Y), -l * (l + 1) * Y, atol=1e-10)
        npt.assert_allclose(s.laplacian_operator()(Y), -l * (l + 1) * Y, atol=1e-10)

    def test_laplace_beltrami_spectrum(self):
        vals, _ = (-sphere().laplacian_operator()).eig(k=25)
        expected = [l * (l + 1) for l in range(5) for _ in range(2 * l + 1)]
        npt.assert_allclose(vals, expected, atol=1e-9)

    def test_radius_scaling(self):
        vals, _ = (-sphere(radius=2.0).laplacian_operator()).eig(k=4)
        npt.assert_allclose(vals, np.array([0, 2, 2, 2]) / 4, atol=1e-10)
        s = sphere(radius=2.0)
        assert s.integrate(np.ones(s.shape)) == pytest.approx(16 * np.pi, rel=1e-14)

    def test_orthonormality(self):
        s = sphere()
        harmonics = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 2), (3, 1), (4, 3)]
        Ys = [s.evaluate(real_harmonic(l, m)) for l, m in harmonics]
        gram = np.array([[s.integrate(a * b) for b in Ys] for a in Ys])
        npt.assert_allclose(gram, np.eye(len(Ys)), atol=1e-13)

    def test_heat_equation(self):
        s = sphere()
        Y = s.evaluate(real_harmonic(3, 1))
        res = evolve(s, s.laplacian_operator(), Y, (0, 0.05), t_eval=[0.05], method="BDF",
                     rtol=1e-10, atol=1e-12)
        npt.assert_allclose(res.final, np.exp(-12 * 0.05) * Y, atol=1e-8)

    def test_wave_equation_is_stable(self):
        s = sphere(16, 32)
        X, Y, Z = s.cartesian_coords
        u0 = np.exp(-5 * ((X - 0.6) ** 2 + Y ** 2 + (Z - 0.8) ** 2))      # near the north pole
        res = evolve(s, lambda t, u, v: (v, s.laplacian(u)), (u0, 0 * u0), (0, 6),
                     t_eval=[0, 3, 6], rtol=1e-9, atol=1e-11)

        def energy(u, v):
            gt, gp = s.gradient(u)
            return 0.5 * s.integrate(v ** 2 + gt ** 2 + gp ** 2)

        E = [energy(u, v) for u, v in zip(*res.u)]
        npt.assert_allclose(E, E[0], rtol=1e-4)

    def test_spectrum_real_and_nonpositive(self):
        s = sphere(10, 16)
        ev = np.linalg.eigvals(s.laplacian_operator().as_matrix().toarray())
        assert np.abs(ev.imag).max() < 1e-8 * np.abs(ev).max()
        assert ev.real.max() < 1e-8

    @pytest.mark.parametrize("method", ["spectral", "cubic"])
    def test_interpolation_at_poles(self, method):
        s = sphere()
        X, Y, Z = s.cartesian_coords
        f = X + 2 * Y + 3 * Z
        pts = np.array([[0.0, 0.4], [np.pi, 2.0], [0.01, 5.0], [1.2, 6.27]])
        exact = (np.sin(pts[:, 0]) * (np.cos(pts[:, 1]) + 2 * np.sin(pts[:, 1])) + 3 * np.cos(pts[:, 0]))
        tol = 1e-12 if method == "spectral" else 1e-2
        npt.assert_allclose(Interpolator(s, f, method=method)(pts), exact, atol=tol)

    def test_refine_and_io(self, tmp_path):
        s = sphere(8, 16, radius=1.5).refine()
        assert s.shape == (16, 32) and s.radius == 1.5
        save_grid(tmp_path / "s.npz", s)
        loaded, _ = load_grid(tmp_path / "s.npz")
        assert loaded == s and isinstance(loaded.axes[0], ColatitudeAxis)

    def test_plot_in_coordinate_space(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = sphere(8, 16)
        assert s.plot(s.cartesian_coords[2]).get_xlabel() == "$theta$"
        plt.close("all")


# ── full ball ───────────────────────────────────────────────────────────

class TestBall:

    def test_only_surface_is_boundary(self):
        assert list(ball().faces) == ["0_high"]

    def test_integrals_are_exact(self):
        b = ball()
        R = b.meshed_coords[0]
        X, Y, Z = b.cartesian_coords
        assert b.integrate(np.ones(b.shape)) == pytest.approx(4 / 3 * np.pi, rel=1e-14)
        assert b.integrate(R ** 2) == pytest.approx(4 * np.pi / 5, rel=1e-13)
        assert b.integrate(Z ** 2) == pytest.approx(4 * np.pi / 15, rel=1e-13)
        assert b.faces["r_high"].integrate(np.ones(b.shape)) == pytest.approx(4 * np.pi, rel=1e-14)

    def test_laplacian_everywhere(self):
        b = ball()
        X, Y, Z = b.cartesian_coords
        for f, lap in [(X ** 2 + Y ** 2 + Z ** 2, 6.0 + 0 * X), (Z ** 3, 6 * Z), (X * Y * Z, 0 * X),
                       (np.exp(Z), np.exp(Z))]:
            npt.assert_allclose(b.laplacian(f), lap, atol=1e-9)
            npt.assert_allclose(b.laplacian_operator()(f), lap, atol=1e-9)

    def test_eigenvalues(self):
        vals, _ = (-ball().laplacian_operator()).eig(k=9, bcs=0)
        expected = [np.pi] + [4.493409457909064] * 3 + [5.763459196894550] * 5
        npt.assert_allclose(np.sqrt(vals), expected, rtol=1e-10)

    def test_poisson(self):
        b = ball()
        X, Y, Z = b.cartesian_coords
        u = b.laplacian_operator().solve(lambda r, t, p: 6 * r * np.cos(t), bcs=lambda c: (c[0] * np.cos(c[1])) ** 3)
        npt.assert_allclose(u, Z ** 3, atol=1e-9)

    def test_interpolation_at_center_and_poles(self):
        b = ball()
        X, Y, Z = b.cartesian_coords
        f = np.exp(X + 2 * Z)
        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 1.0], [1.0, np.pi, 3.0], [0.3, 1.0, 2.0]])
        x = pts[:, 0] * np.sin(pts[:, 1]) * np.cos(pts[:, 2])
        z = pts[:, 0] * np.cos(pts[:, 1])
        npt.assert_allclose(b.interpolate(f, pts, method="spectral"), np.exp(x + 2 * z), atol=1e-10)


# ── requirements and compatibility ─────────────────────────────────────

def test_odd_azimuth_rejected():
    with pytest.raises(ValueError, match="even number"):
        SphericalSurfaceGrid(colatitude(8), periodic(15))


def test_colatitude_only_as_polar_axis():
    with pytest.raises(ValueError):
        SphericalGrid(colatitude(6), colatitude(6), periodic(8))


def test_chebyshev_polar_axis_still_works():
    g = SphericalGrid(chebyshev(8, 0.5, 1.0), chebyshev(10, 0.2, np.pi - 0.2), periodic(12))
    R = g.meshed_coords[0]
    npt.assert_allclose(g.laplacian(R ** 2), 6.0, atol=1e-8)


# ── robustness of the sparse eigenvalue solver ─────────────────────────

def test_sparse_eig_with_zero_eigenvalue_and_degeneracies():
    # The sphere has an exact zero eigenvalue (constants) and degenerate
    # clusters; the iterative solver must neither lose accuracy nor members.
    s = sphere(24, 48)
    expected = [l * (l + 1) for l in range(4) for _ in range(2 * l + 1)]
    vals, _ = (-s.laplacian_operator()).eig(k=16, dense_threshold=10)
    assert np.isrealobj(vals)
    npt.assert_allclose(vals, expected, atol=1e-9)


def test_sparse_eig_does_not_miss_degenerate_eigenvalues():
    b = ball()
    assert b.size > 2000      # uses the sparse (Arnoldi) path
    expected = [np.pi] + [4.493409457909064] * 3 + [5.763459196894550] * 5
    vals, _ = (-b.laplacian_operator()).eig(k=9, bcs=0)
    npt.assert_allclose(np.sqrt(vals), expected, rtol=1e-10)
    # Results are reproducible:
    vals2, _ = (-b.laplacian_operator()).eig(k=9, bcs=0)
    npt.assert_array_equal(vals, vals2)

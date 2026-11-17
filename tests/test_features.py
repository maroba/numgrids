"""Tests for features added in 0.5.0: new coordinate systems, spectral
interpolation and quadrature, boundary-face operators, time evolution,
convergence studies, xarray interoperability and plotting."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import numpy.testing as npt  # noqa: E402
import pytest  # noqa: E402

from numgrids import (  # noqa: E402
    CurvilinearGrid,
    CylindricalGrid,
    Dirichlet,
    EllipticCylindricalGrid,
    Grid,
    Integral,
    Interpolator,
    Neumann,
    OblateSpheroidalGrid,
    PolarGrid,
    ProlateSpheroidalGrid,
    Robin,
    SphericalGrid,
    chebyshev,
    convergence_study,
    equidistant,
    evolve,
    load_grid,
    logarithmic,
    periodic,
    resolve_bcs,
    save_grid,
)

# ── coordinate systems ────────────────────────────────────────────────

BUILTIN_GRIDS = {
    "polar": lambda: PolarGrid(chebyshev(14, 0.5, 1.5), periodic(16)),
    "cylindrical": lambda: CylindricalGrid(chebyshev(14, 0.5, 1.5), periodic(16), chebyshev(10, -1, 1)),
    "spherical": lambda: SphericalGrid(chebyshev(14, 0.5, 1.5), chebyshev(14, 0.4, 2.7), periodic(16)),
    "elliptic": lambda: EllipticCylindricalGrid(chebyshev(14, 0.3, 1.2), periodic(24), chebyshev(10, -1, 1), a=1.5),
    "prolate": lambda: ProlateSpheroidalGrid(chebyshev(16, 0.3, 1.2), chebyshev(16, 0.4, 2.7), periodic(16), a=0.8),
    "oblate": lambda: OblateSpheroidalGrid(chebyshev(16, 0.3, 1.2), chebyshev(16, -1.2, 1.2), periodic(16), a=1.3),
}


@pytest.mark.parametrize("name", BUILTIN_GRIDS)
def test_scale_factors_match_cartesian_mapping(name):
    """h_i = |dX/dq_i|, checked with spectral derivatives of the Cartesian map."""
    grid = BUILTIN_GRIDS[name]()
    X = grid.cartesian_coords
    for i, h in enumerate(grid.scale_factors):
        dX = [grid.diff(Xc, 1, i) for Xc in X]
        npt.assert_allclose(np.sqrt(sum(d ** 2 for d in dX)), h, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", BUILTIN_GRIDS)
def test_laplacian_of_cartesian_quadratic(name):
    grid = BUILTIN_GRIDS[name]()
    X = grid.cartesian_coords
    r2 = sum(Xc ** 2 for Xc in X)
    expected = 2 * grid.ndims
    npt.assert_allclose(grid.laplacian(r2), expected, rtol=1e-6)
    npt.assert_allclose(grid.laplacian_operator()(r2), expected, rtol=1e-6)


@pytest.mark.parametrize("name", BUILTIN_GRIDS)
def test_builtin_grid_save_load_roundtrip(name, tmp_path):
    grid = BUILTIN_GRIDS[name]()
    save_grid(tmp_path / "g.npz", grid)
    loaded, _ = load_grid(tmp_path / "g.npz")
    assert type(loaded) is type(grid)
    assert loaded == grid
    for a, b in zip(loaded.scale_factors, grid.scale_factors):
        npt.assert_allclose(a, b)


def test_params_and_repr():
    grid = BUILTIN_GRIDS["prolate"]()
    assert grid.a == 0.8
    assert "a=0.8" in repr(grid)
    assert grid.refine().a == 0.8


def test_custom_curvilinear_grid_without_cartesian():
    grid = CurvilinearGrid(chebyshev(5, 1, 2), periodic(6),
                           scale_factors=(lambda c: np.ones_like(c[0]), lambda c: c[0]))
    with pytest.raises(NotImplementedError):
        grid.cartesian_coords
    with pytest.raises(TypeError, match="custom scale factors"):
        save_grid("never_written.npz", grid)


def test_integral_coordinate_space():
    grid = PolarGrid(chebyshev(10, 0, 1), periodic(16))
    one = np.ones(grid.shape)
    assert Integral(grid)(one) == pytest.approx(np.pi)
    assert Integral(grid, volume_element=False)(one) == pytest.approx(2 * np.pi)


def test_volume_of_prolate_spheroid():
    a, mu0 = 1.0, 0.8
    grid = ProlateSpheroidalGrid(chebyshev(20, 0, mu0), chebyshev(20, 0, np.pi), periodic(8), a=a)
    semi_major, semi_minor = a * np.cosh(mu0), a * np.sinh(mu0)
    expected = 4 / 3 * np.pi * semi_minor ** 2 * semi_major
    assert grid.integrate(np.ones(grid.shape)) == pytest.approx(expected, rel=1e-10)


# ── quadrature ────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [2, 3, 8, 9, 20])
def test_clenshaw_curtis_exact_for_polynomials(n):
    ax = chebyshev(n, -1, 3)
    w = ax.quadrature_weights
    x = ax.coords
    for k in range(n):
        assert w @ x ** k == pytest.approx((3 ** (k + 1) - (-1) ** (k + 1)) / (k + 1), rel=1e-10)


def test_quadrature_weights_readonly():
    with pytest.raises(ValueError):
        chebyshev(5).quadrature_weights[0] = 1


def test_integral_log_and_equidistant():
    grid = Grid(logarithmic(200, 1, 10))
    assert Integral(grid)(1 / grid.coords) == pytest.approx(np.log(10), rel=1e-10)
    errors = []
    for n in (50, 100):
        grid = Grid(equidistant(n, 1, 10))
        errors.append(abs(Integral(grid)(1 / grid.coords) - np.log(10)))
    assert errors[1] < 1e-4
    assert errors[0] / errors[1] > 2 ** 3.5   # fourth order


def test_integral_complex():
    grid = Grid(periodic(16))
    assert Integral(grid)(np.exp(1j * grid.coords) + 1) == pytest.approx(2 * np.pi)


# ── interpolation ─────────────────────────────────────────────────────

def test_spectral_interpolation_accuracy():
    grid = Grid(chebyshev(25, 0, 2), periodic(24), equidistant(30, 0, 1), logarithmic(40, 1, 3))
    f = grid.evaluate(lambda x, y, z, w: np.exp(x) * np.cos(y) * z ** 2 * np.log(w))
    inter = Interpolator(grid, f, method="spectral")
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 2, 50), rng.uniform(0, 2 * np.pi, 50),
                           rng.uniform(0, 1, 50), rng.uniform(1, 3, 50)])
    exact = np.exp(pts[:, 0]) * np.cos(pts[:, 1]) * pts[:, 2] ** 2 * np.log(pts[:, 3])
    npt.assert_allclose(inter(pts), exact, atol=1e-7)


def test_spectral_interpolation_onto_grid():
    grid = Grid(chebyshev(20, 0, 1), periodic(16))
    target = Grid(equidistant(7, 0, 1), periodic(5))
    f = grid.evaluate(lambda x, y: np.sin(3 * x) * np.cos(2 * y))
    expected = target.evaluate(lambda x, y: np.sin(3 * x) * np.cos(2 * y))
    npt.assert_allclose(Interpolator(grid, f, method="spectral")(target), expected, atol=1e-10)
    npt.assert_allclose(Interpolator(grid, f, method="cubic")(target), expected, atol=1e-2)


def test_spectral_interpolation_hits_nodes_exactly():
    grid = Grid(chebyshev(10, 0, 1))
    f = np.cos(grid.coords)
    npt.assert_allclose(Interpolator(grid, f, method="spectral")(grid.coords), f, atol=1e-14)


def test_interpolation_out_of_range():
    grid = Grid(chebyshev(10, 0, 1))
    with pytest.raises(ValueError):
        Interpolator(grid, grid.coords, method="spectral")(1.5)


def test_interpolation_bad_input():
    grid = Grid(chebyshev(10, 0, 1), chebyshev(10, 0, 1))
    inter = Interpolator(grid, np.zeros(grid.shape))
    with pytest.raises(ValueError):
        inter((0.1, 0.2, 0.3))
    with pytest.raises(ValueError):
        inter(0.5)
    with pytest.raises(ValueError):
        Interpolator(grid, np.zeros(grid.shape), method="quadratic")
    with pytest.raises(ValueError):
        Interpolator(grid, np.zeros(grid.shape), method="nope")


def test_interpolation_zip_and_generators():
    grid = Grid(chebyshev(15, 0, 2), chebyshev(15, 0, 1))
    X, Y = grid.meshed_coords
    inter = Interpolator(grid, X + Y, method="spectral")
    t = np.linspace(0, 1, 5)
    npt.assert_allclose(inter(zip(2 * t, t ** 2)), 2 * t + t ** 2, atol=1e-12)
    npt.assert_allclose(inter((p, p / 2) for p in t), 1.5 * t, atol=1e-12)


# ── boundary faces & specs ────────────────────────────────────────────

def test_face_restrict_and_coords():
    grid = Grid(x=chebyshev(6, 0, 1), y=equidistant(4, 0, 1))
    face = grid.faces["x_high"]
    X, Y = grid.meshed_coords
    npt.assert_array_equal(face.restrict(Y), grid.coords[1])
    assert face.shape == (4,)
    assert face.key == "0_high"
    npt.assert_array_equal(face.coords[0], np.ones(4))


def test_face_normal_derivative_and_flux():
    grid = Grid(x=chebyshev(12, 0, 1), y=chebyshev(12, 0, 2))
    X, Y = grid.meshed_coords
    f = X ** 2 * Y
    npt.assert_allclose(grid.faces["x_high"].normal_derivative(f), 2 * grid.coords[1], atol=1e-10)
    npt.assert_allclose(grid.faces["x_low"].normal_derivative(f), 0, atol=1e-10)
    # Gauss theorem: total outward flux of grad f equals the volume integral of the Laplacian.
    g = grid.gradient(f)
    total_flux = sum(face.flux(*g) for face in grid.faces.values())
    assert total_flux == pytest.approx(grid.integrate(grid.laplacian(f)))


def test_face_integrate_curvilinear_surface():
    grid = SphericalGrid(chebyshev(6, 0.5, 2), chebyshev(16, 0, np.pi), periodic(16))
    sphere = grid.faces["r_high"]
    assert sphere.integrate(np.ones(grid.shape)) == pytest.approx(4 * np.pi * 4)
    assert sphere.integrate(np.ones(sphere.shape)) == pytest.approx(4 * np.pi * 4)
    # Physical normal derivative of r^2 is 2r.
    R = grid.meshed_coords[0]
    npt.assert_allclose(sphere.normal_derivative(R ** 2), 4, atol=1e-10)
    with pytest.raises(ValueError):
        sphere.integrate(np.ones(3))


def test_face_integrate_1d():
    grid = Grid(chebyshev(5, 0, 1))
    assert grid.faces["0_high"].integrate(grid.coords + 1) == pytest.approx(2)


def test_resolve_bcs_forms():
    grid = Grid(x=chebyshev(6, 0, 1), y=chebyshev(6, 0, 1))
    assert resolve_bcs(grid, None) == []
    assert len(resolve_bcs(grid, 0.0)) == 4
    assert len(resolve_bcs(grid, Neumann(0))) == 4
    bcs = resolve_bcs(grid, {"x_low": Robin(1, 2, 3), grid.faces["y_high"]: Dirichlet(1)})
    assert [type(b).__name__ for b in bcs] == ["RobinBC", "DirichletBC"]
    assert "Robin" in repr(bcs[0])
    with pytest.raises(TypeError):
        resolve_bcs(grid, {"x_low": bcs[1]})
    with pytest.raises(TypeError):
        resolve_bcs(grid, [Dirichlet(0)])
    with pytest.raises(KeyError):
        resolve_bcs(grid, {"z_low": 0})


def test_apply_bcs_compact_form():
    import scipy.sparse

    from numgrids import apply_bcs
    grid = Grid(x=chebyshev(6, 0, 1))
    L = scipy.sparse.identity(grid.size) * 2
    L2, rhs = apply_bcs(0.0, L, np.ones(grid.size), grid=grid)
    assert rhs[0] == 0 and rhs[-1] == 0 and rhs[1] == 1
    with pytest.raises(ValueError):
        apply_bcs(0.0, L, np.ones(grid.size))


def test_vectorized_bcs_match_rowwise_definition():
    from numgrids import Diff, apply_bcs
    grid = Grid(chebyshev(7, 0, 1), equidistant(9, 0, 1))
    L = Diff(grid, 2, 0).as_matrix() + Diff(grid, 2, 1).as_matrix()
    bcs = resolve_bcs(grid, {"0_low": 1.0, "0_high": Neumann(2.0), "1_low": Robin(1, 2, 3)})
    L_bc, rhs = apply_bcs(bcs, L, np.zeros(grid.size))
    expected = L.toarray()
    D0 = Diff(grid, 1, 0).as_matrix().toarray()
    D1 = Diff(grid, 1, 1).as_matrix().toarray()
    for i in grid.faces["0_low"].flat_indices:
        expected[i] = np.eye(grid.size)[i]
    for i in grid.faces["0_high"].flat_indices:
        expected[i] = D0[i]
    for i in grid.faces["1_low"].flat_indices:
        expected[i] = np.eye(grid.size)[i] - 2 * D1[i]
    npt.assert_allclose(L_bc.toarray(), expected, atol=1e-12)


# ── time evolution ────────────────────────────────────────────────────

def test_evolve_heat_equation_callable():
    grid = Grid(x=chebyshev(20, 0, 1))
    x = grid.coords
    res = evolve(grid, lambda t, u: grid.laplacian(u), np.sin(np.pi * x), (0, 0.1),
                 bcs=0, method="BDF", t_eval=[0, 0.05, 0.1], rtol=1e-9, atol=1e-12)
    assert res.success
    assert res.u.shape == (3, 20)
    npt.assert_allclose(res.final, np.exp(-np.pi ** 2 * 0.1) * np.sin(np.pi * x), atol=1e-6)


def test_evolve_nonhomogeneous_dirichlet_steady_state():
    grid = Grid(x=chebyshev(16, 0, 1))
    res = evolve(grid, grid.laplacian_operator(), np.zeros(16), (0, 5),
                 bcs={"x_low": 1, "x_high": 3}, method="Radau", rtol=1e-8, atol=1e-10)
    npt.assert_allclose(res.final, 1 + 2 * grid.coords, atol=1e-6)


def test_evolve_wave_system():
    grid = Grid(x=chebyshev(24, 0, 1))
    x = grid.coords
    u0 = np.sin(np.pi * x)
    res = evolve(grid, lambda t, u, v: (v, grid.laplacian(u)), (u0, np.zeros_like(u0)),
                 (0, 1.0), bcs=0, t_eval=[1.0], rtol=1e-10, atol=1e-12)
    u, v = res.final
    npt.assert_allclose(u, np.cos(np.pi * 1.0) * u0, atol=1e-6)
    assert isinstance(res.u, tuple) and res.u[0].shape == (1, 24)


def test_evolve_schroedinger_complex():
    grid = Grid(x=periodic(32))
    x = grid.coords
    psi0 = np.exp(2j * x)
    res = evolve(grid, lambda t, psi: 0.5j * grid.diff(psi, 2), psi0, (0, 1.0),
                 t_eval=[1.0], rtol=1e-10, atol=1e-12)
    npt.assert_allclose(res.final, np.exp(-2j) * psi0, atol=1e-6)


def test_evolve_errors():
    grid = Grid(x=chebyshev(8, 0, 1))
    u0 = np.zeros(8)
    with pytest.raises(ValueError, match="singular boundary system"):
        evolve(grid, lambda t, u: u, u0, (0, 1), bcs=Robin(0, 0))
    with pytest.raises(ValueError):
        evolve(grid, lambda t, u: u[:3], u0, (0, 1))
    with pytest.raises(ValueError):
        evolve(grid, lambda t, u, v: (u,), (u0, u0), (0, 1))
    with pytest.raises(TypeError):
        evolve(grid, grid.laplacian_operator(), (u0, u0), (0, 1))


# ── convergence ───────────────────────────────────────────────────────

def test_convergence_algebraic_order():
    res = convergence_study(
        lambda n: Grid(equidistant(n, 0, 1)),
        lambda g: g.diff(np.sin(g.coords), acc=4),
        ns=[10, 20, 40, 80],
        exact=lambda g: np.cos(g.coords),
    )
    assert res.order == pytest.approx(4, abs=0.5)
    assert len(res.orders) == 3


def test_convergence_spectral_rate_and_reference():
    res = convergence_study(
        lambda n: Grid(chebyshev(n, 0, 1)),
        lambda g: g.integrate(np.exp(g.coords)),
        ns=[4, 6, 8, 10],
        exact=np.e - 1,
    )
    assert res.rate > 2
    res2 = convergence_study(
        lambda n: Grid(chebyshev(n, 0, 1)),
        lambda g: g.diff(np.exp(3 * g.coords)),
        ns=[6, 8, 10, 40],
    )
    assert len(res2.ns) == 3
    assert np.all(np.diff(res2.errors) < 0)


def test_too_few_points_for_stencil():
    with pytest.raises(ValueError, match="Too few grid points"):
        Grid(equidistant(5)).diff(np.zeros(5), 2)


def test_convergence_needs_two():
    with pytest.raises(ValueError):
        convergence_study(lambda n: Grid(chebyshev(n)), lambda g: 0.0, ns=[5])


def test_convergence_plot():
    res = convergence_study(lambda n: Grid(chebyshev(n, 0, 1)),
                            lambda g: g.integrate(np.exp(g.coords)), ns=[4, 6, 8], exact=np.e - 1)
    assert res.plot() is not None


# ── xarray ────────────────────────────────────────────────────────────

xr = pytest.importorskip("xarray")


def test_xarray_roundtrip():
    grid = SphericalGrid(chebyshev(6, 0.5, 1), chebyshev(5, 0.2, 2.9), periodic(8))
    R, T, P = grid.meshed_coords
    ds = grid.to_xarray(u=R * np.cos(T))
    assert set(ds.dims) == {"r", "theta", "phi"}
    loaded, data = Grid.from_xarray(ds)
    assert loaded == grid
    npt.assert_array_equal(data["u"], R * np.cos(T))
    # DataArray input and transposed variables
    _, data2 = Grid.from_xarray(ds["u"].transpose("phi", "r", "theta"))
    npt.assert_array_equal(data2["u"], R * np.cos(T))


def test_xarray_netcdf_roundtrip(tmp_path):
    pytest.importorskip("netCDF4")
    grid = Grid(x=chebyshev(6, 0, 1), y=periodic(8))
    X, Y = grid.meshed_coords
    grid.to_xarray(u=X * Y).to_netcdf(tmp_path / "d.nc")
    with xr.open_dataset(tmp_path / "d.nc") as ds:
        loaded, data = Grid.from_xarray(ds)
    assert loaded == grid
    npt.assert_allclose(data["u"], X * Y)


def test_xarray_foreign_dataset_inference():
    x = chebyshev(8, 0, 1).coords
    y = np.linspace(0, 2, 5)
    z = np.logspace(0, 2, 4)
    ds = xr.Dataset({"f": (("x", "y", "z"), np.ones((8, 5, 4)))}, coords={"x": x, "y": y, "z": z})
    grid, data = Grid.from_xarray(ds)
    assert [type(a).__name__ for a in grid.axes] == ["ChebyshevAxis", "EquidistantAxis", "LogAxis"]
    assert data["f"].shape == (8, 5, 4)
    bad = xr.Dataset({"f": (("x",), np.ones(3))}, coords={"x": [0.0, 0.1, 0.7]})
    with pytest.raises(ValueError):
        Grid.from_xarray(bad)


def test_to_xarray_shape_check():
    grid = Grid(chebyshev(6, 0, 1))
    with pytest.raises(ValueError):
        grid.to_xarray(u=np.ones(3))


# ── plotting ──────────────────────────────────────────────────────────

def test_plot_field_1d_2d_3d():
    import matplotlib.pyplot as plt
    g1 = Grid(x=chebyshev(10, 0, 1))
    assert g1.plot(g1.coords ** 2) is not None
    g2 = PolarGrid(chebyshev(8, 0.1, 1), periodic(12))
    ax = g2.plot(g2.meshed_coords[0])
    assert ax.get_xlabel() == "$x$"
    g2.plot(g2.meshed_coords[0], cartesian=False, colorbar=False)
    g3 = Grid(x=chebyshev(5, 0, 1), y=periodic(6), z=chebyshev(4, 0, 1))
    ax = g3.plot(g3.meshed_coords[0])
    assert ax.get_xlabel() == "$x$"
    g3.plot(g3.meshed_coords[0], fixed={"x": 0})
    with pytest.raises(ValueError):
        g3.plot(g3.meshed_coords[0], fixed={})
    plt.close("all")


def test_plotter_uses_coordinate_axis():
    import matplotlib.pyplot as plt

    from numgrids.plots import Plotter
    grid = Grid(x=chebyshev(10, 0, 1), y=chebyshev(10, 0, 1))
    p = Plotter(grid)
    t = np.linspace(0.1, 0.9, 7)
    p.plot(grid.meshed_coords[0], t, 0.5)
    line = p.ax.get_lines()[0]
    npt.assert_allclose(line.get_xdata(), t)
    with pytest.raises(ValueError):
        p.plot(grid.meshed_coords[0], 0.1, 0.2)
    plt.close("all")


def test_plot_surface_grid_embedded_in_3d_falls_back_to_coordinates():
    import matplotlib.pyplot as plt
    sphere = CurvilinearGrid(
        chebyshev(8, 0.1, 3.0, name="theta"), periodic(10, name="phi"),
        scale_factors=(lambda c: np.ones_like(c[0]), lambda c: np.sin(c[0])),
        cartesian=lambda c: (np.sin(c[0]) * np.cos(c[1]), np.sin(c[0]) * np.sin(c[1]), np.cos(c[0])),
    )
    ax = sphere.plot(sphere.meshed_coords[0])
    assert ax.get_xlabel() == "$theta$"
    plt.close("all")

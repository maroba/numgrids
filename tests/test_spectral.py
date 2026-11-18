"""Spectral coefficients, resolution diagnostics and filtering."""

import numpy as np
import pytest
from numpy.polynomial import chebyshev as C
from scipy.special import iv

from numgrids import (
    Grid,
    Interpolator,
    PolarGrid,
    SphericalGrid,
    SphericalSurfaceGrid,
    chebyshev,
    colatitude,
    convergence_study,
    equidistant,
    infinite,
    logarithmic,
    periodic,
    radial,
    semi_infinite,
)
from numgrids.spectral import (
    AxisResolution,
    Resolution,
    coefficients,
    from_coefficients,
    is_resolved,
    mode_numbers,
    resample,
    resolution,
    spectral_filter,
)

# ---------------------------------------------------------------------------
# Known coefficients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 8, 9, 33])
def test_single_chebyshev_polynomials(n):
    grid = Grid(chebyshev(n, -1, 1))
    for k in range(n):
        expected = np.zeros(n)
        expected[k] = 1.0
        assert np.allclose(grid.coefficients(C.chebval(grid.coords, expected)), expected, atol=1e-13)


def test_chebyshev_coefficients_refer_to_the_mapped_variable():
    grid = Grid(chebyshev(9, 2, 5))
    xi = (2 * grid.coords - 7) / 3
    c = grid.coefficients(4 * xi ** 3 - 3 * xi + 0.25)              # T_3 + T_0 / 4
    assert np.allclose(c, [0.25, 0, 0, 1, 0, 0, 0, 0, 0], atol=1e-13)


def test_chebyshev_coefficients_of_exp_are_bessel_functions():
    grid = Grid(chebyshev(24, -1, 1))
    k = np.arange(24)
    expected = 2 * iv(k, 1.0)
    expected[0] /= 2
    assert np.allclose(grid.coefficients(np.exp(grid.coords)), expected, atol=1e-14)


def test_chebyshev_coefficients_are_those_of_the_interpolant():
    grid = Grid(chebyshev(15, 0, 3))
    f = np.random.default_rng(0).standard_normal(15)
    x = np.linspace(0, 3, 40)
    series = C.chebval(2 * x / 3 - 1, grid.coefficients(f))
    assert np.allclose(series, grid.interpolate(f, x, method="spectral"), atol=1e-12)


def test_single_fourier_modes():
    grid = Grid(periodic(16))
    x = grid.coords
    k = mode_numbers(grid)
    assert np.array_equal(k, [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1])

    c = grid.coefficients(2.5 + 3 * np.cos(3 * x) + np.sin(5 * x))
    expected = np.zeros(16, dtype=complex)
    expected[k == 0] = 2.5                       # the mean value
    expected[np.abs(k) == 3] = 1.5               # A / 2 at +k and -k
    expected[k == 5], expected[k == -5] = -0.5j, 0.5j
    assert np.allclose(c, expected, atol=1e-14)

    c = grid.coefficients(np.exp(2j * x))        # a complex mode has a single coefficient
    assert np.allclose(c, k == 2, atol=1e-14)


@pytest.mark.parametrize("n", [12, 13])
def test_fourier_phases_refer_to_the_coordinate(n):
    # The same function on shifted grids has the same coefficients.
    P = 3.0
    kappa = 2 * np.pi / P
    func = lambda x: 1 + np.cos(2 * kappa * x) + 2 * np.sin(kappa * x)
    for low in (0.0, -1.5, 0.7):
        grid = Grid(periodic(n, low, low + P))
        c = grid.coefficients(func(grid.coords))
        k = mode_numbers(grid)
        expected = 1.0 * (k == 0) + 0.5 * (np.abs(k) == 2) - 1j * (k == 1) + 1j * (k == -1)
        assert np.allclose(c, expected, atol=1e-13)


def test_nyquist_coefficient():
    grid = Grid(periodic(8, 1.0, 1.0 + 2 * np.pi))
    c = grid.coefficients(np.cos(4 * (grid.coords - 1.0)))         # +1, -1, +1, ... on the grid
    assert np.allclose(c, np.arange(8) == 4, atol=1e-14)


def test_fourier_series_is_the_trigonometric_interpolant():
    rng = np.random.default_rng(1)
    x = np.linspace(-1, 2, 50)
    for n in (9, 10):
        grid = Grid(periodic(n, -1, 2))
        f = rng.standard_normal(n)
        c, k = grid.coefficients(f), mode_numbers(grid)
        modes = np.exp(2j * np.pi * np.outer(x, k) / 3)
        if n % 2 == 0:
            modes[:, n // 2] = np.cos(2 * np.pi * (n // 2) * (x + 1) / 3)
        series = modes @ c
        assert np.allclose(series.imag, 0, atol=1e-13)
        assert np.allclose(series.real, grid.interpolate(f, x, method="spectral"), atol=1e-12)


def test_coefficients_on_unbounded_axes():
    # Polynomials in the computational variable xi have finite Chebyshev series.
    grid = Grid(semi_infinite(12, scale=2.0, low=1.0))
    x = grid.coords
    c = grid.coefficients(2 / (x + 1))                              # (1 - xi) / 2
    assert np.allclose(c, [0.5, -0.5] + [0] * 10, atol=1e-13)

    grid = Grid(infinite(11, scale=2.0, center=1.0))
    x = grid.coords
    c = grid.coefficients(4 / (4 + (x - 1) ** 2))                   # 1 - xi^2 = (T_0 - T_2) / 2
    assert np.allclose(c, [0.5, 0, -0.5] + [0] * 8, atol=1e-13)
    c = grid.coefficients((x - 1) / np.sqrt(4 + (x - 1) ** 2))      # xi = T_1
    assert np.allclose(c, [0, 1] + [0] * 9, atol=1e-13)


@pytest.mark.parametrize("axis", [semi_infinite(14, scale=3.0), infinite(15, scale=2.0)])
def test_unbounded_coefficients_are_those_of_the_interpolant(axis):
    grid = Grid(axis)
    f = np.exp(-0.3 * grid.coords ** 2) * (1 + grid.coords)
    x = np.linspace(grid.coords[1], grid.coords[-2], 30)
    series = C.chebval(axis.to_internal(x), grid.coefficients(f))
    assert np.allclose(series, grid.interpolate(f, x, method="spectral"), atol=1e-12)


def test_mode_numbers():
    grid = Grid(x=chebyshev(5), y=periodic(7), r=semi_infinite(4))
    assert np.array_equal(mode_numbers(grid, "x"), [0, 1, 2, 3, 4])
    assert np.array_equal(mode_numbers(grid, 1), [0, 1, 2, 3, -3, -2, -1])
    assert np.array_equal(mode_numbers(grid, "r"), [0, 1, 2, 3])
    polar = PolarGrid(radial(4), periodic(8))
    assert np.array_equal(mode_numbers(polar, "r"), np.arange(8))


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

AXES = {
    "chebyshev": lambda: chebyshev(17, -2, 3),
    "periodic even": lambda: periodic(16, -1, 4),
    "periodic odd": lambda: periodic(15),
    "semi-infinite": lambda: semi_infinite(18, scale=2.0, low=-1.0),
    "infinite even": lambda: infinite(16, scale=1.5),
    "infinite odd": lambda: infinite(17, scale=1.5, center=2.0),
}


@pytest.mark.parametrize("name", AXES)
@pytest.mark.parametrize("complex_field", [False, True])
def test_round_trip_1d(name, complex_field):
    grid = Grid(AXES[name]())
    rng = np.random.default_rng(2)
    f = rng.standard_normal(grid.shape)
    if complex_field:
        f = f + 1j * rng.standard_normal(grid.shape)
    c = grid.coefficients(f)
    assert c.shape == grid.shape
    back = grid.from_coefficients(c)
    assert np.allclose(back, f, atol=1e-12)
    if not complex_field and not grid.axes[0].periodic:
        assert c.dtype == float and back.dtype == float            # real series stay real


def test_round_trip_mixed_2d_and_3d():
    rng = np.random.default_rng(3)
    grid2 = Grid(x=chebyshev(9, 0, 1), y=periodic(8))
    grid3 = Grid(x=chebyshev(7, 0, 1), y=periodic(6, -1, 1), z=infinite(8), )
    for grid in (grid2, grid3):
        f = rng.standard_normal(grid.shape) + 1j * rng.standard_normal(grid.shape)
        for axis in (None, 0, "y", (0, 1), ["y", "x"]):
            c = grid.coefficients(f, axis)
            assert c.shape == grid.shape
            assert np.allclose(grid.from_coefficients(c, axis), f, atol=1e-12)


def test_tensor_product_coefficients():
    grid = Grid(x=chebyshev(10, -1, 1), y=periodic(12))
    x, y = grid.coords
    X, Y = grid.meshed_coords
    c = grid.coefficients(X ** 2 * np.cos(3 * Y))                   # (T_0 + T_2) / 2 times cos(3 y)
    expected = np.zeros(grid.shape, dtype=complex)
    expected[0, 3] = expected[0, -3] = expected[2, 3] = expected[2, -3] = 0.25
    assert np.allclose(c, expected, atol=1e-14)

    # The transforms along different axes commute.
    f = np.random.default_rng(4).standard_normal(grid.shape)
    one_by_one = grid.coefficients(grid.coefficients(f, "x").real, "y") \
        + 1j * grid.coefficients(grid.coefficients(f, "x").imag, "y")
    assert np.allclose(grid.coefficients(f), one_by_one, atol=1e-13)
    assert np.allclose(grid.coefficients(f), coefficients(grid, coefficients(grid, f, "y"), "x"), atol=1e-13)


def test_single_axis_of_a_grid_with_finite_difference_axes():
    grid = Grid(x=chebyshev(8, -1, 1), y=equidistant(5, 0, 1))
    X, Y = grid.meshed_coords
    c = grid.coefficients(X * Y, axis="x")
    assert np.allclose(c[1], grid.coords[1], atol=1e-14)            # T_1 times y
    assert np.allclose(np.delete(c, 1, axis=0), 0, atol=1e-14)
    assert np.allclose(grid.from_coefficients(c, axis="x"), X * Y, atol=1e-14)


def test_integer_fields_are_accepted():
    grid = Grid(chebyshev(4, -1, 1))
    assert np.allclose(grid.coefficients(np.ones(4, dtype=int)), [1, 0, 0, 0], atol=1e-15)
    assert np.allclose(grid.from_coefficients(np.array([1, 0, 0, 0])), 1.0)


def test_functions_and_grid_methods_agree():
    grid = Grid(chebyshev(9))
    f = np.exp(grid.coords)
    assert np.array_equal(coefficients(grid, f), grid.coefficients(f))
    assert np.array_equal(from_coefficients(grid, f), grid.from_coefficients(f))
    assert np.array_equal(resample(grid, f, 12), grid.resample(f, 12))
    assert np.array_equal(spectral_filter(grid, f, keep=4), grid.filter(f, keep=4))
    assert is_resolved(grid, f, 1e-6) is grid.is_resolved(f, 1e-6)
    assert str(resolution(grid, f)) == str(grid.resolution(f))


# ---------------------------------------------------------------------------
# Axes through the origin and the poles
# ---------------------------------------------------------------------------

def _polar_field(r, phi):
    x, y = r * np.cos(phi), r * np.sin(phi)
    return np.exp(x) * np.cos(2 * y - 0.3)


def _ball_field(r, theta, phi):
    x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)
    return np.exp(x + 0.5 * z) * np.cos(y)


def test_polar_grid_coefficients_on_the_diameter():
    grid = PolarGrid(radial(8, 2.0), periodic(12))
    R, Phi = grid.meshed_coords
    X, Y = R * np.cos(Phi), R * np.sin(Phi)
    c = grid.coefficients(X ** 2 + Y, axis="r")
    assert c.shape == (16, 12)                                      # the whole diameter
    # Along the diameter s = 2 xi at angle phi: 4 xi^2 cos(phi)^2 + 2 xi sin(phi).
    phi = grid.coords[1]
    expected = np.zeros((16, 12))
    expected[0] = expected[2] = 2 * np.cos(phi) ** 2
    expected[1] = 2 * np.sin(phi)
    assert np.allclose(c, expected, atol=1e-13)
    assert np.allclose(grid.from_coefficients(c, axis="r"), X ** 2 + Y, atol=1e-13)


def test_polar_grid_round_trip_and_parity_rule():
    grid = PolarGrid(radial(8, 1.5), periodic(12))
    f = grid.evaluate(_polar_field)
    c = grid.coefficients(f)
    assert c.shape == (16, 12)
    assert np.allclose(grid.from_coefficients(c).real, f, atol=1e-12)
    # A smooth field only has modes T_k(r) exp(i m phi) with k + m even.
    k, m = np.meshgrid(mode_numbers(grid, 0), mode_numbers(grid, 1), indexing="ij")
    odd = ((k + m) % 2 == 1) & (np.abs(m) < 6)
    assert np.abs(c[odd]).max() < 1e-13
    assert np.abs(c[~odd]).max() > 0.1


def test_spherical_grid_round_trips():
    grid = SphericalGrid(radial(5), colatitude(6), periodic(8))
    f = grid.evaluate(_ball_field)
    shapes = {None: (10, 12, 8), "r": (10, 6, 8), "theta": (5, 12, 8), ("r", "theta"): (10, 12, 8), 2: (5, 6, 8)}
    for axis, shape in shapes.items():
        c = grid.coefficients(f, axis)
        assert c.shape == shape
        assert np.allclose(grid.from_coefficients(c, axis), f, atol=1e-12)


def test_colatitude_coefficients_on_the_great_circle():
    grid = SphericalSurfaceGrid(colatitude(6), periodic(8))
    Theta, Phi = grid.meshed_coords
    c = grid.coefficients(np.cos(Theta) + np.sin(Theta) * np.cos(Phi), axis=0)
    k = mode_numbers(grid, 0)
    assert c.shape == (12, 8) and len(k) == 12
    # cos(theta) is the same on every meridian; sin(theta) cos(phi) changes with it.
    assert np.allclose(c[np.abs(k) == 1].real, 0.5, atol=1e-13)
    assert np.allclose(c[k == 1].imag.ravel(), -0.5 * np.cos(grid.coords[1]), atol=1e-13)
    assert np.allclose(c[np.abs(k) != 1], 0, atol=1e-13)


def test_from_coefficients_checks_the_shape():
    grid = PolarGrid(radial(4), periodic(8))
    with pytest.raises(ValueError, match="twice as many"):
        grid.from_coefficients(np.zeros((4, 8)))
    with pytest.raises(ValueError, match="shape"):
        Grid(chebyshev(5)).from_coefficients(np.zeros(6))


# ---------------------------------------------------------------------------
# Finite-difference axes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [equidistant(10, 0, 1), logarithmic(10, 1, 10)])
def test_finite_difference_axes_have_no_coefficients(axis):
    grid = Grid(axis)
    f = np.ones(10)
    for call in (lambda: grid.coefficients(f), lambda: grid.from_coefficients(f),
                 lambda: mode_numbers(grid), lambda: grid.filter(f, axis=0), lambda: grid.resample(f, 20)):
        with pytest.raises(ValueError, match="finite differences"):
            call()
    with pytest.raises(ValueError, match="no spectral axis"):
        grid.filter(f)
    with pytest.raises(ValueError, match="no spectral axis"):
        grid.resolution(f)
    with pytest.raises(ValueError, match="no spectral axis"):
        grid.is_resolved(f)


def test_all_axes_are_transformed_by_default():
    grid = Grid(x=chebyshev(6), y=equidistant(7))
    with pytest.raises(ValueError, match="'y'.*finite differences"):
        grid.coefficients(np.ones(grid.shape))
    with pytest.raises(ValueError, match="repeated"):
        grid.coefficients(np.ones(grid.shape), axis=("x", 0))
    with pytest.raises(ValueError, match="shape"):
        grid.coefficients(np.ones(6))


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", [(8, 20), (9, 20), (8, 21), (9, 21), (8, 16)])
def test_upsampling_is_trigonometric_interpolation(n, m):
    grid, fine = Grid(periodic(n, 1, 4)), Grid(periodic(m, 1, 4))
    f = np.random.default_rng(5).standard_normal(n)
    result = grid.resample(f, fine)
    assert result.dtype == float
    assert np.allclose(result, Interpolator(grid, f, method="spectral")(fine), atol=1e-13)


@pytest.mark.parametrize("make", [
    lambda n: Grid(chebyshev(n, -1, 2)),
    lambda n: Grid(semi_infinite(n, scale=3.0)),
    lambda n: Grid(infinite(n, scale=3.0)),
])
def test_upsampling_is_polynomial_interpolation(make):
    grid, fine = make(12), make(29)
    f = np.random.default_rng(6).standard_normal(12)
    assert np.allclose(grid.resample(f, fine), Interpolator(grid, f, method="spectral")(fine), atol=1e-11)
    assert np.allclose(fine.resample(grid.resample(f, 29), 12), f, atol=1e-11)     # nothing is lost


def test_resampling_resolved_functions_is_exact():
    cases = [
        (lambda n: Grid(chebyshev(n, -1, 2)), lambda x: np.exp(x) * np.cos(3 * x), 40, 28),
        (lambda n: Grid(periodic(n)), lambda x: np.exp(np.sin(x)), 40, 31),
        (lambda n: Grid(periodic(n)), lambda x: np.exp(np.sin(x)), 41, 30),
        (lambda n: Grid(semi_infinite(n, scale=1.0)), lambda x: 1 / (1 + x) ** 2, 30, 12),
        (lambda n: Grid(infinite(n, scale=1.0)), lambda x: 1 / (1 + x ** 2) ** 2, 30, 14),
    ]
    for make, func, n, m in cases:
        grid, coarse = make(n), make(m)
        assert np.allclose(grid.resample(func(grid.coords), coarse), func(coarse.coords), atol=1e-10)
        assert np.allclose(coarse.resample(func(coarse.coords), grid), func(grid.coords), atol=1e-10)


def test_resample_several_axes_and_keep_finite_difference_axes():
    grid = Grid(x=chebyshev(14, 0, 1), y=periodic(12), z=equidistant(5, 0, 1))
    func = lambda x, y, z: np.exp(x) * np.sin(y) * z ** 2
    target = Grid(x=chebyshev(20, 0, 1), y=periodic(18), z=equidistant(5, 0, 1))
    f = grid.evaluate(func)
    for result in (grid.resample(f, target), grid.resample(f, (20, 18, 5))):
        assert result.shape == (20, 18, 5)
        assert np.allclose(result, target.evaluate(func), atol=1e-12)
    same = grid.resample(f, grid.shape)
    assert np.array_equal(same, f) and same is not f


def test_resample_complex_fields_and_1d_shorthand():
    grid = Grid(chebyshev(16, -1, 1))
    f = np.exp(1j * grid.coords)
    fine = grid.resample(f, 30)
    assert fine.dtype == complex
    assert np.allclose(fine, np.exp(1j * Grid(chebyshev(30, -1, 1)).coords), atol=1e-13)


def test_resample_curvilinear_grids():
    grid = PolarGrid(radial(10, 1.5), periodic(16))
    fine = PolarGrid(radial(15, 1.5), periodic(24))
    f = grid.evaluate(_polar_field)
    assert np.allclose(grid.resample(f, fine), Interpolator(grid, f, method="spectral")(fine), atol=1e-12)
    resolved = PolarGrid(radial(16, 1.5), periodic(40))
    assert np.allclose(resolved.resample(resolved.evaluate(_polar_field), (20, 48)),
                       PolarGrid(radial(20, 1.5), periodic(48)).evaluate(_polar_field), atol=1e-9)

    ball = SphericalGrid(radial(6), colatitude(8), periodic(8))
    finer = SphericalGrid(radial(9), colatitude(13), periodic(12))
    f = ball.evaluate(_ball_field)
    assert np.allclose(ball.resample(f, finer), Interpolator(ball, f, method="spectral")(finer), atol=1e-12)


def test_resample_rejects_other_domains_and_bad_shapes():
    grid = Grid(chebyshev(8, 0, 1), periodic(8))
    f = np.zeros(grid.shape)
    with pytest.raises(ValueError, match="differs in more than the number of points"):
        grid.resample(f, Grid(chebyshev(12, 0, 2), periodic(8)))
    with pytest.raises(ValueError, match="dimensions"):
        grid.resample(f, Grid(chebyshev(12, 0, 1)))
    with pytest.raises(ValueError, match="numbers of points"):
        grid.resample(f, 12)
    with pytest.raises(ValueError):
        PolarGrid(radial(8), periodic(8)).resample(np.zeros((8, 8)), (8, 9))      # odd azimuthal count


# ---------------------------------------------------------------------------
# Resolution diagnostics
# ---------------------------------------------------------------------------

RUNGE = lambda x: 1 / (1 + 25 * x ** 2)


def test_smooth_function_is_resolved():
    grid = Grid(x=chebyshev(40, -1, 1))
    result = grid.resolution(np.exp(grid.coords))
    assert isinstance(result, Resolution) and isinstance(result["x"], AxisResolution)
    assert result.resolved and result["x"].resolved and result[0] is result["x"]
    assert 8 <= result["x"].needed <= 14
    assert result["x"].tail < 1e-13
    assert result["x"].kind == "chebyshev" and result["x"].num_points == 40
    assert result.needed == (result["x"].needed,)
    assert result.suggested_shape == (result["x"].needed,)
    assert grid.is_resolved(np.exp(grid.coords))
    assert str(result) == repr(result) == result.summary()
    assert str(result).splitlines() == [
        "Resolution at tol=1e-08: resolved",
        f"  axis x: resolved, {result['x'].needed} of 40 points needed",
    ]


def test_polynomials_need_their_degree():
    grid = Grid(chebyshev(30, -1, 1))
    for degree in (0, 1, 3, 10):
        res = grid.resolution(grid.coords ** degree)[0]
        assert res.resolved
        assert res.needed == degree + 3                 # degree + 1 modes and two negligible ones
    assert grid.resolution(np.zeros(30))[0].needed == 2


def test_steep_function_is_not_resolved():
    grid = Grid(x=chebyshev(48, -1, 1))
    result = grid.resolution(np.tanh(10 * grid.coords))
    axis = result["x"]
    assert not result.resolved and not axis.resolved
    assert axis.tail > 1e-5
    assert axis.decay == "geometric"
    assert 100 < axis.needed < 200                      # 122 points give 1e-8, see the test below
    assert result.suggested_shape == (axis.needed,)
    assert "NOT resolved" in str(result) and f"about {axis.needed} points needed" in str(result)
    assert not grid.is_resolved(np.tanh(10 * grid.coords))
    assert grid.is_resolved(np.tanh(10 * grid.coords), tol=1e-2)


def test_oscillatory_function_without_decay():
    grid = Grid(chebyshev(30, -1, 1))
    axis = grid.resolution(np.cos(50 * grid.coords))[0]
    assert not axis.resolved and axis.needed is None and axis.decay == "none"
    assert axis.tail > 0.1
    assert "not decaying" in str(axis)
    assert grid.resolution(np.cos(50 * grid.coords)).suggested_shape == (60,)
    fine = Grid(chebyshev(120, -1, 1))
    assert fine.is_resolved(np.cos(50 * fine.coords))


@pytest.mark.parametrize("level, resolved", [(1e-3, False), (1e-6, False), (1e-11, True), (0.0, True)])
def test_noise(level, resolved):
    grid = Grid(chebyshev(60, -1, 1))
    noise = np.random.default_rng(7).standard_normal(60)
    axis = grid.resolution(np.exp(grid.coords) + level * noise)[0]
    assert axis.resolved is resolved
    if resolved:
        assert axis.needed < 16                         # the noise is below the tolerance
    else:
        assert axis.needed is None and axis.decay == "none"        # a plateau: more points do not help
        assert level / 10 < axis.tail < 100 * level


def test_pure_noise_is_not_resolved():
    grid = Grid(periodic(64))
    axis = grid.resolution(np.random.default_rng(8).standard_normal(64))[0]
    assert not axis.resolved and axis.needed is None
    assert axis.kind == "fourier" and len(axis.spectrum) == 33


def test_decay_rate_and_order():
    grid = Grid(chebyshev(200, -1, 1))
    x = grid.coords
    smooth = grid.resolution(RUNGE(x))[0]
    # Poles at +-i/5: the coefficients decay like rho^-k with rho = (1 + sqrt(26)) / 5.
    assert smooth.decay == "geometric"
    assert smooth.rate == pytest.approx(np.log((1 + np.sqrt(26)) / 5), rel=0.05)

    kink = grid.resolution(np.abs(x) ** 3)[0]           # coefficients decay like k^-4
    assert not kink.resolved
    assert kink.decay == "algebraic"
    assert kink.order == pytest.approx(4, abs=0.5)
    assert kink.needed > 400
    assert "algebraic decay" in str(kink)

    jump = grid.resolution(np.sign(x - 0.1))[0]
    assert not jump.resolved and jump.needed is None


def test_spectrum_holds_amplitudes():
    grid = Grid(periodic(16))
    x = grid.coords
    axis = grid.resolution(2 + 3 * np.cos(2 * x + 1) + 0.5 * np.sin(4 * x))[0]
    assert np.allclose(axis.spectrum, [2, 0, 3, 0, 0.5, 0, 0, 0, 0], atol=1e-14)
    assert axis.resolved and axis.needed == 12          # modes 0 to 4, two negligible ones: 7 modes

    grid = Grid(chebyshev(6, -1, 1))
    axis = grid.resolution(C.chebval(grid.coords, [1, -2, 0.5]))[0]
    assert np.allclose(axis.spectrum, [1, 2, 0.5, 0, 0, 0], atol=1e-14)


def test_complex_fields():
    grid = Grid(periodic(32))
    axis = grid.resolution(np.exp(3j * grid.coords))[0]
    assert axis.resolved and axis.needed == 10
    assert np.allclose(axis.spectrum[:5], [0, 0, 0, 1, 0], atol=1e-14)


def test_anisotropic_function():
    grid = Grid(x=chebyshev(40, -1, 1), y=periodic(64))
    func = lambda x, y: np.tanh(10 * x) * np.exp(np.sin(y))
    result = grid.resolution(grid.evaluate(func))
    assert not result.resolved
    assert not result["x"].resolved and result["x"].needed > 80        # under-resolved
    assert result["y"].resolved and result["y"].needed <= 24           # over-resolved
    lines = str(result).splitlines()
    assert lines[0] == "Resolution at tol=1e-08: NOT resolved"
    assert lines[1].startswith("  axis x: NOT resolved, tail at ")
    assert lines[2] == f"  axis y: resolved, {result['y'].needed} of 64 points needed"

    better = result.suggested_grid
    assert type(better) is Grid and better.shape == result.suggested_shape
    assert [ax.name for ax in better.axes] == ["x", "y"]
    assert better.shape[0] > 80 and better.shape[1] <= 24
    assert better.is_resolved(better.evaluate(func))


def test_finite_difference_axes_are_not_assessed():
    grid = Grid(x=chebyshev(20, 0, 1), y=equidistant(30, 0, 1), z=logarithmic(10, 1, 10))
    result = grid.resolution(grid.evaluate(lambda x, y, z: np.exp(x) * y * z))
    assert result.resolved                              # judged by the spectral axis alone
    assert result["y"].resolved is None and result["z"].resolved is None
    assert result["y"].kind == "" and result["y"].spectrum.size == 0
    assert result.needed[1:] == (None, None)
    assert result.suggested_shape[1:] == (30, 10)
    assert "axis y: not assessed" in str(result)


def test_unbounded_axes_are_assessed():
    grid = Grid(r=semi_infinite(40, scale=4.0), x=infinite(50, scale=3.0))
    func = lambda r, x: np.exp(-r) * np.exp(-x ** 2)
    result = grid.resolution(grid.evaluate(func))
    assert result.resolved
    assert 15 < result["r"].needed < 40 and 30 < result["x"].needed <= 50

    coarse = Grid(r=semi_infinite(12, scale=4.0), x=infinite(50, scale=3.0))
    result = coarse.resolution(coarse.evaluate(func))
    assert not result["r"].resolved and result["x"].resolved


def test_resolution_on_curvilinear_grids():
    grid = PolarGrid(radial(16, 1.5), periodic(32))
    result = grid.resolution(grid.evaluate(_polar_field))
    assert result["r"].resolved and result["r"].needed < 16
    assert len(result["r"].spectrum) == 32              # the modes of the whole diameter
    assert not result["phi"].resolved and result["phi"].needed % 2 == 0

    better = result.suggested_grid
    assert type(better) is PolarGrid
    assert better.is_resolved(better.evaluate(_polar_field), tol=1e-7)

    ball = SphericalGrid(radial(10), colatitude(14), periodic(24))
    result = ball.resolution(ball.evaluate(_ball_field), tol=1e-6)
    assert result.resolved
    assert [a.kind for a in result.axes] == ["chebyshev", "fourier", "fourier"]
    assert all(a.needed <= a.num_points for a in result.axes)
    assert type(result.suggested_grid) is SphericalGrid


def test_tolerance_at_round_off_level():
    grid = Grid(chebyshev(40, -1, 1))
    assert grid.is_resolved(np.exp(grid.coords), tol=1e-15)
    with pytest.raises(ValueError, match="tol"):
        grid.resolution(np.exp(grid.coords), tol=0)
    with pytest.raises(ValueError, match="tol"):
        grid.resolution(np.exp(grid.coords), tol=2)
    with pytest.raises(ValueError, match="non-finite"):
        grid.resolution(np.full(40, np.nan))


def _interpolation_error(make_grid, func, n, points):
    """Error of the interpolant on n points, by way of a convergence study."""
    study = convergence_study(
        make_grid,
        lambda grid: grid.interpolate(func(grid.coords), points, method="spectral"),
        ns=[n // 2, n],
        exact=lambda grid: func(points),
    )
    return study.errors / np.abs(func(points)).max()


POINT_ESTIMATES = {
    "runge": (lambda n: Grid(chebyshev(n, -1, 1)), RUNGE, np.linspace(-1, 1, 400)),
    "tanh": (lambda n: Grid(chebyshev(n, -1, 1)), lambda x: np.tanh(10 * x), np.linspace(-1, 1, 400)),
    "periodic": (lambda n: Grid(periodic(n)), lambda x: 1 / (1.2 - np.cos(x)), np.linspace(0, 6.2, 400)),
    "semi-infinite": (lambda n: Grid(semi_infinite(n, scale=4.0)), lambda x: np.exp(-x) * np.cos(2 * x),
                      np.linspace(0, 30, 400)),
    "infinite": (lambda n: Grid(infinite(n, scale=4.0)), lambda x: np.exp(-x ** 2 / 2), np.linspace(-20, 20, 400)),
}


@pytest.mark.parametrize("name", POINT_ESTIMATES)
@pytest.mark.parametrize("tol", [1e-4, 1e-8])
def test_points_needed_give_the_tolerance(name, tol):
    make_grid, func, points = POINT_ESTIMATES[name]
    grid = make_grid(300)                               # over-resolved
    axis = grid.resolution(func(grid.coords), tol)[0]
    assert axis.resolved and axis.needed < 200

    # With the suggested number of points the error is at the tolerance, with half as many it is not.
    half, suggested = _interpolation_error(make_grid, func, axis.needed, points)
    assert suggested < tol < half
    assert suggested > 1e-4 * tol                       # not wasteful either

    # The estimate is consistent: the function is resolved on that grid, but not on a clearly smaller one.
    assert make_grid(axis.needed).is_resolved(func(make_grid(axis.needed).coords), tol)
    smaller = make_grid(int(0.8 * axis.needed))
    assert not smaller.is_resolved(func(smaller.coords), tol)


@pytest.mark.parametrize("name", POINT_ESTIMATES)
def test_estimate_for_under_resolved_functions(name):
    make_grid, func, points = POINT_ESTIMATES[name]
    tol = 1e-8
    reference = make_grid(300).resolution(func(make_grid(300).coords), tol)[0].needed

    grid = make_grid(24 if name == "infinite" else 48)
    result = grid.resolution(func(grid.coords), tol)
    assert not result.resolved
    # Extrapolated from a third to two thirds of the points needed: right within a factor of 1.5.
    assert reference / 1.5 < result[0].needed < 1.5 * reference

    # Following the suggestions leads to a resolved function in a few steps.
    for _ in range(3):
        grid = result.suggested_grid
        result = grid.resolution(func(grid.coords), tol)
        if result.resolved:
            break
    assert result.resolved
    assert grid.shape[0] < 2 * reference
    assert _interpolation_error(make_grid, func, grid.shape[0], points)[1] < tol


def test_suggested_shape_is_capped():
    grid = Grid(chebyshev(100, -1, 1))
    result = grid.resolution(np.abs(grid.coords) ** 3)
    assert result[0].needed > 400
    assert result.suggested_shape == (400,)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_lowpass_keeps_two_thirds_by_default():
    rng = np.random.default_rng(9)
    grid = Grid(periodic(64))
    f = rng.standard_normal(64)
    filtered = grid.filter(f)
    assert filtered.dtype == float
    c, k = grid.coefficients(filtered), np.abs(mode_numbers(grid))
    assert np.abs(c[k > 21]).max() < 1e-14              # 2/3 rule: |k| < 64 / 3
    assert np.allclose(c[k <= 21], grid.coefficients(f)[k <= 21], atol=1e-14)

    grid = Grid(chebyshev(30, -1, 1))
    f = rng.standard_normal(30)
    c = grid.coefficients(grid.filter(f))
    assert np.abs(c[20:]).max() < 1e-13
    assert np.allclose(c[:20], grid.coefficients(f)[:20], atol=1e-13)


@pytest.mark.parametrize("make", [
    lambda: Grid(periodic(48)), lambda: Grid(periodic(33)), lambda: Grid(chebyshev(25)),
    lambda: Grid(semi_infinite(20)), lambda: Grid(infinite(21)), lambda: Grid(chebyshev(12), periodic(16)),
    lambda: PolarGrid(radial(8), periodic(16)), lambda: SphericalSurfaceGrid(colatitude(9), periodic(8)),
])
@pytest.mark.parametrize("keep", [2 / 3, 0.5, 4])
def test_truncation_is_idempotent(make, keep):
    grid = make()
    if isinstance(grid, (PolarGrid, SphericalSurfaceGrid)):
        f = grid.evaluate(lambda *c: np.exp(np.cos(c[-1]) * np.sin(c[0])))      # a single-valued field
    else:
        f = np.random.default_rng(10).standard_normal(grid.shape)
    once = grid.filter(f, keep=keep)
    assert not np.allclose(once, f, atol=1e-6)
    assert np.allclose(grid.filter(once, keep=keep), once, atol=1e-12)


@pytest.mark.parametrize("n", [48, 64, 45])
def test_two_thirds_rule_removes_aliasing(n):
    # The product of two truncated functions, truncated again, is the exact
    # truncation of the product (computed on a finer grid for comparison).
    rng = np.random.default_rng(11)
    grid, fine = Grid(periodic(n)), Grid(periodic(4 * n))
    u, v = grid.filter(rng.standard_normal(n)), grid.filter(rng.standard_normal(n))
    product = grid.resample(u, fine) * grid.resample(v, fine)
    exact = grid.filter(fine.resample(product, grid))
    assert np.allclose(grid.filter(u * v), exact, atol=1e-12)
    assert np.abs(u * v - exact).max() > 1e-2           # without the filter the product is aliased


def test_two_thirds_rule_on_chebyshev_axes():
    rng = np.random.default_rng(12)
    grid, fine = Grid(chebyshev(30, -1, 1)), Grid(chebyshev(90, -1, 1))
    u, v = grid.filter(rng.standard_normal(30)), grid.filter(rng.standard_normal(30))
    product = grid.resample(u, fine) * grid.resample(v, fine)
    exact = grid.filter(fine.resample(product, grid))
    assert np.allclose(grid.filter(u * v), exact, atol=1e-11)


def test_number_of_modes_kept():
    rng = np.random.default_rng(13)
    grid = Grid(x=chebyshev(12, -1, 1), y=periodic(16))
    f = rng.standard_normal(grid.shape)
    c = grid.coefficients(grid.filter(f, keep=5))
    kx, ky = mode_numbers(grid, "x"), np.abs(mode_numbers(grid, "y"))
    assert np.abs(c[kx >= 5]).max() < 1e-13 and np.abs(c[:, ky >= 5]).max() < 1e-13
    assert np.allclose(c[np.ix_(kx < 5, ky < 5)], grid.coefficients(f)[np.ix_(kx < 5, ky < 5)], atol=1e-13)
    assert np.allclose(grid.filter(f, keep=1.0), f, atol=1e-13)
    assert np.allclose(grid.filter(f, keep=100), f, atol=1e-13)
    assert np.allclose(grid.filter(f, keep=1), f.mean(axis=1, keepdims=True).mean(), atol=1.0)  # a constant
    assert np.ptp(grid.filter(f, keep=1)) < 1e-13


def test_filter_single_axes():
    rng = np.random.default_rng(14)
    grid = Grid(x=chebyshev(12, -1, 1), y=periodic(16))
    f = rng.standard_normal(grid.shape)
    both = grid.filter(f)
    assert np.allclose(grid.filter(grid.filter(f, axis="x"), axis="y"), both, atol=1e-13)
    assert np.allclose(grid.filter(f, axis=("y", "x")), both, atol=1e-13)
    only_x = grid.coefficients(grid.filter(f, axis=0))
    assert np.abs(only_x[8:]).max() < 1e-13
    assert np.abs(only_x[:, np.abs(mode_numbers(grid, "y")) > 5]).max() > 1e-3


def test_exponential_filter_damps_high_modes():
    grid = Grid(chebyshev(41, -1, 1))
    f = np.random.default_rng(15).standard_normal(41)
    c = grid.coefficients(f)
    sigma = grid.coefficients(grid.filter(f, "exponential")) / c
    k = np.arange(41)
    assert np.allclose(sigma, np.exp(-36 * (k / 40) ** 8), atol=1e-10)
    assert np.all(np.diff(sigma) < 1e-10)               # monotone
    assert np.allclose(sigma[:8], 1, atol=1e-4) and abs(sigma[-1]) < 1e-10

    weak = grid.coefficients(grid.filter(f, "exponential", alpha=2.0, order=2)) / c
    assert np.allclose(weak, np.exp(-2 * (k / 40) ** 2), atol=1e-10)
    assert np.allclose(grid.filter(f, "exponential", alpha=0), f, atol=1e-12)


def test_exponential_filter_on_periodic_axes_keeps_real_fields_real():
    grid = Grid(periodic(32, -1, 1))
    f = np.random.default_rng(16).standard_normal(32)
    filtered = grid.filter(f, "exponential")
    assert filtered.dtype == float
    sigma = grid.coefficients(filtered) / grid.coefficients(f)
    assert np.allclose(sigma, np.exp(-36 * (np.abs(mode_numbers(grid)) / 16) ** 8), atol=1e-10)

    z = f + 1j * np.roll(f, 3)
    assert np.allclose(grid.filter(z), grid.filter(z.real) + 1j * grid.filter(z.imag), atol=1e-13)


def test_filter_leaves_resolved_functions_alone_but_changes_boundary_values():
    grid = Grid(chebyshev(30, -1, 1))
    x = grid.coords
    assert np.allclose(grid.filter(np.exp(x)), np.exp(x), atol=1e-12)
    steep = np.tanh(20 * x)
    assert abs(grid.filter(steep)[-1] - steep[-1]) > 1e-3


def test_filter_skips_finite_difference_axes():
    grid = Grid(x=chebyshev(12, -1, 1), y=equidistant(7, 0, 1))
    g = np.random.default_rng(17).standard_normal(12)
    f = np.outer(g, grid.coords[1] ** 2)
    expected = np.outer(Grid(chebyshev(12, -1, 1)).filter(g), grid.coords[1] ** 2)
    assert np.allclose(grid.filter(f), expected, atol=1e-13)
    with pytest.raises(ValueError, match="finite differences"):
        grid.filter(f, axis="y")


def test_filter_arguments():
    grid = Grid(chebyshev(12))
    f = np.ones(12)
    with pytest.raises(ValueError, match="Unknown filter"):
        grid.filter(f, "gaussian")
    for keep in (0, 0.0, 1.5, -1):
        with pytest.raises(ValueError, match="keep"):
            grid.filter(f, keep=keep)
    with pytest.raises(TypeError, match="keep"):
        grid.filter(f, keep=True)
    with pytest.raises(ValueError, match="alpha"):
        grid.filter(f, "exponential", alpha=-1)
    with pytest.raises(ValueError, match="order"):
        grid.filter(f, "exponential", order=0)
    with pytest.raises(ValueError, match="shape"):
        grid.filter(np.ones(5))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def test_plot_spectra():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = Grid(x=chebyshev(40, -1, 1), y=periodic(32), z=equidistant(6))
    result = grid.resolution(grid.evaluate(lambda x, y, z: np.tanh(10 * x) * np.exp(np.sin(y)) * z))
    ax = result.plot()
    assert len(ax.lines) == 3                           # two spectral axes and the tolerance
    assert ax.get_yscale() == "log"
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["x (40 points, not resolved)", "y (32 points, resolved)"]

    _, own = plt.subplots()
    assert result.plot(ax=own, color="k") is own
    plt.close("all")

"""Tests for the convenience API: axis factories, named axes, grid methods."""

import numpy as np
import numpy.testing as npt
import pytest

from numgrids import (
    AxisType,
    ChebyshevAxis,
    EquidistantAxis,
    Grid,
    LogAxis,
    PolarGrid,
    SphericalGrid,
    chebyshev,
    create_axis,
    equidistant,
    logarithmic,
    periodic,
)


class TestAxisFactories:

    def test_types(self):
        assert isinstance(chebyshev(10), ChebyshevAxis)
        assert isinstance(equidistant(10), EquidistantAxis)
        assert not equidistant(10).periodic
        assert periodic(10).periodic
        assert isinstance(logarithmic(10, 1, 2), LogAxis)

    def test_defaults(self):
        assert chebyshev(5).coords[0] == -1 and chebyshev(5).coords[-1] == 1
        assert periodic(8).period == pytest.approx(2 * np.pi)

    def test_name(self):
        assert chebyshev(5, name="x").name == "x"

    def test_create_axis_accepts_strings(self):
        assert isinstance(create_axis("chebyshev", 5, 0, 1), ChebyshevAxis)
        assert create_axis("periodic", 5, 0, 1).periodic
        assert isinstance(create_axis("log", 5, 1, 2), LogAxis)
        with pytest.raises(NotImplementedError, match="Valid types"):
            create_axis("nonsense", 5, 0, 1)

    def test_create_axis_enum_still_works(self):
        assert isinstance(create_axis(AxisType.CHEBYSHEV, 5, 0, 1), ChebyshevAxis)

    def test_invalid_kwargs(self):
        with pytest.raises(TypeError):
            EquidistantAxis(10, 0, 1, nmae="x")

    def test_invalid_num_points(self):
        with pytest.raises(TypeError):
            chebyshev(10.5)

    def test_eq_and_hash(self):
        assert chebyshev(5, 0, 1) == chebyshev(5, 0, 1)
        assert chebyshev(5, 0, 1) != chebyshev(6, 0, 1)
        assert chebyshev(5, 0, 1) != equidistant(5, 0, 1)
        assert len({chebyshev(5, 0, 1), chebyshev(5, 0, 1)}) == 1

    def test_renamed(self):
        ax = chebyshev(5, 0, 1)
        ax2 = ax.renamed("y")
        assert ax.name is None and ax2.name == "y"
        npt.assert_array_equal(ax.coords, ax2.coords)

    def test_period_only_for_periodic(self):
        with pytest.raises(AttributeError):
            chebyshev(5).period


class TestNamedAxes:

    def test_keyword_construction(self):
        grid = Grid(x=chebyshev(10, 0, 1), y=periodic(12))
        assert [ax.name for ax in grid.axes] == ["x", "y"]
        assert grid.axis_index("y") == 1
        assert grid.get_axis("x") is grid.axes[0]

    def test_mixed_positional_and_keyword(self):
        grid = Grid(chebyshev(10, name="a"), b=equidistant(5))
        assert grid.axis_index("a") == 0
        assert grid.axis_index("b") == 1

    def test_duplicate_names(self):
        with pytest.raises(ValueError):
            Grid(chebyshev(10, name="x"), x=chebyshev(10))

    def test_unknown_name(self):
        grid = Grid(x=chebyshev(10))
        with pytest.raises(KeyError):
            grid.axis_index("z")

    def test_negative_index(self):
        grid = Grid(chebyshev(10), chebyshev(11))
        assert grid.axis_index(-1) == 1

    def test_non_axis_rejected(self):
        with pytest.raises(TypeError):
            Grid(np.linspace(0, 1, 10))
        with pytest.raises(TypeError):
            Grid(x=np.linspace(0, 1, 10))

    def test_faces_aliases(self):
        grid = Grid(x=chebyshev(10), y=periodic(12), z=equidistant(5))
        assert grid.faces["x_low"] is grid.faces["0_low"]
        assert grid.faces["z_high"] is grid.faces["2_high"]
        assert "x_high" in grid.faces
        assert grid.faces.get("y_low") is None
        assert sorted(grid.faces) == ["0_high", "0_low", "2_high", "2_low"]
        with pytest.raises(KeyError, match="Available faces"):
            grid.faces["y_low"]

    def test_default_names_of_builtin_grids(self):
        grid = SphericalGrid(chebyshev(5, 0.1, 1), chebyshev(5, 0.1, 3), periodic(6))
        assert [ax.name for ax in grid.axes] == ["r", "theta", "phi"]
        assert "r_high" in grid.faces

    def test_user_names_take_precedence(self):
        grid = PolarGrid(chebyshev(5, 0.1, 1, name="rho"), periodic(6))
        assert [ax.name for ax in grid.axes] == ["rho", "phi"]


class TestGridMethods:

    @pytest.fixture
    def grid(self):
        return Grid(x=chebyshev(24, 0, 1), y=periodic(32))

    def test_evaluate(self, grid):
        X, Y = grid.meshed_coords
        npt.assert_array_equal(grid.evaluate(lambda x, y: x * np.sin(y)), X * np.sin(Y))
        npt.assert_array_equal(grid.evaluate(lambda x, y: 3.0), np.full(grid.shape, 3.0))

    def test_diff(self, grid):
        X, Y = grid.meshed_coords
        f = X ** 3 * np.sin(Y)
        npt.assert_allclose(grid.diff(f, axis="x"), 3 * X ** 2 * np.sin(Y), atol=1e-10)
        npt.assert_allclose(grid.diff(f, 2, "y"), -f, atol=1e-10)

    def test_integrate(self, grid):
        X, Y = grid.meshed_coords
        assert grid.integrate(X * np.cos(Y) ** 2) == pytest.approx(np.pi / 2)

    def test_interpolate(self, grid):
        X, Y = grid.meshed_coords
        val = grid.interpolate(X ** 2 * np.cos(Y), (0.5, 1.0), method="spectral")
        assert val == pytest.approx(0.25 * np.cos(1.0))

    def test_laplacian(self, grid):
        X, Y = grid.meshed_coords
        f = X ** 3 * np.sin(Y)
        npt.assert_allclose(grid.laplacian(f), 6 * X * np.sin(Y) - f, atol=1e-8)

    def test_gradient_divergence(self, grid):
        X, Y = grid.meshed_coords
        f = X ** 3 * np.sin(Y)
        gx, gy = grid.gradient(f)
        npt.assert_allclose(gx, 3 * X ** 2 * np.sin(Y), atol=1e-10)
        npt.assert_allclose(gy, X ** 3 * np.cos(Y), atol=1e-10)
        npt.assert_allclose(grid.divergence(gx, gy), grid.laplacian(f), atol=1e-8)

    def test_curl_2d(self, grid):
        X, Y = grid.meshed_coords
        npt.assert_allclose(grid.curl(-Y * 0 + 0 * X, X), np.ones(grid.shape), atol=1e-10)

    def test_curl_3d_of_gradient_vanishes(self):
        grid = Grid(chebyshev(12, 0, 1), chebyshev(12, 0, 1), chebyshev(12, 0, 1))
        X, Y, Z = grid.meshed_coords
        f = np.sin(X) * Y ** 2 * np.exp(Z)
        for c in grid.curl(*grid.gradient(f)):
            npt.assert_allclose(c, 0, atol=1e-8)

    def test_curl_wrong_dims(self):
        grid = Grid(chebyshev(10))
        with pytest.raises(ValueError):
            grid.curl(grid.coords)

    def test_shape_errors(self, grid):
        with pytest.raises(ValueError, match="does not match"):
            grid.laplacian(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="does not match"):
            grid.diff(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="does not match"):
            grid.integrate(np.zeros((3, 3)))

    def test_interior_mask(self, grid):
        npt.assert_array_equal(grid.interior, ~grid.boundary)

    def test_volume_element_and_cartesian(self, grid):
        assert grid.volume_element == 1.0
        assert grid.cartesian_coords is grid.meshed_coords

    def test_size_is_int(self, grid):
        assert type(grid.size) is int

    def test_repr_lists_axes(self, grid):
        r = repr(grid)
        assert "ChebyshevAxis" in r and "name='y'" in r and str(grid.shape) in r

    def test_equality(self):
        assert Grid(chebyshev(5), chebyshev(6)) == Grid(chebyshev(5), chebyshev(6))
        assert Grid(chebyshev(5)) != Grid(chebyshev(6))
        assert hash(Grid(chebyshev(5))) == hash(Grid(chebyshev(5)))

    def test_refine_axis_by_name(self, grid):
        assert grid.refine_axis("y").shape == (24, 64)

    def test_save_load(self, grid, tmp_path):
        X, Y = grid.meshed_coords
        grid.save(tmp_path / "g.npz", u=X)
        loaded, data = Grid.load(tmp_path / "g.npz")
        assert loaded == grid
        npt.assert_array_equal(data["u"], X)

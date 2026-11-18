import unittest

import numpy as np
import numpy.testing as npt
import pytest

from numgrids import AxisType, Grid, PolarGrid, chebyshev, create_axis, evolve, periodic, radial
from numgrids.plots import Plotter


class TestPlotter(unittest.TestCase):

    def test_plot(self):
        axis = create_axis(AxisType.EQUIDISTANT, 30, -1, 1)
        grid = Grid(axis, axis)

        X, Y = grid.meshed_coords
        f = X**2 + Y**2

        plotter = Plotter(grid)

        plotter.plot(f, axis.coords, 0)
        self.assertIsNotNone(plotter.ax)


# ── animations ──────────────────────────────────────────────────────────

def _heat_result(n=12, nt=5):
    grid = Grid(x=chebyshev(n, 0, 1))
    result = evolve(grid, 0.1 * grid.laplacian_operator(), np.sin(np.pi * grid.coords), (0, 1),
                    bcs=0, method="BDF", t_eval=np.linspace(0, 1, nt))
    return grid, result


def _axes():
    """The axes of the figure created last (the colorbar, if any, comes second)."""
    import matplotlib.pyplot as plt
    return plt.gcf().axes


def _play(anim, probe):
    """Run through the animation without rendering; return ``probe()`` after every frame."""
    from matplotlib.animation import AbstractMovieWriter

    class Recorder(AbstractMovieWriter):
        def setup(self, fig, outfile, dpi=None):
            self.states = []

        def grab_frame(self, **kwargs):
            self.states.append(probe())

        def finish(self):
            pass

    recorder = Recorder()
    anim.save("unused", writer=recorder)
    return recorder.states


class TestAnimations:

    @pytest.fixture(autouse=True)
    def _matplotlib(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        yield
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_line_1d(self):
        grid, result = _heat_result()
        anim = grid.animate(result.u, result.t)
        (ax,) = _axes()
        (line,) = ax.get_lines()
        npt.assert_array_equal(line.get_xdata(), grid.coords)
        npt.assert_array_equal(line.get_ydata(), result.u[0])
        assert ax.get_xlabel() == "$x$"
        # limits cover all frames, with a little room
        low, high = ax.get_ylim()
        assert low < result.u.min() and high > result.u.max() and high - low < 1.2 * np.ptp(result.u)

        states = _play(anim, lambda: (line.get_ydata().copy(), ax.get_title(), ax.get_ylim()))
        assert len(states) == len(result.t)
        for k, (ydata, title, ylim) in enumerate(states):
            npt.assert_array_equal(ydata, result.u[k])
            assert title.startswith("t = ") and float(title[4:]) == pytest.approx(result.t[k])
            assert ylim == (low, high)
        # the line is updated, not drawn anew
        assert ax.get_lines() == [line]

    def test_color_plot_2d(self):
        grid = Grid(x=chebyshev(6, 0, 1), y=chebyshev(5, 0, 2))
        X, Y = grid.meshed_coords
        t = np.array([0.0, 0.5, 1.0])
        frames = np.array([np.cos(tk) * X + tk * Y for tk in t])
        anim = grid.animate(frames, t, cmap="magma", title="field")
        assert len(_axes()) == 2                       # the plot and its colorbar
        ax = _axes()[0]
        (mesh,) = ax.collections
        assert mesh.get_cmap().name == "magma"
        assert mesh.get_clim() == (frames.min(), frames.max())
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("$x$", "$y$")

        states = _play(anim, lambda: (np.array(mesh.get_array()).ravel(), ax.get_title(), mesh.get_clim()))
        for k, (values, title, clim) in enumerate(states):
            npt.assert_array_equal(values, frames[k].ravel())
            assert title == f"field, t = {t[k]:.2f}"
            assert clim == (frames.min(), frames.max())
        # the mesh is updated, not drawn anew
        assert list(ax.collections) == [mesh] and len(_axes()) == 2

    def test_polar_grid_is_drawn_as_closed_disk(self):
        disk = PolarGrid(radial(6), periodic(8))
        X, Y = disk.cartesian_coords
        frames = np.array([1 + k * (X + Y ** 2) for k in range(3)])
        anim = disk.animate(frames)
        ax = _axes()[0]
        (mesh,) = ax.collections
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("$x$", "$y$")
        states = _play(anim, lambda: (np.array(mesh.get_array()).reshape(7, 9), ax.get_title()))
        for k, (values, title) in enumerate(states):
            npt.assert_array_equal(values[1:, :-1], frames[k])            # the grid values
            npt.assert_array_equal(values[:, -1], values[:, 0])           # periodic direction closed
            npt.assert_allclose(values[0], 1.0, atol=1e-12)               # the center
            assert title == ""                                            # no times given

    def test_same_picture_as_plot(self):
        disk = PolarGrid(radial(6), periodic(8))
        f = disk.cartesian_coords[0] ** 2
        (static,) = disk.plot(f).collections
        anim = disk.animate(f[None])
        (animated,) = _axes()[0].collections
        _play(anim, lambda: None)
        npt.assert_array_equal(np.ravel(static.get_array()), np.ravel(animated.get_array()))
        npt.assert_array_equal(static.get_coordinates(), animated.get_coordinates())
        npt.assert_allclose(animated.get_clim(), static.get_clim(), atol=1e-12)

    def test_slice_of_3d_grid(self):
        grid = Grid(x=chebyshev(4, 0, 1), y=periodic(6), z=chebyshev(5, 0, 1))
        X, Y, Z = grid.meshed_coords
        frames = np.array([X + k * Z for k in range(2)])
        anim = grid.animate(frames, fixed={"x": 1})
        ax = _axes()[0]
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("$y$", "$z$")
        (mesh,) = ax.collections
        states = _play(anim, lambda: np.array(mesh.get_array()).reshape(7, 5))
        npt.assert_array_equal(states[1][:-1], frames[1][1])
        # the color scale covers the slice shown, not the whole grid
        assert mesh.get_clim() == (frames[:, 1].min(), frames[:, 1].max())
        import matplotlib.pyplot as plt
        figures = plt.get_fignums()
        with pytest.raises(ValueError, match="Fix exactly 1"):
            grid.animate(frames, fixed={})
        assert plt.get_fignums() == figures            # no empty figure is left behind

    def test_result_knows_its_grid(self):
        grid, result = _heat_result()
        assert result.grid is grid
        assert "grid" not in repr(result)
        anim = result.animate(fps=4)
        (ax,) = _axes()
        npt.assert_array_equal(ax.get_lines()[0].get_ydata(), result.u[0])
        assert _play(anim, ax.get_title)[-1] == "t = 1.00"

    def test_result_of_expm_knows_its_grid(self):
        grid = Grid(x=chebyshev(8, 0, 1))
        result = evolve(grid, grid.laplacian_operator(), np.sin(np.pi * grid.coords), (0, 0.1),
                        bcs=0, method="expm")
        assert result.grid is grid

    def test_field_of_a_system(self):
        grid = Grid(x=chebyshev(12, 0, 1))
        x = grid.coords
        L = grid.laplacian_operator()
        result = evolve(grid, [[None, 1], [L, None]], (np.sin(np.pi * x), np.zeros_like(x)), (0, 0.2),
                        bcs=0, t_eval=[0, 0.1, 0.2])
        assert result.grid is grid
        for field in (0, 1, -1):
            anim = result.animate(field=field)
            (line,) = _axes()[0].get_lines()
            states = _play(anim, lambda line=line: line.get_ydata().copy())
            npt.assert_array_equal(np.array(states), result.u[field])
        with pytest.raises(IndexError, match="2 fields"):
            result.animate(field=2)
        with pytest.raises(TypeError, match="tuple of fields"):
            grid.animate(result.u, result.t)

    def test_single_field_has_no_other_fields(self):
        _, result = _heat_result()
        with pytest.raises(IndexError, match="single field"):
            result.animate(field=1)

    def test_result_without_grid(self):
        from numgrids import EvolutionResult
        result = EvolutionResult(t=np.arange(2.0), u=np.zeros((2, 4)), success=True, message="")
        assert result.grid is None
        with pytest.raises(ValueError, match="does not know its grid"):
            result.animate()

    def test_fixed_limits(self):
        grid, result = _heat_result()
        _play(grid.animate(result.u, ylim=(-1, 2)), lambda: None)
        assert _axes()[0].get_ylim() == (-1, 2)
        _play(grid.animate(result.u, vmin=-0.5, vmax=1.5), lambda: None)
        assert _axes()[0].get_ylim() == (-0.5, 1.5)
        _play(grid.animate(np.ones((2,) + grid.shape)), lambda: None)
        assert _axes()[0].get_ylim() == (0.5, 1.5)

        grid = Grid(x=chebyshev(5, 0, 1), y=chebyshev(4, 0, 1))
        frames = np.array([k * grid.meshed_coords[0] for k in range(3)])
        anim = grid.animate(frames, vmin=-1, vmax=4, colorbar=False)
        assert len(_axes()) == 1
        (mesh,) = _axes()[0].collections
        assert _play(anim, mesh.get_clim) == [(-1, 4)] * 3
        _play(grid.animate(frames, vmax=1), lambda: None)
        assert _axes()[0].collections[0].get_clim() == (0, 1)

    def test_speed(self):
        grid, result = _heat_result()
        for options, interval in [({}, 50), ({"interval": 200}, 200), ({"fps": 4}, 250)]:
            anim = grid.animate(result.u, **options)
            assert anim.event_source.interval == interval
            _play(anim, lambda: None)
        with pytest.raises(ValueError, match="either"):
            grid.animate(result.u, interval=100, fps=10)

    def test_complex_fields(self):
        grid = Grid(x=periodic(8))
        t = np.linspace(0, 1, 3)
        frames = np.exp(1j * (grid.coords[None, :] - t[:, None])) * (1 + t[:, None])
        for part, expected in [("real", frames.real), ("imag", frames.imag), ("abs", np.abs(frames)),
                               ("abs2", np.abs(frames) ** 2), (np.angle, np.angle(frames))]:
            anim = grid.animate(frames, t, part=part)
            (ax,) = _axes()
            states = _play(anim, lambda ax=ax: ax.get_lines()[0].get_ydata().copy())
            npt.assert_allclose(np.array(states), expected, atol=1e-14)
            low, high = ax.get_ylim()
            assert low <= expected.min() and high >= expected.max()
        with pytest.raises(ValueError, match="Unknown part"):
            grid.animate(frames, part="phase")

    def test_draws_into_given_axes(self):
        import matplotlib.pyplot as plt
        grid, result = _heat_result()
        fig, (left, right) = plt.subplots(1, 2)
        anim = grid.animate(result.u, result.t, ax=right, color="k", title="heat")
        assert plt.get_fignums() == [fig.number] and not left.get_lines()
        assert right.get_lines()[0].get_color() == "k"
        assert right.get_title() == "heat"
        assert _play(anim, right.get_title)[0] == "heat, t = 0.00"

    def test_wrong_input(self):
        grid, result = _heat_result()
        with pytest.raises(ValueError, match="expected \\(nt, \\*grid.shape\\)"):
            grid.animate(result.u[0])
        with pytest.raises(ValueError, match="expected \\(nt, \\*grid.shape\\)"):
            grid.animate(result.u[:, :-1])
        with pytest.raises(ValueError, match="time values"):
            grid.animate(result.u, result.t[:-1])

    def test_time_format(self):
        from numgrids.plots import _time_format
        assert _time_format(np.linspace(0, 1, 11)).format(0.3) == "0.30"
        assert _time_format(np.array([0, 0.25, 0.5])).format(0.25) == "0.25"
        assert _time_format(np.arange(0.0, 50, 10)).format(20) == "20"
        assert _time_format(np.array([0.0, 1e-7])).format(1e-7) == "1.000e-07"
        assert _time_format(np.array([2.0])).format(2) == "2.00"

    def test_to_jshtml(self):
        _, result = _heat_result(nt=3)
        anim = result.animate()
        html = anim.to_jshtml()
        assert "<img" in html and html.count("data:image/png;base64") == 3
        # the last frame was drawn last
        npt.assert_array_equal(_axes()[0].get_lines()[0].get_ydata(), result.u[-1])

    def test_save_gif(self, tmp_path):
        Image = pytest.importorskip("PIL.Image")
        from numgrids.plots import save_animation

        disk = PolarGrid(radial(5), periodic(8))
        frames = np.array([k * disk.cartesian_coords[0] for k in range(3)])
        path = tmp_path / "disk.gif"
        anim = disk.animate(frames, save=path)
        with Image.open(path) as image:
            assert image.format == "GIF" and image.n_frames == 3

        slow = save_animation(anim, tmp_path / "slow.gif", fps=2, dpi=40)
        with Image.open(slow) as image:
            assert image.n_frames == 3 and image.info["duration"] == 500

    def test_save_html(self, tmp_path):
        from numgrids.plots import save_animation
        _, result = _heat_result(nt=2)
        path = save_animation(result.animate(), str(tmp_path / "heat.html"))
        assert path == tmp_path / "heat.html"
        assert path.read_text(encoding="utf-8").count("data:image/png;base64") == 2

    def test_missing_writer(self, tmp_path, monkeypatch):
        from matplotlib import animation

        from numgrids.plots import save_animation
        _, result = _heat_result(nt=2)
        anim = result.animate()
        with pytest.raises(RuntimeError, match="writer 'no-such-writer' is not available"):
            save_animation(anim, tmp_path / "heat.mp4", writer="no-such-writer")
        # e.g. no ffmpeg on the system
        monkeypatch.setattr(animation.writers, "is_available", lambda name: False)
        with pytest.raises(RuntimeError, match="Cannot save 'heat.mp4'.*'.gif'.*'.html'"):
            save_animation(anim, tmp_path / "heat.mp4")
        assert not (tmp_path / "heat.mp4").exists()
        _play(anim, lambda: None)

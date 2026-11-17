"""VTK export: the files are parsed and decoded here the way a VTK reader does it."""

import base64
import xml.etree.ElementTree as ET

import numpy as np
import numpy.testing as npt
import pytest

from numgrids import (
    CurvilinearGrid,
    CylindricalGrid,
    EvolutionResult,
    Grid,
    PolarGrid,
    SphericalGrid,
    SphericalSurfaceGrid,
    chebyshev,
    colatitude,
    equidistant,
    evolve,
    periodic,
    radial,
)
from numgrids.vtk import cartesian_components, write_vtk, write_vtk_series

# ── reading the files back ──────────────────────────────────────────────


def decode(element):
    """The numbers of a DataArray element, in the order of the file."""
    assert element.get("type") == "Float64"
    if element.get("format") == "ascii":
        return np.array(element.text.split(), dtype=float)
    assert element.get("format") == "binary"
    raw = base64.b64decode(element.text.strip(), validate=True)
    # header_type="UInt64": the number of bytes that follow
    (nbytes,) = np.frombuffer(raw[:8], dtype="<u8")
    assert nbytes == len(raw) - 8
    return np.frombuffer(raw[8:], dtype="<f8")


class VtsFile:
    """A parsed ``.vts`` file; arrays are reshaped to the index order of numgrids."""

    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        assert self.root.tag == "VTKFile"
        assert self.root.get("type") == "StructuredGrid"
        assert self.root.get("byte_order") == "LittleEndian"
        assert self.root.get("header_type") == "UInt64"
        grid = self.root.find("StructuredGrid")
        (piece,) = grid.findall("Piece")
        self.extent = grid.get("WholeExtent")
        assert piece.get("Extent") == self.extent
        numbers = [int(n) for n in self.extent.split()]
        assert numbers[::2] == [0, 0, 0]
        self.dims = tuple(n + 1 for n in numbers[1::2])
        self.point_data = piece.find("PointData")
        (points,) = piece.find("Points").findall("DataArray")
        assert points.get("NumberOfComponents") == "3"
        self.flat_points = decode(points).reshape(-1, 3)
        assert len(self.flat_points) == int(np.prod(self.dims))
        self.flat = {}
        for element in self.point_data.findall("DataArray"):
            components = int(element.get("NumberOfComponents"))
            values = decode(element)
            assert values.size == components * len(self.flat_points)
            self.flat[element.get("Name")] = values.reshape(-1, components) if components > 1 else values

    def _mesh(self, flat):
        # In the file the first index runs fastest.
        return flat.reshape(self.dims + flat.shape[1:], order="F")

    @property
    def points(self):
        """Array of shape ``(n0, n1, n2, 3)``."""
        return self._mesh(self.flat_points)

    @property
    def xyz(self):
        return tuple(self.points[..., k] for k in range(3))

    def __getitem__(self, name):
        return self._mesh(self.flat[name])

    @property
    def names(self):
        return list(self.flat)


def read_pvd(path):
    """``[(time, file name), ...]`` of a collection file."""
    root = ET.parse(path).getroot()
    assert root.tag == "VTKFile" and root.get("type") == "Collection"
    assert root.get("byte_order") == "LittleEndian"
    (collection,) = root.findall("Collection")
    return [(float(d.get("timestep")), d.get("file")) for d in collection.findall("DataSet")]


# ── grids ───────────────────────────────────────────────────────────────


def disk(nr=6, nphi=8):
    return PolarGrid(radial(nr), periodic(nphi))


def ball(nr=6, ntheta=6, nphi=8):
    return SphericalGrid(radial(nr), colatitude(ntheta), periodic(nphi))


def smooth(x, y, z=0.0):
    """A polynomial: reproduced exactly by the spectral interpolation."""
    return 1 + x - 2 * y + 3 * z + x * y + z ** 2 - x ** 2


def cell_volumes(points):
    """Signed volumes of the hexahedra between neighbouring points (positive: right-handed)."""
    n0, n1, n2 = (n - 1 for n in points.shape[:3])
    corner = {(a, b, c): points[a:a + n0, b:b + n1, c:c + n2] for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    center = sum(corner.values()) / 8
    faces = [                                                  # corners in counter-clockwise order, seen from outside
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    ]
    volume = 0.0
    for face in faces:
        middle = sum(corner[c] for c in face) / 4 - center
        for k in range(4):                                     # a fan of triangles around the middle of the face
            a, b = corner[face[k]] - center, corner[face[(k + 1) % 4]] - center
            volume = volume + np.sum(middle * np.cross(a, b), axis=-1) / 6
    return volume


# ── Cartesian grids ─────────────────────────────────────────────────────


class TestCartesianGrids:

    def test_1d(self, tmp_path):
        grid = Grid(x=chebyshev(7, -1, 2))
        x = grid.coords
        path = write_vtk(tmp_path / "line.vts", grid, u=x ** 2)
        assert path == tmp_path / "line.vts"
        file = VtsFile(path)
        assert file.extent == "0 6 0 0 0 0"
        npt.assert_array_equal(file.flat_points, np.column_stack([x, 0 * x, 0 * x]))
        npt.assert_array_equal(file.flat["u"], x ** 2)

    def test_2d(self, tmp_path):
        grid = Grid(x=chebyshev(5, 0, 1), y=equidistant(4, 0, 2))
        X, Y = grid.meshed_coords
        u, w = np.sin(X) * Y, X - Y
        file = VtsFile(write_vtk(tmp_path / "plane.vts", grid, u=u, w=w))
        assert file.extent == "0 4 0 3 0 0" and file.dims == (5, 4, 1)
        x, y, z = file.xyz
        npt.assert_array_equal(x[..., 0], X)
        npt.assert_array_equal(y[..., 0], Y)
        assert not z.any()
        assert file.names == ["u", "w"]
        npt.assert_array_equal(file["u"][..., 0], u)
        npt.assert_array_equal(file["w"][..., 0], w)
        assert file.point_data.get("Scalars") == "u" and file.point_data.get("Vectors") is None

    def test_3d_point_order(self, tmp_path):
        grid = Grid(chebyshev(3, 0, 1), chebyshev(4, 0, 1), chebyshev(5, 0, 1))
        X, Y, Z = grid.meshed_coords
        I, J, K = grid.meshed_indices
        file = VtsFile(write_vtk(tmp_path / "box.vts", grid, code=I + 10 * J + 100 * K))
        assert file.extent == "0 2 0 3 0 4"
        # VTK numbers the points with the first index running fastest
        flat = [(i, j, k) for k in range(5) for j in range(4) for i in range(3)]
        npt.assert_array_equal(file.flat["code"], [i + 10 * j + 100 * k for i, j, k in flat])
        npt.assert_array_equal(file.flat_points, [(X[p], Y[p], Z[p]) for p in flat])

    def test_periodic_direction_is_closed(self, tmp_path):
        grid = Grid(x=periodic(8), y=chebyshev(4, 0, 1))
        X, Y = grid.meshed_coords
        u = np.sin(X) + Y
        file = VtsFile(write_vtk(tmp_path / "strip.vts", grid, u=u))
        assert file.extent == "0 8 0 3 0 0"
        x, y, _ = file.xyz
        npt.assert_array_equal(x[:-1, :, 0], X)
        # on a Cartesian grid the closing points lie one period further
        npt.assert_array_equal(x[-1], 2 * np.pi)
        npt.assert_array_equal(y[-1], y[0])
        npt.assert_array_equal(file["u"][:-1, :, 0], u)
        npt.assert_array_equal(file["u"][-1], file["u"][0])

    def test_exactly_the_grid_points_without_closing(self, tmp_path):
        grid = Grid(x=periodic(8), y=chebyshev(4, 0, 1))
        X, Y = grid.meshed_coords
        file = VtsFile(write_vtk(tmp_path / "strip.vts", grid, close=False, u=X * Y))
        assert file.extent == "0 7 0 3 0 0"
        npt.assert_array_equal(file.xyz[0][..., 0], X)
        npt.assert_array_equal(file["u"][..., 0], X * Y)

    def test_vector_field(self, tmp_path):
        grid = Grid(x=chebyshev(4, 0, 1), y=chebyshev(3, 0, 1))
        X, Y = grid.meshed_coords
        file = VtsFile(write_vtk(tmp_path / "v.vts", grid, v=(X, -Y), s=X))
        v = file["v"][..., 0, :]
        npt.assert_array_equal(v[..., 0], X)
        npt.assert_array_equal(v[..., 1], -Y)
        assert not v[..., 2].any()
        assert file.point_data.get("Vectors") == "v" and file.point_data.get("Scalars") == "s"
        npt.assert_array_equal(cartesian_components(grid, X, -Y), (X, -Y))

    def test_no_fields(self, tmp_path):
        grid = Grid(x=chebyshev(4, 0, 1))
        file = VtsFile(write_vtk(tmp_path / "empty.vts", grid))
        assert file.names == [] and file.dims == (4, 1, 1)


# ── encoding ────────────────────────────────────────────────────────────


class TestEncoding:

    def test_binary_layout(self, tmp_path):
        grid = Grid(x=chebyshev(5, 0, 1), y=chebyshev(4, 0, 1))
        u = np.arange(20.0).reshape(5, 4) / 7
        path = write_vtk(tmp_path / "u.vts", grid, u=u)
        (element,) = ET.parse(path).getroot().iter("PointData").__next__().findall("DataArray")
        assert element.get("format") == "binary"
        text = element.text.strip()
        # one base64 string: the byte count as little-endian UInt64, then the doubles
        raw = base64.b64decode(text, validate=True)
        assert len(raw) == 8 + 20 * 8
        assert int.from_bytes(raw[:8], "little") == 20 * 8
        npt.assert_array_equal(np.frombuffer(raw[8:], dtype="<f8"), u.ravel(order="F"))
        assert text == base64.b64encode(raw).decode()

    def test_ascii_and_binary_hold_the_same_numbers(self, tmp_path):
        grid = disk()
        R, Phi = grid.meshed_coords
        fields = {"u": np.sin(3 * R) * np.cos(Phi) / 3, "v": (R ** 2, np.cos(Phi)), "psi": np.exp(1j * Phi) / 7}
        binary = VtsFile(write_vtk(tmp_path / "binary.vts", grid, **fields))
        ascii_ = VtsFile(write_vtk(tmp_path / "ascii.vts", grid, binary=False, **fields))
        assert {e.get("format") for e in ascii_.root.iter("DataArray")} == {"ascii"}
        assert {e.get("format") for e in binary.root.iter("DataArray")} == {"binary"}
        assert binary.extent == ascii_.extent
        assert binary.names == ascii_.names == ["u", "v", "psi_real", "psi_imag", "psi_abs"]
        npt.assert_array_equal(binary.flat_points, ascii_.flat_points)       # bit for bit
        for name in binary.names:
            npt.assert_array_equal(binary.flat[name], ascii_.flat[name])

    def test_single_precision_and_integers_are_written_as_doubles(self, tmp_path):
        grid = Grid(x=chebyshev(5, 0, 1))
        single = np.linspace(0, 1, 5, dtype=np.float32) / 3
        file = VtsFile(write_vtk(tmp_path / "f.vts", grid, single=single, count=np.arange(5), flag=grid.coords > 0.5))
        npt.assert_array_equal(file.flat["single"], single.astype(float))
        npt.assert_array_equal(file.flat["count"], np.arange(5.0))
        npt.assert_array_equal(file.flat["flag"], (grid.coords > 0.5).astype(float))

    def test_names_are_escaped(self, tmp_path):
        grid = Grid(x=chebyshev(4, 0, 1))
        file = VtsFile(write_vtk(tmp_path / "n.vts", grid, **{'p & "q" <r>': grid.coords, "∇u": 2 * grid.coords}))
        assert file.names == ['p & "q" <r>', "∇u"]
        assert file.point_data.get("Scalars") == 'p & "q" <r>'


# ── curvilinear grids ───────────────────────────────────────────────────


class TestCurvilinearGrids:

    def test_annulus(self, tmp_path):
        grid = PolarGrid(chebyshev(5, 0.5, 1), periodic(8))
        X, Y = grid.cartesian_coords
        u = smooth(X, Y)
        file = VtsFile(write_vtk(tmp_path / "ring.vts", grid, u=u))
        assert file.extent == "0 4 0 8 0 0"
        x, y, z = file.xyz
        npt.assert_array_equal(x[:, :-1, 0], X)
        npt.assert_array_equal(y[:, :-1, 0], Y)
        assert not z.any()
        # no seam: the closing points are the first points, bit for bit
        npt.assert_array_equal(file.points[:, -1], file.points[:, 0])
        npt.assert_array_equal(file["u"][:, :-1, 0], u)
        npt.assert_array_equal(file["u"][:, -1], file["u"][:, 0])

    def test_disk_has_a_center(self, tmp_path):
        grid = disk()
        X, Y = grid.cartesian_coords
        u = smooth(X, Y)
        file = VtsFile(write_vtk(tmp_path / "disk.vts", grid, u=u))
        assert file.extent == "0 6 0 8 0 0"
        assert not file.points[0].any()                        # r = 0, for all angles
        npt.assert_array_equal(file.points[1:, :-1, 0, 0], X)
        npt.assert_array_equal(file.points[1:, :-1, 0, 1], Y)
        values = file["u"][..., 0]
        npt.assert_array_equal(values[1:, :-1], u)             # the grid values, unchanged
        npt.assert_allclose(values[0], smooth(0, 0), atol=1e-13)
        assert np.ptp(values[0]) == 0                          # one value at the center
        npt.assert_array_equal(values[:, -1], values[:, 0])

    def test_disk_without_closing(self, tmp_path):
        grid = disk()
        X, Y = grid.cartesian_coords
        file = VtsFile(write_vtk(tmp_path / "disk.vts", grid, close=False, u=X))
        assert file.extent == "0 5 0 7 0 0"
        npt.assert_array_equal(file.points[..., 0, 0], X)
        npt.assert_array_equal(file["u"][..., 0], X)

    def test_cylinder_has_an_axis(self, tmp_path):
        grid = CylindricalGrid(radial(5), periodic(8), chebyshev(4, -1, 1))
        X, Y, Z = grid.cartesian_coords
        file = VtsFile(write_vtk(tmp_path / "cyl.vts", grid, u=smooth(X, Y, Z)))
        assert file.extent == "0 5 0 8 0 3"
        x, y, z = file.xyz
        assert not x[0].any() and not y[0].any()
        npt.assert_array_equal(z[0], np.broadcast_to(grid.axes[2].coords, (9, 4)))
        npt.assert_array_equal(file["u"][1:, :-1], smooth(X, Y, Z))
        npt.assert_allclose(file["u"], smooth(x, y, z), atol=1e-13)
        assert not np.ptp(file["u"][0], axis=0).any()          # one value per point of the axis

    def test_sphere_has_poles(self, tmp_path):
        grid = SphericalSurfaceGrid(colatitude(6), periodic(8), radius=2.0)
        X, Y, Z = grid.cartesian_coords
        file = VtsFile(write_vtk(tmp_path / "sphere.vts", grid, u=smooth(X, Y, Z)))
        assert file.extent == "0 7 0 8 0 0"
        x, y, z = file.xyz
        npt.assert_array_equal(file.points[0], np.broadcast_to([0, 0, 2.0], (9, 1, 3)))
        npt.assert_allclose(file.points[-1], np.broadcast_to([0, 0, -2.0], (9, 1, 3)), atol=1e-15)
        assert not np.ptp(file.points[-1], axis=0).any()       # all copies of the pole are identical
        npt.assert_allclose(np.sqrt(x ** 2 + y ** 2 + z ** 2), 2.0, atol=1e-15)
        values = file["u"][..., 0]
        npt.assert_array_equal(values[1:-1, :-1], smooth(X, Y, Z))
        npt.assert_allclose(values[0], smooth(0, 0, 2.0), atol=1e-12)
        npt.assert_allclose(values[-1], smooth(0, 0, -2.0), atol=1e-12)
        assert np.ptp(values[0]) == 0 and np.ptp(values[-1]) == 0
        npt.assert_array_equal(values[:, -1], values[:, 0])

    def test_ball_has_center_and_poles(self, tmp_path):
        grid = ball()
        X, Y, Z = grid.cartesian_coords
        file = VtsFile(write_vtk(tmp_path / "ball.vts", grid, u=smooth(X, Y, Z)))
        assert file.extent == "0 6 0 7 0 8"
        x, y, z = file.xyz
        assert not file.points[0].any()                                    # the center
        assert not x[:, 0].any() and not y[:, 0].any()                     # north half of the z axis
        npt.assert_allclose(x[:, -1], 0, atol=1e-15)
        npt.assert_array_equal(z[1:, 0], np.broadcast_to(grid.axes[0].coords[:, None], (6, 9)))
        npt.assert_array_equal(file["u"][1:, 1:-1, :-1], smooth(X, Y, Z))  # the grid values, unchanged
        npt.assert_allclose(file["u"], smooth(x, y, z), atol=1e-12)        # also at all added points
        assert np.ptp(file["u"][0]) == 0
        npt.assert_array_equal(file["u"][:, :, -1], file["u"][:, :, 0])
        npt.assert_array_equal(file.points[:, :, -1], file.points[:, :, 0])

    def test_cells_cover_the_disk_once(self, tmp_path):
        file = VtsFile(write_vtk(tmp_path / "disk.vts", disk(6, 8)))
        x, y, _ = (c[..., 0] for c in file.xyz)
        # the quadrilateral of cell (i, j), as a viewer builds it from the neighbouring points
        cx = [x[:-1, :-1], x[1:, :-1], x[1:, 1:], x[:-1, 1:]]
        cy = [y[:-1, :-1], y[1:, :-1], y[1:, 1:], y[:-1, 1:]]
        area = 0.5 * sum(cx[k] * cy[(k + 1) % 4] - cx[(k + 1) % 4] * cy[k] for k in range(4))
        assert area.shape == (6, 8) and np.all(area > 0)       # no cell is turned over or empty
        # no gap and no overlap: together they are the octagon through the outer points
        assert area.sum() == pytest.approx(8 / 2 * np.sin(2 * np.pi / 8), rel=1e-13)

    def test_cells_fill_the_ball_once(self, tmp_path):
        file = VtsFile(write_vtk(tmp_path / "ball.vts", ball(6, 6, 8)))
        volume = cell_volumes(file.points)
        assert volume.shape == (6, 7, 8) and np.all(volume > 0)
        # a polyhedron inscribed in the unit ball: no hole at the center or along the z axis
        assert 0.75 < volume.sum() / (4 / 3 * np.pi) < 1
        shell = SphericalGrid(chebyshev(4, 0.5, 1), colatitude(6), periodic(8))
        hollow = cell_volumes(VtsFile(write_vtk(tmp_path / "shell.vts", shell)).points)
        assert hollow.sum() == pytest.approx(volume.sum() * (1 - 0.5 ** 3), rel=1e-12)

    def test_unbounded_domain(self, tmp_path):
        from numgrids import infinite, semi_infinite
        grid = Grid(x=infinite(8, scale=2.0), y=semi_infinite(6))
        X, Y = grid.meshed_coords
        file = VtsFile(write_vtk(tmp_path / "plane.vts", grid, u=np.exp(-X ** 2 - Y)))
        assert np.all(np.isfinite(file.flat_points))
        npt.assert_array_equal(file.xyz[0][..., 0], X)

    def test_small_hole_remains_without_radial_and_colatitude_axes(self, tmp_path):
        eps = 0.01
        grid = SphericalGrid(chebyshev(4, eps, 1), chebyshev(5, eps, np.pi - eps), periodic(8))
        file = VtsFile(write_vtk(tmp_path / "shell.vts", grid))
        assert file.extent == "0 3 0 4 0 8"                    # only the periodic direction is closed
        x, y, z = file.xyz
        assert np.sqrt(x ** 2 + y ** 2 + z ** 2).min() == pytest.approx(eps)

    def test_grid_without_cartesian_mapping(self, tmp_path):
        grid = CurvilinearGrid(chebyshev(4, 1, 2), periodic(8), scale_factors=(lambda c: 1 + 0 * c[0], lambda c: c[0]))
        with pytest.raises(ValueError, match="no Cartesian mapping"):
            write_vtk(tmp_path / "g.vts", grid)
        assert not (tmp_path / "g.vts").exists()


# ── vector fields ───────────────────────────────────────────────────────


class TestVectorFields:

    def test_rigid_rotation_on_disk(self, tmp_path):
        grid = disk()
        R, Phi = grid.meshed_coords
        file = VtsFile(write_vtk(tmp_path / "rot.vts", grid, v=(np.zeros(grid.shape), R)))
        element = file.point_data.find("DataArray")
        assert element.get("Name") == "v" and element.get("NumberOfComponents") == "3"
        assert file.point_data.get("Vectors") == "v"
        x, y, _ = file.xyz
        v = file["v"]
        # e_phi r = (-y, x, 0), also at the center and on the closing line
        npt.assert_allclose(v[..., 0], -y, atol=1e-14)
        npt.assert_allclose(v[..., 1], x, atol=1e-14)
        assert not v[..., 2].any()

    def test_radial_field_on_disk(self):
        grid = disk()
        R, Phi = grid.meshed_coords
        X, Y = grid.cartesian_coords
        vx, vy = cartesian_components(grid, R, np.zeros(grid.shape))
        npt.assert_allclose(vx, X, atol=1e-15)
        npt.assert_allclose(vy, Y, atol=1e-15)

    def test_constant_field_on_ball(self, tmp_path):
        grid = ball()
        _, Theta, Phi = grid.meshed_coords
        # (1, 0, 2) in spherical components
        v = (np.sin(Theta) * np.cos(Phi) + 2 * np.cos(Theta),
             np.cos(Theta) * np.cos(Phi) - 2 * np.sin(Theta),
             -np.sin(Phi))
        for component, value in zip(cartesian_components(grid, *v), (1, 0, 2)):
            npt.assert_allclose(component, value, atol=1e-14)
        file = VtsFile(write_vtk(tmp_path / "const.vts", grid, v=v))
        # the same vector at the center and on the z axis, where the unit vectors are not defined
        npt.assert_allclose(file["v"], np.broadcast_to([1, 0, 2], file.dims + (3,)), atol=1e-13)

    def test_tangential_field_on_sphere(self, tmp_path):
        grid = SphericalSurfaceGrid(colatitude(6), periodic(8))
        Theta, _ = grid.meshed_coords
        file = VtsFile(write_vtk(tmp_path / "wind.vts", grid, wind=(np.zeros(grid.shape), np.sin(Theta))))
        x, y, _ = file.xyz
        wind = file["wind"]                                    # rotation about the z axis
        npt.assert_allclose(wind[..., 0], -y, atol=1e-13)
        npt.assert_allclose(wind[..., 1], x, atol=1e-13)
        npt.assert_allclose(wind[..., 2], 0, atol=1e-13)

    def test_mapping_that_is_not_analytic(self):
        # abs() defeats complex-step differentiation; finite differences take over
        grid = CurvilinearGrid(
            chebyshev(5, 0.5, 2), periodic(8),
            scale_factors=(lambda c: np.ones_like(c[0]), lambda c: c[0]),
            cartesian=lambda c: (np.abs(c[0]) * np.cos(c[1]), np.abs(c[0]) * np.sin(c[1])),
        )
        R, _ = grid.meshed_coords
        X, Y = grid.cartesian_coords
        vx, vy = cartesian_components(grid, R, R)
        npt.assert_allclose(vx, X - Y, atol=1e-8)
        npt.assert_allclose(vy, Y + X, atol=1e-8)

    def test_no_direction_at_a_singular_grid_point(self):
        grid = PolarGrid(chebyshev(5, 0, 1), periodic(8))      # r = 0 is a grid point
        ones = np.ones(grid.shape)
        vx, vy = cartesian_components(grid, 0 * ones, ones)
        assert np.all(np.isfinite(vx)) and not vx[0].any() and not vy[0].any()
        npt.assert_allclose(np.hypot(vx, vy)[1:], 1, atol=1e-14)

    def test_wrong_number_of_components(self, tmp_path):
        grid = disk()
        with pytest.raises(ValueError, match="Expected 2 vector components, got 3"):
            write_vtk(tmp_path / "v.vts", grid, v=(np.zeros(grid.shape),) * 3)
        with pytest.raises(ValueError, match="does not match"):
            cartesian_components(grid, np.zeros(3), np.zeros(3))


# ── complex fields ──────────────────────────────────────────────────────


class TestComplexFields:

    def test_scalar(self, tmp_path):
        grid = disk()
        X, Y = grid.cartesian_coords
        psi = (X + 1j * Y) ** 2 + 0.5
        file = VtsFile(write_vtk(tmp_path / "psi.vts", grid, psi=psi))
        assert file.names == ["psi_real", "psi_imag", "psi_abs"]
        npt.assert_array_equal(file["psi_real"][1:, :-1, 0], psi.real)
        npt.assert_array_equal(file["psi_imag"][1:, :-1, 0], psi.imag)
        npt.assert_array_equal(file["psi_abs"][1:, :-1, 0], np.abs(psi))
        npt.assert_allclose(file["psi_real"][0], 0.5, atol=1e-13)
        npt.assert_allclose(file["psi_imag"][0], 0, atol=1e-13)
        npt.assert_allclose(file["psi_abs"][0], 0.5, atol=1e-13)

    def test_vector(self, tmp_path):
        grid = Grid(x=chebyshev(4, 0, 1), y=chebyshev(3, 0, 1))
        X, Y = grid.meshed_coords
        file = VtsFile(write_vtk(tmp_path / "E.vts", grid, E=(X + 2j * Y, Y)))
        assert file.names == ["E_real", "E_imag"]
        npt.assert_array_equal(file["E_real"][..., 0, :2], np.stack([X, Y], axis=-1))
        npt.assert_array_equal(file["E_imag"][..., 0, :2], np.stack([2 * Y, 0 * Y], axis=-1))

    def test_name_clash(self, tmp_path):
        grid = Grid(x=chebyshev(4, 0, 1))
        with pytest.raises(ValueError, match="Duplicate array names.*psi_real"):
            write_vtk(tmp_path / "c.vts", grid, psi=1j * grid.coords, psi_real=grid.coords)


# ── time series ─────────────────────────────────────────────────────────


class TestSeries:

    def test_collection(self, tmp_path):
        grid = disk()
        X, Y = grid.cartesian_coords
        t = np.array([0.0, 0.25, 1.5])
        frames = np.array([smooth(X, Y) * np.exp(-tk) for tk in t])
        path = write_vtk_series(tmp_path / "run.pvd", grid, frames, t, name="T")
        assert path == tmp_path / "run.pvd"
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "run.pvd", "run_0000.vts", "run_0001.vts", "run_0002.vts"]
        entries = read_pvd(path)
        assert entries == [(0.0, "run_0000.vts"), (0.25, "run_0001.vts"), (1.5, "run_0002.vts")]
        for k, (_, name) in enumerate(entries):
            file = VtsFile(tmp_path / name)
            assert file.names == ["T"] and file.extent == "0 6 0 8 0 0"
            npt.assert_array_equal(file["T"][1:, :-1, 0], frames[k])
            npt.assert_array_equal(file.points[1:, :-1, 0, 0], X)

    def test_several_fields_and_vectors(self, tmp_path):
        grid = disk()
        R, _ = grid.meshed_coords
        nt = 2
        u = np.array([k + R for k in range(nt)])
        omega = np.array([1.0, 3.0])
        v = (np.zeros((nt,) + grid.shape), omega[:, None, None] * R)
        path = grid.to_vtk_series(tmp_path / "flow", u=u, v=v, binary=False)
        assert path == tmp_path / "flow.pvd"
        entries = read_pvd(path)
        assert [time for time, _ in entries] == [0.0, 1.0]         # frame numbers without t
        for k, (_, name) in enumerate(entries):
            file = VtsFile(tmp_path / name)
            assert file.names == ["u", "v"]
            x, y, _ = file.xyz
            npt.assert_array_equal(file["u"][1:, :-1, 0], u[k])
            npt.assert_allclose(file["v"][..., 0], -omega[k] * y, atol=1e-13)
            npt.assert_allclose(file["v"][..., 1], omega[k] * x, atol=1e-13)

    def test_many_frames_are_numbered_in_order(self, tmp_path):
        grid = Grid(x=chebyshev(3, 0, 1))
        path = write_vtk_series(tmp_path / "long.pvd", grid, np.zeros((12, 3)))
        names = [name for _, name in read_pvd(path)]
        assert names == sorted(names) and names[-1] == "long_0011.vts"

    def test_grid_method(self, tmp_path):
        grid = Grid(x=chebyshev(3, 0, 1))
        frames = np.arange(6.0).reshape(2, 3)
        path = grid.to_vtk_series(tmp_path / "run.pvd", frames, [0.5, 2.5], name="T", close=False)
        entries = read_pvd(path)
        assert [time for time, _ in entries] == [0.5, 2.5]
        file = VtsFile(tmp_path / entries[1][1])
        assert file.names == ["T"]
        npt.assert_array_equal(file.flat["T"], frames[1])

    def test_result_of_evolve(self, tmp_path):
        grid = Grid(x=chebyshev(10, 0, 1))
        result = evolve(grid, 0.1 * grid.laplacian_operator(), np.sin(np.pi * grid.coords), (0, 1),
                        bcs=0, method="BDF", t_eval=[0, 0.5, 1])
        path = result.to_vtk(tmp_path / "heat.pvd", name="T")
        entries = read_pvd(path)
        npt.assert_array_equal([time for time, _ in entries], result.t)
        for k, (_, name) in enumerate(entries):
            npt.assert_array_equal(VtsFile(tmp_path / name).flat["T"], result.u[k])
        default = result.to_vtk(tmp_path / "default", binary=False)
        assert VtsFile(tmp_path / read_pvd(default)[0][1]).names == ["u"]

    def test_result_of_a_system(self, tmp_path):
        grid = Grid(x=chebyshev(10, 0, 1))
        x = grid.coords
        L = grid.laplacian_operator()
        result = evolve(grid, [[None, 1], [L, None]], (np.sin(np.pi * x), np.zeros_like(x)), (0, 0.2),
                        bcs=0, t_eval=[0, 0.2])
        numbered = result.to_vtk(tmp_path / "wave")
        file = VtsFile(tmp_path / read_pvd(numbered)[1][1])
        assert file.names == ["u0", "u1"]
        npt.assert_array_equal(file.flat["u1"], result.u[1][1])
        named = result.to_vtk(tmp_path / "named", name=("u", "t"))      # also names like "t"
        file = VtsFile(tmp_path / read_pvd(named)[1][1])
        assert file.names == ["u", "t"]
        npt.assert_array_equal(file.flat["t"], result.u[1][1])
        for name in (("u",), ("u", "u"), ("a", "b", "c")):
            with pytest.raises(ValueError, match="2 different names"):
                result.to_vtk(tmp_path / "bad", name=name)

    def test_complex_result(self, tmp_path):
        grid = Grid(x=periodic(8))
        psi0 = np.exp(1j * grid.coords)
        result = evolve(grid, 0.5j * grid.laplacian_operator(), psi0, (0, 1), t_eval=[0, 1], method="expm")
        path = result.to_vtk(tmp_path / "psi", name="psi")
        file = VtsFile(tmp_path / read_pvd(path)[1][1])
        assert file.names == ["psi_real", "psi_imag", "psi_abs"]
        npt.assert_array_equal(file.flat["psi_real"][:-1], result.u[1].real)
        npt.assert_allclose(file.flat["psi_abs"], 1, atol=1e-12)

    def test_result_without_grid(self, tmp_path):
        result = EvolutionResult(t=np.arange(2.0), u=np.zeros((2, 4)), success=True, message="")
        with pytest.raises(ValueError, match="does not know its grid"):
            result.to_vtk(tmp_path / "x.pvd")

    def test_wrong_input(self, tmp_path):
        grid = Grid(x=chebyshev(4, 0, 1))
        frames = np.zeros((3, 4))
        with pytest.raises(ValueError, match="expected \\(nt, \\*grid.shape\\)"):
            write_vtk_series(tmp_path / "a.pvd", grid, np.zeros((3, 5)))
        with pytest.raises(ValueError, match="3 time values"):
            write_vtk_series(tmp_path / "a.pvd", grid, frames, [0, 1])
        with pytest.raises(ValueError, match="same number of frames"):
            write_vtk_series(tmp_path / "a.pvd", grid, frames, w=np.zeros((2, 4)))
        with pytest.raises(ValueError, match="given twice"):
            write_vtk_series(tmp_path / "a.pvd", grid, frames, u=frames)
        with pytest.raises(ValueError, match="at least one field"):
            write_vtk_series(tmp_path / "a.pvd", grid)
        with pytest.raises(ValueError, match="suffix '.pvd'"):
            write_vtk_series(tmp_path / "a.vts", grid, frames)
        assert list(tmp_path.iterdir()) == []                  # nothing is written on errors


# ── API ─────────────────────────────────────────────────────────────────


class TestInterface:

    def test_grid_method_and_suffix(self, tmp_path):
        grid = Grid(x=chebyshev(4, 0, 1))
        path = grid.to_vtk(str(tmp_path / "data"), u=grid.coords)
        assert path == tmp_path / "data.vts"
        npt.assert_array_equal(VtsFile(path).flat["u"], grid.coords)
        assert grid.to_vtk(tmp_path / "run-1.5", u=grid.coords).name == "run-1.5.vts"

    def test_wrong_suffix(self, tmp_path):
        grid = Grid(x=chebyshev(4, 0, 1))
        with pytest.raises(ValueError, match="suffix '.vts'"):
            grid.to_vtk(tmp_path / "data.vtk", u=grid.coords)

    def test_more_than_three_dimensions(self, tmp_path):
        axis = chebyshev(2, 0, 1)
        grid = Grid(axis, axis, axis, axis)
        with pytest.raises(ValueError, match="at most 3 dimensions, but this grid has 4"):
            grid.to_vtk(tmp_path / "g.vts")
        with pytest.raises(ValueError, match="at most 3 dimensions"):
            grid.to_vtk_series(tmp_path / "g.pvd", np.zeros((2,) + grid.shape))

    def test_wrong_shape(self, tmp_path):
        grid = Grid(x=chebyshev(4, 0, 1), y=chebyshev(3, 0, 1))
        with pytest.raises(ValueError, match="Array 'u' has shape \\(4,\\).*tuple"):
            grid.to_vtk(tmp_path / "g.vts", u=np.zeros(4))
        with pytest.raises(ValueError, match="v\\[1\\]"):
            grid.to_vtk(tmp_path / "g.vts", v=(np.zeros((4, 3)), np.zeros(4)))


# ── reading with VTK itself ─────────────────────────────────────────────


def test_read_with_vtk(tmp_path):
    """Only runs where the ``vtk`` package is installed."""
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    grid = disk()
    R, _ = grid.meshed_coords
    X, Y = grid.cartesian_coords
    u = smooth(X, Y)
    for binary in (True, False):
        path = write_vtk(tmp_path / f"disk_{binary}.vts", grid, binary=binary, u=u, v=(np.zeros(grid.shape), R))
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        data = reader.GetOutput()
        dims = [0, 0, 0]
        data.GetDimensions(dims)                  # the signature without arguments was removed in VTK 9
        assert dims == [7, 9, 1]
        expected = VtsFile(path)
        npt.assert_array_equal(vtk_to_numpy(data.GetPoints().GetData()), expected.flat_points)
        npt.assert_array_equal(vtk_to_numpy(data.GetPointData().GetArray("u")), expected.flat["u"])
        npt.assert_array_equal(vtk_to_numpy(data.GetPointData().GetArray("v")), expected.flat["v"])
        assert data.GetPointData().GetScalars().GetName() == "u"
        assert data.GetPointData().GetVectors().GetName() == "v"

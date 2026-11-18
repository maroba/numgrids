"""Export of meshed fields to VTK files for ParaView, VisIt and other VTK-based tools.

The files are written directly, without any dependency: VTK XML
*StructuredGrid* files (``.vts``) with the grid points in Cartesian space
and the fields as point data.  Time series are written as one ``.vts`` file
per time step plus a ``.pvd`` collection file that holds the time values.

Functions
---------
write_vtk
    Write meshed fields on a grid to a ``.vts`` file.
write_vtk_series
    Write a time series as ``.vts`` files with a ``.pvd`` collection file.
cartesian_components
    Convert the components of a vector field to the Cartesian basis.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

import numpy as np
from numpy.typing import NDArray

from numgrids.axes import ChebyshevAxis, ColatitudeAxis, EquidistantAxis, RadialAxis
from numgrids.interpol import interpolation_matrix

if TYPE_CHECKING:
    from numgrids.grids import Grid

#: Points closer than this (relative to the size of the domain) are the same point.
_COINCIDENT = 1e-9


# -- geometry ----------------------------------------------------------------

def _cartesian_map(grid: Grid) -> Callable | None:
    """The map from meshed coordinates to Cartesian coordinates (``None``: the identity)."""
    from numgrids.curvilinear import CurvilinearGrid

    fn = getattr(grid, "_cartesian_fn", None)
    if fn is None and isinstance(grid, CurvilinearGrid):
        raise ValueError(
            "VTK export needs the Cartesian coordinates of the grid points, but this grid has "
            "no Cartesian mapping. Pass `cartesian=` to CurvilinearGrid."
        )
    return fn


def _origin_weights(axis: RadialAxis) -> NDArray:
    """Weights *w* of the spectral interpolant at the origin of a radial axis.

    Along a diameter the value at the center is
    :math:`\\sum_k w_k\\,[f(r_k) + f(-r_k)]`: the weights of the two halves
    of the diameter are equal by symmetry.
    """
    n = len(axis)
    diameter = ChebyshevAxis(2 * n, -axis.high, axis.high)
    return interpolation_matrix(diameter, np.array([0.0]))[0, n:]


def _pole_weights(axis: ColatitudeAxis) -> tuple[NDArray, NDArray]:
    """Weights of the spectral interpolant at the poles ``theta = 0`` and ``theta = pi``.

    As for :func:`_origin_weights`, with the great circle through the pole
    in place of the diameter.
    """
    n = len(axis)
    low = axis.great_circle_coords[0]
    circle = EquidistantAxis(2 * n, low, low + 2 * np.pi, periodic=True)
    # theta = 0 is the same point of the circle as theta = 2 pi, inside [low, low + 2 pi)
    A = interpolation_matrix(circle, np.array([2 * np.pi, np.pi]))
    return A[0, :n], A[1, :n]


class _Fill:
    """Points added at the low and/or high end of a folded axis (origin, poles)."""

    def __init__(self, axis: int, low: NDArray | None, high: NDArray | None) -> None:
        self.axis = axis
        self.low = low
        self.high = high
        #: axes along which the added points coincide in space, per end
        self.collapsed: dict[str, tuple[int, ...]] = {}


class _Mesh:
    """The points written to a file, and the map from grid arrays to their values.

    With *close*, the grid is completed for display: periodic directions
    are closed by repeating their first point, and the origin of a
    :class:`~numgrids.axes.RadialAxis` and the poles of a
    :class:`~numgrids.axes.ColatitudeAxis` (which are not grid points) are
    added.  The original grid points keep their values.
    """

    def __init__(self, grid: Grid, close: bool = True) -> None:
        if grid.ndims > 3:
            raise ValueError(
                f"VTK files hold grids with at most 3 dimensions, but this grid has {grid.ndims}. "
                "Export a slice with fewer dimensions instead."
            )
        self.grid = grid
        self._map = _cartesian_map(grid)
        self._units: list[tuple[NDArray, ...]] | None = None
        self._periodic: list[int] = []
        origins: list[_Fill] = []
        poles: list[_Fill] = []

        coords = [np.asarray(axis.coords, dtype=float) for axis in grid.axes]
        if close:
            for i, axis in enumerate(grid.axes):
                if isinstance(axis, RadialAxis):
                    coords[i] = np.concatenate([[0.0], coords[i]])
                    origins.append(_Fill(i, _origin_weights(axis), None))
                elif isinstance(axis, ColatitudeAxis):
                    coords[i] = np.concatenate([[0.0], coords[i], [np.pi]])
                    poles.append(_Fill(i, *_pole_weights(axis)))
                elif axis.periodic:
                    coords[i] = np.concatenate([coords[i], [axis.low + axis.period]])
                    self._periodic.append(i)
        # Poles before the origin: the value at the origin is then set for all
        # directions at once, including those of the poles.
        self._fills = poles + origins
        self.shape = tuple(len(c) for c in coords)

        meshed = tuple(np.meshgrid(*coords, indexing="ij"))
        cart = meshed if self._map is None else tuple(self._map(meshed))
        if len(cart) > 3:
            raise ValueError(f"The Cartesian mapping of the grid has {len(cart)} components, expected at most 3.")
        points = [np.array(np.broadcast_to(np.asarray(x, dtype=float), self.shape)) for x in cart]
        size = max((float(np.ptp(x)) for x in points), default=0.0) or 1.0
        tol = _COINCIDENT * size

        # The added points of an origin or pole are one point in space, repeated along
        # the angular directions: find these directions and make the copies identical.
        for fill in self._fills:
            for end, index in (("low", 0), ("high", self.shape[fill.axis] - 1)):
                if getattr(fill, end) is None:
                    continue
                slab = [np.take(x, [index], axis=fill.axis) for x in points]
                collapsed = tuple(
                    j for j in range(grid.ndims)
                    if j != fill.axis and all(float(np.ptp(x, axis=j).max()) <= tol for x in slab)
                )
                fill.collapsed[end] = collapsed
                if collapsed:
                    where = [slice(None)] * grid.ndims
                    where[fill.axis] = slice(index, index + 1)
                    first = tuple(slice(0, 1) if j in collapsed else slice(None) for j in range(grid.ndims))
                    for x, s in zip(points, slab):
                        x[tuple(where)] = s[first]
        # On curvilinear grids the closing point of a periodic direction is the first
        # point again (on Cartesian grids it lies one period further).
        for i in self._periodic:
            first = [np.take(x, [0], axis=i) for x in points]
            last = [np.take(x, [self.shape[i] - 1], axis=i) for x in points]
            if all(float(np.abs(a - b).max()) <= tol for a, b in zip(first, last)):
                where = [slice(None)] * grid.ndims
                where[i] = slice(self.shape[i] - 1, self.shape[i])
                for x, a in zip(points, first):
                    x[tuple(where)] = a
        self.points = tuple(points)

    @property
    def extent(self) -> str:
        """The index range of the points, in the notation of VTK."""
        counts = self.shape + (1,) * (3 - len(self.shape))
        return " ".join(f"0 {n - 1}" for n in counts)

    def extend(self, f: NDArray) -> NDArray:
        """Values of a scalar field of grid shape on the points of the mesh."""
        for fill in self._fills:
            parts = [f]
            for end, weights in (("low", fill.low), ("high", fill.high)):
                if weights is None:
                    continue
                # Factor 2: both halves of the diameter (great circle) through the point.
                # Averaged over the directions, the far half contributes as much as the near one.
                slab = np.expand_dims(2 * np.tensordot(weights, f, axes=([0], [fill.axis])), fill.axis)
                collapsed = fill.collapsed[end]
                if collapsed:
                    slab = np.broadcast_to(slab.mean(axis=collapsed, keepdims=True), slab.shape)
                if end == "low":
                    parts.insert(0, slab)
                else:
                    parts.append(slab)
            f = np.concatenate(parts, axis=fill.axis)
        for i in self._periodic:
            f = np.concatenate([f, np.take(f, [0], axis=i)], axis=i)
        return f

    # -- vector fields -------------------------------------------------------

    def unit_vectors(self) -> list[tuple[NDArray, ...]]:
        """Cartesian components of the unit vectors along the coordinate lines, per axis."""
        if self._units is None:
            grid = self.grid
            scale = getattr(grid, "scale_factors", None)
            units = []
            for i in range(grid.ndims):
                try:
                    tangent = self._tangent(i, complex_step=True)
                    norm = np.sqrt(sum(x ** 2 for x in tangent))
                    # The length of the tangent vector is the scale factor. If it is not,
                    # the mapping is not an analytic function of the coordinates.
                    reliable = scale is None or np.allclose(norm, scale[i], rtol=1e-6, atol=1e-12)
                except (TypeError, ValueError):
                    reliable = False
                if not reliable:
                    tangent = self._tangent(i, complex_step=False)
                    norm = np.sqrt(sum(x ** 2 for x in tangent))
                with np.errstate(divide="ignore", invalid="ignore"):
                    # No direction where the coordinate line degenerates to a point (e.g. at r = 0).
                    units.append(tuple(np.where(norm > 0, x / norm, 0.0) for x in tangent))
            self._units = units
        return self._units

    def _tangent(self, i: int, complex_step: bool) -> list[NDArray]:
        """The tangent vector dx/dq_i of the coordinate lines of axis *i* on the grid."""
        grid = self.grid

        def mapped(shift, dtype):
            coords = list(grid.meshed_coords)
            coords[i] = coords[i] + shift
            return [np.broadcast_to(np.asarray(x, dtype=dtype), grid.shape) for x in self._map(tuple(coords))]

        if complex_step:
            step = 1e-30
            return [np.imag(x) / step for x in mapped(1j * step, complex)]
        step = 1e-6 * np.maximum(1.0, np.abs(grid.meshed_coords[i]))
        return [(a - b) / (2 * step) for a, b in zip(mapped(step, float), mapped(-step, float))]

    def cartesian_components(self, v) -> tuple[NDArray, ...]:
        """Cartesian components of a vector field given by its components along the axes."""
        grid = self.grid
        if len(v) != grid.ndims:
            raise ValueError(f"Expected {grid.ndims} vector components, got {len(v)}.")
        v = [grid._check_field(vi, f"v[{i}]") for i, vi in enumerate(v)]
        if self._map is None:
            return tuple(v)
        units = self.unit_vectors()
        return tuple(sum(vi * e[k] for vi, e in zip(v, units)) for k in range(len(units[0])))


def cartesian_components(grid: Grid, *v: NDArray) -> tuple[NDArray, ...]:
    r"""Convert the components of a vector field to the Cartesian basis.

    On curvilinear grids vector fields are given by their physical
    components along the coordinate lines, e.g. :math:`(v_r, v_\varphi)`.
    This function returns the components :math:`(v_x, v_y, \ldots)` of the
    same field, one per Cartesian coordinate of the grid.  The unit vectors
    are derived from the Cartesian mapping of the grid.

    Parameters
    ----------
    grid : Grid
        The grid.  Curvilinear grids need a Cartesian mapping.
    *v : NDArray
        The components of the vector field, one array of grid shape per axis.

    Returns
    -------
    tuple of NDArray
        The Cartesian components (the arrays as given on a Cartesian grid).

    Examples
    --------
    A rigid rotation :math:`\mathbf{v} = r\,\mathbf{e}_\varphi` is
    :math:`(-y, x)`:

    >>> import numpy as np
    >>> from numgrids import PolarGrid, radial, periodic
    >>> disk = PolarGrid(radial(8), periodic(16))
    >>> R, Phi = disk.meshed_coords
    >>> X, Y = disk.cartesian_coords
    >>> vx, vy = cartesian_components(disk, np.zeros(disk.shape), R)
    >>> bool(np.allclose(vx, -Y) and np.allclose(vy, X))
    True
    """
    return _Mesh(grid, close=False).cartesian_components(v)


# -- encoding ----------------------------------------------------------------

def _data_array(name: str, data: NDArray, components: int, binary: bool) -> str:
    """A ``DataArray`` element; *data* holds the values in the order of the file."""
    data = np.ascontiguousarray(data, dtype="<f8")
    if binary:
        # Uncompressed binary data: the number of bytes, then the bytes, as one base64 string.
        raw = data.tobytes()
        text = base64.b64encode(np.array(len(raw), dtype="<u8").tobytes() + raw).decode("ascii")
    else:
        # repr() of a float is the shortest text that reads back as the same number
        values = data.tolist()
        per_line = components if components > 1 else 6
        text = "\n".join(" ".join(map(repr, values[k:k + per_line])) for k in range(0, len(values), per_line))
    return (
        f'<DataArray type="Float64" Name={quoteattr(name)} NumberOfComponents="{components}" '
        f'format="{"binary" if binary else "ascii"}">\n{text}\n</DataArray>\n'
    )


def _flat(a: NDArray) -> NDArray:
    """Values in the point order of VTK: the first index runs fastest."""
    return np.real(a).ravel(order="F")


def _vector(components) -> NDArray:
    """Three components per point, interleaved; missing components are zero."""
    flat = [_flat(c) for c in components]
    flat += [np.zeros_like(flat[0])] * (3 - len(flat))
    return np.stack(flat, axis=1).ravel()


def _field_arrays(mesh: _Mesh, name: str, value) -> list[tuple[str, int, NDArray]]:
    """The data arrays ``(name, number of components, values)`` of one field."""
    grid = mesh.grid
    if not isinstance(name, str) or not name:
        raise ValueError(f"Field names must be non-empty strings, got {name!r}.")
    if isinstance(value, tuple):
        comps = [mesh.extend(c) for c in mesh.cartesian_components(value)]
        if any(np.iscomplexobj(c) for c in comps):
            return [(f"{name}_real", 3, _vector([c.real for c in comps])),
                    (f"{name}_imag", 3, _vector([np.imag(c) for c in comps]))]
        return [(name, 3, _vector(comps))]
    value = np.asarray(value)
    if value.shape != grid.shape:
        raise ValueError(
            f"Array '{name}' has shape {value.shape}, expected grid shape {grid.shape}. "
            "Pass vector fields as a tuple with one array per axis."
        )
    f = mesh.extend(value)
    if np.iscomplexobj(f):
        return [(f"{name}_real", 1, _flat(f.real)), (f"{name}_imag", 1, _flat(f.imag)),
                (f"{name}_abs", 1, _flat(np.abs(f)))]
    return [(name, 1, _flat(f))]


def _points_array(mesh: _Mesh, binary: bool) -> str:
    return _data_array("Points", _vector(mesh.points), 3, binary)


def _write_vts(path: Path, mesh: _Mesh, points: str, fields: dict, binary: bool) -> None:
    arrays = [a for name, value in fields.items() for a in _field_arrays(mesh, name, value)]
    names = [a[0] for a in arrays]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Duplicate array names in the VTK file: {duplicates}.")
    # The first scalar and vector are the ones a viewer shows by default.
    active = ""
    for label, components in (("Scalars", 1), ("Vectors", 3)):
        first = next((n for n, c, _ in arrays if c == components), None)
        if first is not None:
            active += f" {label}={quoteattr(first)}"
    parts = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n',
        f'<StructuredGrid WholeExtent="{mesh.extent}">\n',
        f'<Piece Extent="{mesh.extent}">\n',
        f"<PointData{active}>\n",
        *(_data_array(n, data, c, binary) for n, c, data in arrays),
        "</PointData>\n",
        "<Points>\n",
        points,
        "</Points>\n",
        "</Piece>\n",
        "</StructuredGrid>\n",
        "</VTKFile>\n",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.writelines(parts)


#: Suffixes of other kinds of VTK files; a reader would take the file for one of those.
_OTHER_SUFFIXES = (".vtk", ".vts", ".vtr", ".vti", ".vtu", ".vtp", ".vtm", ".pvd")


def _with_suffix(path: str | Path, suffix: str, what: str) -> Path:
    path = Path(path)
    given = path.suffix.lower()
    if given == suffix:
        return path
    if given in _OTHER_SUFFIXES:
        raise ValueError(f"{what} need the suffix '{suffix}', got '{path.name}'.")
    return path.with_name(path.name + suffix)


# -- public API --------------------------------------------------------------

def write_vtk(path: str | Path, grid: Grid, *, binary: bool = True, close: bool = True, **fields) -> Path:
    r"""Write meshed fields on a grid to a VTK file for ParaView or VisIt.

    The file is a VTK XML *StructuredGrid* file (``.vts``).  It holds the
    grid points in Cartesian space (for curvilinear grids via their
    Cartesian mapping; 1D and 2D grids lie in the plane :math:`z = 0`
    unless the mapping embeds them in space) and the fields as point data.

    For display, the grid is completed (*close*): periodic directions are
    closed by repeating their first point, so that there is no gap, and
    the origin of a :func:`~numgrids.radial` axis and the poles of a
    :func:`~numgrids.colatitude` axis are added with values from the
    spectral interpolation.  The values at the grid points are written
    unchanged.

    Parameters
    ----------
    path : str or Path
        Destination file.  The suffix ``.vts`` is appended if there is none.
    grid : Grid
        The grid (at most 3 dimensions).
    binary : bool
        Write the numbers base64-encoded (default) instead of as text.
        Both hold the full double precision.
    close : bool
        Complete the grid for display as described above (default).  With
        ``False``, the file holds exactly the grid points.
    **fields : NDArray or tuple of NDArray
        The fields by name.  An array of grid shape is a scalar field.  A
        tuple with one array per axis is a vector field, given by its
        components along the axes; it is written with its Cartesian
        components (see :func:`cartesian_components`).  A complex scalar
        field ``psi`` is written as the arrays ``psi_real``, ``psi_imag``
        and ``psi_abs``, a complex vector field as ``E_real`` and ``E_imag``.

    Returns
    -------
    Path
        The path of the file written.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from numgrids import PolarGrid, radial, periodic
    >>> disk = PolarGrid(radial(8), periodic(16))
    >>> R, Phi = disk.meshed_coords
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = write_vtk(os.path.join(folder, "disk"), disk,
    ...                      u=R ** 2 * np.cos(2 * Phi), v=(np.zeros(disk.shape), R))
    ...     path.name
    'disk.vts'
    """
    path = _with_suffix(path, ".vts", "VTK StructuredGrid files")
    mesh = _Mesh(grid, close)
    _write_vts(path, mesh, _points_array(mesh, binary), fields, binary)
    return path


def write_vtk_series(
    path: str | Path,
    grid: Grid,
    frames=None,
    t=None,
    *,
    name: str = "u",
    binary: bool = True,
    close: bool = True,
    **fields,
) -> Path:
    """Write a time series of fields as VTK files with a ``.pvd`` collection file.

    Every time step is written to its own ``.vts`` file (see
    :func:`write_vtk`) next to the collection file: ``run.pvd`` comes with
    ``run_0000.vts``, ``run_0001.vts``, ...  Open the ``.pvd`` file in
    ParaView to get the series with its time values.

    Parameters
    ----------
    path : str or Path
        The collection file.  The suffix ``.pvd`` is appended if there is none.
    grid : Grid
        The grid (at most 3 dimensions).
    frames : NDArray or tuple of NDArray, optional
        The values of one field in time, shape ``(nt, *grid.shape)`` (a
        tuple of such arrays for the components of a vector field).  It is
        stored under *name*.
    t : array_like, optional
        The time values, shape ``(nt,)``.  Defaults to the frame numbers.
    name : str
        Name of the field given by *frames* (default ``"u"``).
    binary, close : bool
        As for :func:`write_vtk`.
    **fields : NDArray or tuple of NDArray
        Further fields by name, each given like *frames*.

    Returns
    -------
    Path
        The path of the collection file.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev
    >>> grid = Grid(x=chebyshev(8, 0, 1), y=chebyshev(8, 0, 1))
    >>> X, Y = grid.meshed_coords
    >>> t = np.linspace(0, 1, 3)
    >>> frames = np.array([np.exp(-tk) * X * Y for tk in t])
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = write_vtk_series(os.path.join(folder, "run.pvd"), grid, frames, t)
    ...     sorted(os.listdir(folder))
    ['run.pvd', 'run_0000.vts', 'run_0001.vts', 'run_0002.vts']
    """
    if frames is not None:
        if name in fields:
            raise ValueError(f"The field name {name!r} is given twice.")
        fields = {name: frames, **fields}
    return _write_series(path, grid, fields, t, binary, close)


def _write_series(path: str | Path, grid: Grid, fields: dict, t, binary: bool = True, close: bool = True) -> Path:
    path = _with_suffix(path, ".pvd", "VTK collection files")
    if not fields:
        raise ValueError("A time series needs at least one field.")
    series = {}
    for field_name, value in fields.items():
        vector = isinstance(value, tuple)
        comps = [np.asarray(c) for c in (value if vector else [value])]
        for c in comps:
            if c.shape[1:] != grid.shape:
                raise ValueError(
                    f"The frames of '{field_name}' have shape {c.shape}, expected (nt, *grid.shape) "
                    f"with grid shape {grid.shape}."
                )
        series[field_name] = (vector, comps)
    counts = {len(c) for _, comps in series.values() for c in comps}
    if len(counts) != 1:
        raise ValueError(f"All fields need the same number of frames, got {sorted(counts)}.")
    nt = counts.pop()
    times = np.arange(nt, dtype=float) if t is None else np.asarray(t, dtype=float)
    if times.shape != (nt,):
        raise ValueError(f"Expected {nt} time values (one per frame), got an array of shape {times.shape}.")

    mesh = _Mesh(grid, close)
    points = _points_array(mesh, binary)  # the same in all files
    digits = max(4, len(str(nt - 1)))
    entries = []
    for k in range(nt):
        frame = {n: tuple(c[k] for c in comps) if vector else comps[0][k]
                 for n, (vector, comps) in series.items()}
        file = path.with_name(f"{path.stem}_{k:0{digits}d}.vts")
        _write_vts(file, mesh, points, frame, binary)
        entries.append(
            f'<DataSet timestep="{float(times[k])!r}" group="" part="0" file={quoteattr(file.name)}/>\n'
        )
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.writelines([
            '<?xml version="1.0"?>\n',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n',
            "<Collection>\n",
            *entries,
            "</Collection>\n",
            "</VTKFile>\n",
        ])
    return path

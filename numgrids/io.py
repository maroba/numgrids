"""Save and load grids and meshed function data.

Grids are serialized to NumPy ``.npz`` files.  The file stores a compact
JSON description of the grid (type, axis types, axis parameters) together
with any number of user-supplied meshed arrays.

Functions
---------
save_grid
    Persist a grid (and optional data arrays) to a ``.npz`` file.
load_grid
    Reconstruct a grid (and any saved data) from a ``.npz`` file.
to_xarray
    Convert a grid and meshed arrays to an :class:`xarray.Dataset`.
from_xarray
    Reconstruct a grid and data from an xarray object.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numgrids.grids import Grid


# -- axis / grid class registries -------------------------------------------

_AXIS_CLASSES: dict[str, type] = {}
_GRID_CLASSES: dict[str, type] = {}


def _ensure_registries() -> None:
    """Lazily populate the class look-up tables."""
    if _AXIS_CLASSES:
        return
    from numgrids.axes import (
        ChebyshevAxis,
        ColatitudeAxis,
        EquidistantAxis,
        InfiniteAxis,
        LogAxis,
        RadialAxis,
        SemiInfiniteAxis,
    )
    from numgrids.curvilinear import (
        CylindricalGrid,
        EllipticCylindricalGrid,
        OblateSpheroidalGrid,
        PolarGrid,
        ProlateSpheroidalGrid,
        SphericalGrid,
        SphericalSurfaceGrid,
    )
    from numgrids.grids import Grid as _Grid

    _AXIS_CLASSES.update({
        "EquidistantAxis": EquidistantAxis,
        "ChebyshevAxis": ChebyshevAxis,
        "LogAxis": LogAxis,
        "RadialAxis": RadialAxis,
        "ColatitudeAxis": ColatitudeAxis,
        "SemiInfiniteAxis": SemiInfiniteAxis,
        "InfiniteAxis": InfiniteAxis,
    })
    _GRID_CLASSES.update({
        "Grid": _Grid,
        "SphericalGrid": SphericalGrid,
        "CylindricalGrid": CylindricalGrid,
        "PolarGrid": PolarGrid,
        "EllipticCylindricalGrid": EllipticCylindricalGrid,
        "ProlateSpheroidalGrid": ProlateSpheroidalGrid,
        "OblateSpheroidalGrid": OblateSpheroidalGrid,
        "SphericalSurfaceGrid": SphericalSurfaceGrid,
    })


# -- public API --------------------------------------------------------------

def save_grid(path: str | Path, grid: Grid, **data: NDArray) -> None:
    """Save a grid and optional meshed data arrays to a ``.npz`` file.

    The grid structure (axis types, parameters) is stored as compact JSON
    metadata.  Any additional keyword arguments are stored as named arrays
    and will be returned by :func:`load_grid`.

    Parameters
    ----------
    path : str or Path
        Destination file path.  A ``.npz`` suffix is appended automatically
        by NumPy if not already present.
    grid : Grid
        The grid to save.
    **data : NDArray
        Arbitrary meshed arrays to persist alongside the grid.  Each array
        must have shape ``grid.shape``.

    Examples
    --------
    >>> save_grid("my_grid.npz", grid, temperature=T, pressure=P)  # doctest: +SKIP
    """
    _ensure_registries()

    for name, arr in data.items():
        arr = np.asarray(arr)
        if arr.shape != grid.shape:
            raise ValueError(
                f"Array '{name}' has shape {arr.shape}, "
                f"expected grid shape {grid.shape}"
            )

    meta = _grid_to_meta(grid)
    meta_bytes = np.void(json.dumps(meta).encode("utf-8"))

    arrays: dict[str, NDArray] = {"__meta__": np.array(meta_bytes)}
    for name, arr in data.items():
        arrays[f"data_{name}"] = arr

    np.savez(str(path), **arrays)


def load_grid(path: str | Path) -> tuple[Grid, dict[str, NDArray]]:
    """Load a grid and any saved data arrays from a ``.npz`` file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file (the suffix may be omitted).

    Returns
    -------
    grid : Grid
        The reconstructed grid.
    data : dict[str, NDArray]
        Dictionary of data arrays that were saved alongside the grid.
        Keys match the original keyword names passed to :func:`save_grid`.

    Examples
    --------
    >>> grid, data = load_grid("my_grid.npz")  # doctest: +SKIP
    >>> T = data["temperature"]                # doctest: +SKIP
    """
    _ensure_registries()

    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")

    with np.load(str(path), allow_pickle=False) as npz:
        meta_bytes = npz["__meta__"].item()
        if isinstance(meta_bytes, bytes):
            meta = json.loads(meta_bytes.decode("utf-8"))
        else:
            meta = json.loads(str(meta_bytes))

        grid = _meta_to_grid(meta)

        data: dict[str, NDArray] = {}
        prefix = "data_"
        for key in npz.files:
            if key.startswith(prefix):
                data[key[len(prefix):]] = npz[key]

    return grid, data


# -- internal helpers --------------------------------------------------------

def _grid_to_meta(grid: Grid) -> dict:
    """Serialise the grid definition to a JSON-compatible dict."""
    _ensure_registries()
    grid_type = type(grid).__name__
    if grid_type not in _GRID_CLASSES:
        hint = ""
        if grid_type == "CurvilinearGrid":
            hint = " (custom scale factors cannot be serialized; save the axes/data instead)"
        raise TypeError(f"Unsupported grid type: {grid_type}{hint}")

    axes_meta = [_axis_to_meta(axis) for axis in grid.axes]
    meta = {"grid_type": grid_type, "axes": axes_meta}
    params = getattr(grid, "_params", None)
    if params:
        meta["params"] = {k: float(v) for k, v in params.items()}
    return meta


def _axis_to_meta(axis) -> dict:
    axis_type = type(axis).__name__
    if axis_type not in _AXIS_CLASSES:
        raise TypeError(f"Unsupported axis type: {axis_type}")
    return {
        "type": axis_type,
        "num_points": len(axis),
        # Infinite ends are stored as null: JSON has no infinity.
        "low": float(axis.low) if np.isfinite(axis.low) else None,
        "high": float(axis.high) if np.isfinite(axis.high) else None,
        "periodic": axis.periodic,
        "name": axis.name,
        **axis._extra_params(),
    }


def _meta_to_axis(ax: dict):
    axis_cls = _AXIS_CLASSES[ax["type"]]
    kwargs: dict = {}
    if ax.get("periodic"):
        kwargs["periodic"] = True
    if ax.get("name") is not None:
        kwargs["name"] = ax["name"]
    for key in axis_cls._extra_param_names:
        if key in ax:
            kwargs[key] = ax[key]
    low = -np.inf if ax["low"] is None else ax["low"]
    high = np.inf if ax["high"] is None else ax["high"]
    return axis_cls(int(ax["num_points"]), low, high, **kwargs)


def _meta_to_grid(meta: dict) -> Grid:
    """Reconstruct a grid from a metadata dict."""
    _ensure_registries()
    grid_cls = _GRID_CLASSES[meta["grid_type"]]
    axes = [_meta_to_axis(ax) for ax in meta["axes"]]
    return grid_cls(*axes, **meta.get("params", {}))


# -- xarray interoperability --------------------------------------------------

def _require_xarray():
    try:
        import xarray as xr
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "xarray is required for this feature. Install it with "
            "`pip install numgrids[xarray]`."
        ) from e
    return xr


def _dim_names(grid: Grid) -> list[str]:
    return [ax.name if ax.name is not None else f"x{i}" for i, ax in enumerate(grid.axes)]


def to_xarray(grid: Grid, **fields: NDArray):
    """Convert a grid and meshed arrays to an :class:`xarray.Dataset`.

    Each axis becomes a dimension coordinate (named after the axis, or
    ``x0``, ``x1``, ... for unnamed axes).  The full grid definition is
    stored in the attributes, so that :func:`from_xarray` can reconstruct
    the grid exactly -- also after writing to and reading from NetCDF.

    Parameters
    ----------
    grid : Grid
        The grid.
    **fields : NDArray
        Meshed arrays of shape ``grid.shape``, stored as data variables.

    Returns
    -------
    xarray.Dataset

    Examples
    --------
    >>> from numgrids import Grid, chebyshev
    >>> grid = Grid(x=chebyshev(5, 0, 1))
    >>> ds = to_xarray(grid, u=grid.coords ** 2)  # doctest: +SKIP
    >>> ds.to_netcdf("data.nc")                    # doctest: +SKIP
    """
    xr = _require_xarray()
    _ensure_registries()
    dims = _dim_names(grid)
    coords = {}
    for dim, axis in zip(dims, grid.axes):
        attrs = {"numgrids_axis": json.dumps(_axis_to_meta(axis))}
        coords[dim] = xr.Variable(dim, axis.coords, attrs=attrs)
    attrs = {"numgrids_grid": json.dumps(_grid_to_meta(grid))}
    data_vars = {}
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if arr.shape != grid.shape:
            raise ValueError(
                f"Array '{name}' has shape {arr.shape}, expected grid shape {grid.shape}"
            )
        # The grid definition is attached to every variable as well, so
        # that single DataArrays extracted from the dataset keep it.
        data_vars[name] = xr.Variable(dims, arr, attrs=dict(attrs))
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def from_xarray(obj) -> tuple[Grid, dict[str, NDArray]]:
    """Reconstruct a grid and data arrays from an xarray object.

    Works with objects created by :func:`to_xarray` (exact round trip).
    For other datasets, the axis types are inferred from the coordinate
    values (equidistant, Chebyshev or logarithmic; periodic axes cannot be
    detected and are treated as non-periodic, and other axis types such as
    the axes on unbounded domains are not recognized).

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray

    Returns
    -------
    grid : Grid
    data : dict[str, NDArray]
        The data variables, transposed to the grid's axis order.
    """
    xr = _require_xarray()
    _ensure_registries()
    if isinstance(obj, xr.DataArray):
        attrs = dict(obj.attrs)
        obj = obj.to_dataset(name=obj.name if obj.name is not None else "data")
        obj.attrs.update({k: v for k, v in attrs.items() if k == "numgrids_grid"})

    meta_json = obj.attrs.get("numgrids_grid")
    if meta_json is None:
        for var in obj.data_vars.values():
            if "numgrids_grid" in var.attrs:
                meta_json = var.attrs["numgrids_grid"]
                break

    if meta_json is not None:
        meta = json.loads(meta_json)
        grid = _meta_to_grid(meta)
        dims = _dim_names(grid)
    else:
        if not obj.data_vars:
            raise ValueError("Cannot infer a grid from a dataset without data variables.")
        dims = list(next(iter(obj.data_vars.values())).dims)
        axes = []
        for dim in dims:
            var = obj.coords[dim] if dim in obj.coords else None
            if var is None:
                raise ValueError(f"Dimension {dim!r} has no coordinate values.")
            if "numgrids_axis" in var.attrs:
                axes.append(_meta_to_axis(json.loads(var.attrs["numgrids_axis"])))
            else:
                axes.append(_infer_axis(np.asarray(var.values, dtype=float), dim))
        from numgrids.grids import Grid as _Grid
        grid = _Grid(*axes)

    data = {}
    for name, var in obj.data_vars.items():
        if set(var.dims) != set(dims):
            continue
        data[str(name)] = np.asarray(var.transpose(*dims).values)
    return grid, data


def _infer_axis(values: NDArray, name: str):
    from numgrids.axes import ChebyshevAxis, EquidistantAxis, LogAxis

    n = len(values)
    lo, hi = values[0], values[-1]
    candidates = [EquidistantAxis(n, lo, hi, name=name), ChebyshevAxis(n, lo, hi, name=name)]
    if lo > 0:
        candidates.append(LogAxis(n, lo, hi, name=name))
    for ax in candidates:
        if np.allclose(ax.coords, values, rtol=1e-9, atol=1e-12 * max(1.0, abs(hi))):
            return ax
    raise ValueError(
        f"Cannot infer the axis type of coordinate {name!r}: values are neither "
        "equidistant, Chebyshev nodes, nor logarithmically spaced."
    )

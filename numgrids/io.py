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
    from numgrids.axes import EquidistantAxis, ChebyshevAxis, LogAxis
    from numgrids.grids import Grid as _Grid
    from numgrids.api import SphericalGrid, CylindricalGrid, PolarGrid

    _AXIS_CLASSES.update({
        "EquidistantAxis": EquidistantAxis,
        "ChebyshevAxis": ChebyshevAxis,
        "LogAxis": LogAxis,
    })
    _GRID_CLASSES.update({
        "Grid": _Grid,
        "SphericalGrid": SphericalGrid,
        "CylindricalGrid": CylindricalGrid,
        "PolarGrid": PolarGrid,
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
    >>> save_grid("my_grid.npz", grid, temperature=T, pressure=P)
    """
    _ensure_registries()

    for name, arr in data.items():
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
    >>> grid, data = load_grid("my_grid.npz")
    >>> T = data["temperature"]
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
    grid_type = type(grid).__name__
    if grid_type not in _GRID_CLASSES:
        raise TypeError(f"Unsupported grid type: {grid_type}")

    axes_meta = []
    for axis in grid.axes:
        axis_type = type(axis).__name__
        if axis_type not in _AXIS_CLASSES:
            raise TypeError(f"Unsupported axis type: {axis_type}")
        # For periodic axes the endpoint is excluded from the coordinate
        # array, so coords[-1] != high.  Recover the true domain endpoint
        # from the uniform spacing: high = coords[-1] + spacing.
        if axis.periodic and len(axis) > 1:
            high = float(axis.coords[-1]) + float(axis.coords[1] - axis.coords[0])
        else:
            high = float(axis.coords[-1])

        axes_meta.append({
            "type": axis_type,
            "num_points": len(axis),
            "low": float(axis.coords[0]),
            "high": high,
            "periodic": axis.periodic,
            "name": axis.name,
        })

    return {"grid_type": grid_type, "axes": axes_meta}


def _meta_to_grid(meta: dict) -> Grid:
    """Reconstruct a grid from a metadata dict."""
    grid_cls = _GRID_CLASSES[meta["grid_type"]]
    axes = []
    for ax in meta["axes"]:
        axis_cls = _AXIS_CLASSES[ax["type"]]
        kwargs: dict = {}
        if ax.get("periodic"):
            kwargs["periodic"] = True
        if ax.get("name") is not None:
            kwargs["name"] = ax["name"]
        axes.append(axis_cls(ax["num_points"], ax["low"], ax["high"], **kwargs))

    return grid_cls(*axes)

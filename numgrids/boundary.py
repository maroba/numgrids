"""Boundary condition application for numerical grids.

Provides classes to apply Dirichlet, Neumann, and Robin boundary conditions
to both array data and sparse linear systems built from differentiation
matrices.

Classes
-------
BoundaryFace
    Identifies one face (side) of a grid boundary.
DirichletBC
    Dirichlet condition: *u = g* on a boundary face.
NeumannBC
    Neumann condition: *du/dn = g* on a boundary face.
RobinBC
    Robin condition: *a u + b du/dn = g* on a boundary face.

Functions
---------
apply_bcs
    Apply a list of boundary conditions to a sparse linear system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
from numpy.typing import NDArray
from scipy.sparse import spmatrix

if TYPE_CHECKING:
    from numgrids.grids import Grid


class BoundaryFace:
    """A selection of boundary points on one face of a grid.

    For an *N*-dimensional grid, each non-periodic axis contributes two
    faces: ``"low"`` (index 0 along that axis) and ``"high"`` (last index).

    Parameters
    ----------
    grid : Grid
        The numerical grid.
    axis_index : int
        Which axis this face belongs to.
    side : str
        ``"low"`` or ``"high"``.
    """

    def __init__(self, grid: Grid, axis_index: int, side: str) -> None:
        if side not in ("low", "high"):
            raise ValueError(f"side must be 'low' or 'high', got {side!r}")
        if axis_index < 0 or axis_index >= grid.ndims:
            raise ValueError(
                f"axis_index {axis_index} out of range for {grid.ndims}-dimensional grid"
            )
        if grid.get_axis(axis_index).periodic:
            raise ValueError(
                f"Axis {axis_index} is periodic and has no boundary faces"
            )
        self.grid = grid
        self.axis_index = axis_index
        self.side = side

    @property
    def mask(self) -> NDArray:
        """Boolean mask of shape ``grid.shape``, True on this face."""
        m = np.zeros(self.grid.shape, dtype=bool)
        idx = [slice(None)] * self.grid.ndims
        idx[self.axis_index] = 0 if self.side == "low" else -1
        m[tuple(idx)] = True
        return m

    @property
    def flat_indices(self) -> NDArray:
        """Flat (1-D) indices of this face's points in the ravelled grid."""
        return np.flatnonzero(self.mask)

    @property
    def normal_sign(self) -> int:
        """Sign of the outward unit normal along the face's axis.

        Returns ``-1`` for the ``"low"`` face (outward normal points in the
        negative axis direction) and ``+1`` for ``"high"``.
        """
        return -1 if self.side == "low" else 1

    def __repr__(self) -> str:
        return f"BoundaryFace(axis={self.axis_index}, side={self.side!r})"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BoundaryCondition:
    """Base class for boundary conditions.

    Parameters
    ----------
    face : BoundaryFace
        The boundary face where this condition applies.
    value : float, NDArray, or callable
        The boundary value *g*.  A scalar is broadcast to every point on
        the face.  An array must be broadcastable to the face shape.  A
        callable receives a tuple of coordinate arrays (one per grid
        dimension, each restricted to the face) and must return an array.
    """

    def __init__(self, face: BoundaryFace, value=0.0) -> None:
        self.face = face
        self._value_spec = value

    def _resolve_value(self) -> NDArray:
        """Return a 1-D array of values, one per face point."""
        face = self.face
        grid = face.grid
        n = len(face.flat_indices)

        if callable(self._value_spec):
            idx = [slice(None)] * grid.ndims
            idx[face.axis_index] = 0 if face.side == "low" else -1
            face_coords = tuple(mc[tuple(idx)] for mc in grid.meshed_coords)
            g = np.asarray(self._value_spec(face_coords))
            return g.ravel()
        elif isinstance(self._value_spec, np.ndarray):
            return self._value_spec.ravel()
        else:
            return np.full(n, float(self._value_spec))

    # Subclasses override these:
    def apply(self, u: NDArray) -> NDArray:
        raise NotImplementedError

    def apply_to_system(self, L: spmatrix, rhs: NDArray) -> tuple[spmatrix, NDArray]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dirichlet
# ---------------------------------------------------------------------------

class DirichletBC(BoundaryCondition):
    r"""Dirichlet boundary condition: :math:`u = g` on a boundary face.

    Parameters
    ----------
    face : BoundaryFace
        The boundary face.
    value : float, NDArray, or callable
        The prescribed boundary value *g*.
    """

    def apply(self, u: NDArray) -> NDArray:
        """Set *u* to *g* on the boundary face (in-place).

        Returns
        -------
        NDArray
            The modified array (same object as *u*).
        """
        u[self.face.mask] = self._resolve_value()
        return u

    def apply_to_system(self, L: spmatrix, rhs: NDArray) -> tuple[spmatrix, NDArray]:
        """Replace boundary rows in *L* with identity rows and set *rhs = g*.

        Parameters
        ----------
        L : spmatrix
            System matrix of shape ``(grid.size, grid.size)``.
        rhs : NDArray
            Right-hand side vector of shape ``(grid.size,)``.

        Returns
        -------
        tuple of (spmatrix, NDArray)
            Modified ``(L, rhs)``.
        """
        L = L.tolil()
        rhs = rhs.copy()
        g = self._resolve_value()
        for i, idx in enumerate(self.face.flat_indices):
            L[idx, :] = 0
            L[idx, idx] = 1.0
            rhs[idx] = g[i]
        return L.tocsc(), rhs


# ---------------------------------------------------------------------------
# Neumann
# ---------------------------------------------------------------------------

class NeumannBC(BoundaryCondition):
    r"""Neumann boundary condition: :math:`\partial u / \partial n = g`.

    The outward normal derivative is ``sign * du/d(axis)`` where *sign* is
    ``+1`` on the ``"high"`` face and ``-1`` on the ``"low"`` face.

    Parameters
    ----------
    face : BoundaryFace
        The boundary face.
    value : float, NDArray, or callable
        The prescribed normal-derivative value *g*.
    """

    def __init__(self, face: BoundaryFace, value=0.0) -> None:
        super().__init__(face, value)
        from numgrids.api import Diff
        self._diff = Diff(face.grid, order=1, axis_index=face.axis_index)

    def apply(self, u: NDArray) -> NDArray:
        """Adjust the boundary layer of *u* so that the discrete normal
        derivative approximates *g* (first-order, in-place).

        For higher accuracy, prefer :meth:`apply_to_system`.
        """
        face = self.face
        axis = face.grid.get_axis(face.axis_index)
        coords = axis.coords
        g = self._resolve_value()

        src = [slice(None)] * face.grid.ndims
        dst = [slice(None)] * face.grid.ndims

        if face.side == "low":
            h = coords[1] - coords[0]
            src[face.axis_index] = 1
            dst[face.axis_index] = 0
            # u[0] = u[1] - h * g  (outward normal is -x, so du/dn = -du/dx = g => du/dx = -g)
            u[tuple(dst)] = u[tuple(src)] + h * face.normal_sign * g.reshape(
                u[tuple(dst)].shape
            )
        else:
            h = coords[-1] - coords[-2]
            src[face.axis_index] = -2
            dst[face.axis_index] = -1
            u[tuple(dst)] = u[tuple(src)] + h * face.normal_sign * g.reshape(
                u[tuple(dst)].shape
            )
        return u

    def apply_to_system(self, L: spmatrix, rhs: NDArray) -> tuple[spmatrix, NDArray]:
        """Replace boundary rows with the normal-derivative operator row.

        Uses the full differentiation matrix from :class:`~numgrids.api.Diff`.
        """
        L = L.tolil()
        rhs = rhs.copy()
        g = self._resolve_value()
        sign = self.face.normal_sign
        D = self._diff.as_matrix().tolil()
        for i, idx in enumerate(self.face.flat_indices):
            L[idx, :] = sign * D[idx, :]
            rhs[idx] = g[i]
        return L.tocsc(), rhs


# ---------------------------------------------------------------------------
# Robin
# ---------------------------------------------------------------------------

class RobinBC(BoundaryCondition):
    r"""Robin boundary condition: :math:`a\,u + b\,\partial u/\partial n = g`.

    Parameters
    ----------
    face : BoundaryFace
        The boundary face.
    a : float
        Coefficient of *u*.
    b : float
        Coefficient of the normal derivative.
    value : float, NDArray, or callable
        The prescribed value *g*.
    """

    def __init__(self, face: BoundaryFace, a: float, b: float, value=0.0) -> None:
        super().__init__(face, value)
        self.a = a
        self.b = b
        from numgrids.api import Diff
        self._diff = Diff(face.grid, order=1, axis_index=face.axis_index)

    def apply(self, u: NDArray) -> NDArray:
        """Not supported for Robin conditions; use :meth:`apply_to_system`."""
        raise NotImplementedError(
            "Function-level apply is not supported for Robin BCs. "
            "Use apply_to_system() for sparse linear system modification."
        )

    def apply_to_system(self, L: spmatrix, rhs: NDArray) -> tuple[spmatrix, NDArray]:
        """Replace boundary rows with ``a * I + b * sign * D``."""
        L = L.tolil()
        rhs = rhs.copy()
        g = self._resolve_value()
        sign = self.face.normal_sign
        D = self._diff.as_matrix().tolil()
        I = scipy.sparse.identity(self.face.grid.size, format="lil")
        for i, idx in enumerate(self.face.flat_indices):
            L[idx, :] = self.a * I[idx, :] + self.b * sign * D[idx, :]
            rhs[idx] = g[i]
        return L.tocsc(), rhs


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def apply_bcs(
    bcs: list[BoundaryCondition], L: spmatrix, rhs: NDArray
) -> tuple[spmatrix, NDArray]:
    """Apply a sequence of boundary conditions to a linear system.

    Conditions are applied in list order; for overlapping points (e.g.
    corners) the last condition wins.

    Parameters
    ----------
    bcs : list of BoundaryCondition
        Boundary conditions to apply.
    L : spmatrix
        System matrix of shape ``(grid.size, grid.size)``.
    rhs : NDArray
        Right-hand side vector of shape ``(grid.size,)``.

    Returns
    -------
    tuple of (spmatrix, NDArray)
    """
    for bc in bcs:
        L, rhs = bc.apply_to_system(L, rhs)
    return L, rhs

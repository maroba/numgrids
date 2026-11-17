"""Boundary condition application for numerical grids.

Provides classes to apply Dirichlet, Neumann, and Robin boundary conditions
to both array data and sparse linear systems built from differentiation
matrices.

Classes
-------
BoundaryFace
    Identifies one face (side) of a grid boundary, with helpers to evaluate,
    differentiate and integrate on it.
DirichletBC
    Dirichlet condition: *u = g* on a boundary face.
NeumannBC
    Neumann condition: *du/dn = g* on a boundary face.
RobinBC
    Robin condition: *a u + b du/dn = g* on a boundary face.
Dirichlet, Neumann, Robin
    Unbound boundary condition specifications, used in dictionaries passed
    to :meth:`numgrids.operators.Operator.solve` and :func:`resolve_bcs`.

Functions
---------
apply_bcs
    Apply a list of boundary conditions to a sparse linear system.
resolve_bcs
    Turn a compact boundary condition specification into a list of
    bound boundary conditions.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
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
    Ends that are not a boundary have no face: the origin of a radial axis,
    the poles of a colatitude axis and the ends at infinity of the axes on
    unbounded domains.

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
        if isinstance(axis_index, str):
            axis_index = grid.axis_index(axis_index)
        if axis_index < 0 or axis_index >= grid.ndims:
            raise ValueError(
                f"axis_index {axis_index} out of range for {grid.ndims}-dimensional grid"
            )
        axis = grid.get_axis(axis_index)
        if axis.periodic:
            raise ValueError(
                f"Axis {axis_index} is periodic and has no boundary faces"
            )
        if side not in axis.boundary_sides:
            # e.g. the origin of a radial axis or an end at infinity
            sides = " or ".join(repr(s) for s in axis.boundary_sides) or "none"
            raise ValueError(
                f"Axis {axis_index} ({type(axis).__name__}) has no {side!r} boundary face "
                f"(its faces: {sides})."
            )
        self.grid = grid
        self.axis_index = axis_index
        self.side = side

    @property
    def key(self) -> str:
        """The canonical key of this face in :attr:`Grid.faces`, e.g. ``"0_low"``."""
        return f"{self.axis_index}_{self.side}"

    @property
    def _index(self) -> tuple:
        idx = [slice(None)] * self.grid.ndims
        idx[self.axis_index] = 0 if self.side == "low" else -1
        return tuple(idx)

    @property
    def mask(self) -> NDArray:
        """Boolean mask of shape ``grid.shape``, True on this face."""
        m = np.zeros(self.grid.shape, dtype=bool)
        m[self._index] = True
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

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of arrays restricted to this face."""
        return tuple(n for i, n in enumerate(self.grid.shape) if i != self.axis_index)

    @property
    def coords(self) -> tuple[NDArray, ...]:
        """Meshed coordinates (all grid dimensions) restricted to this face."""
        return tuple(mc[self._index] for mc in self.grid.meshed_coords)

    def restrict(self, f: NDArray) -> NDArray:
        """Restrict a meshed function to this face.

        Parameters
        ----------
        f : NDArray
            Array of shape ``grid.shape``.

        Returns
        -------
        NDArray
            Array of shape :attr:`shape`.
        """
        f = self.grid._check_field(f)
        return f[self._index]

    def normal_derivative(self, f: NDArray) -> NDArray:
        r"""Outward normal derivative :math:`\partial f / \partial n` on this face.

        On curvilinear grids the physical normal derivative
        :math:`\pm (1/h_i)\, \partial f / \partial q_i` is returned.

        Parameters
        ----------
        f : NDArray
            Array of shape ``grid.shape``.

        Returns
        -------
        NDArray
            Array of shape :attr:`shape`.
        """
        df = self.grid.diff(self.grid._check_field(f), 1, self.axis_index)
        dn = self.normal_sign * df[self._index]
        h = self._scale_factors()
        if h is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                dn = dn / h[self.axis_index][self._index]
            dn = np.where(np.isfinite(dn), dn, 0.0)
        return dn

    def _scale_factors(self):
        from numgrids.curvilinear import CurvilinearGrid
        if isinstance(self.grid, CurvilinearGrid):
            return self.grid.scale_factors
        return None

    @property
    def area_element(self) -> float | NDArray:
        r"""Surface element on this face (product of the other scale factors).

        ``1.0`` for Cartesian grids.
        """
        h = self._scale_factors()
        if h is None:
            return 1.0
        dA = np.ones(self.shape)
        for i, hi in enumerate(h):
            if i != self.axis_index:
                dA = dA * hi[self._index]
        return dA

    def integrate(self, f: NDArray) -> float:
        """Integrate a function over this face.

        Parameters
        ----------
        f : NDArray
            Either an array of shape ``grid.shape`` (restricted to the face
            automatically) or an array of shape :attr:`shape`.

        Returns
        -------
        float
            The surface integral (including the surface element on
            curvilinear grids).  For 1D grids the face is a single point
            and its value is returned.
        """
        f = np.asarray(f)
        if f.shape == self.grid.shape:
            f = f[self._index]
        if f.shape != self.shape:
            raise ValueError(
                f"Array of shape {f.shape} matches neither the grid shape "
                f"{self.grid.shape} nor the face shape {self.shape}."
            )
        f = f * self.area_element
        result = f
        weights = [self.grid._quadrature_weights(i) for i in range(self.grid.ndims)
                   if i != self.axis_index]
        for w in reversed(weights):
            result = result @ w
        return complex(result) if np.iscomplexobj(result) else float(result)

    def flux(self, *v: NDArray) -> float:
        r"""Outward flux :math:`\int \mathbf{v}\cdot\mathbf{n}\, dA` of a vector field.

        Parameters
        ----------
        *v : NDArray
            The (physical) components of the vector field, one per axis.
        """
        if len(v) != self.grid.ndims:
            raise ValueError(f"Expected {self.grid.ndims} vector components, got {len(v)}.")
        return self.normal_sign * self.integrate(v[self.axis_index])

    def __repr__(self) -> str:
        return f"BoundaryFace(axis={self.axis_index}, side={self.side!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_rows(L: spmatrix, rows: NDArray, R: spmatrix) -> spmatrix:
    """Return a copy of *L* whose *rows* are replaced by the rows of *R*."""
    n = L.shape[0]
    sel = np.zeros(n)
    sel[rows] = 1.0
    keep = scipy.sparse.diags(1.0 - sel)
    take = scipy.sparse.diags(sel)
    L_new = keep @ scipy.sparse.csr_matrix(L) + take @ scipy.sparse.csr_matrix(R)
    L_new = scipy.sparse.csc_matrix(L_new)
    L_new.eliminate_zeros()
    return L_new


# ---------------------------------------------------------------------------
# Time-dependent boundary values
# ---------------------------------------------------------------------------

class TimeDependent:
    """A boundary value that changes in time, for :func:`~numgrids.evolve`.

    Parameters
    ----------
    func : callable
        ``func(t)`` returns the boundary value at time *t*: a scalar, an
        array matching the face, or a callable of the face coordinates
        (as for time-independent values).

    Examples
    --------
    A rod whose left end is heated periodically, while the right end is
    insulated:

    >>> import numpy as np
    >>> from numgrids import Grid, Neumann, TimeDependent, chebyshev, evolve
    >>> grid = Grid(x=chebyshev(20, 0, 1))
    >>> bcs = {"x_low": TimeDependent(lambda t: np.sin(t)), "x_high": Neumann(0)}
    >>> result = evolve(grid, grid.laplacian_operator(), np.zeros(grid.shape), (0, 2),
    ...                 bcs=bcs, method="BDF")
    >>> bool(np.isclose(result.final[0], np.sin(2)))
    True
    """

    def __init__(self, func: Callable) -> None:
        if not callable(func):
            raise TypeError("TimeDependent needs a function of time.")
        self.func = func

    def __call__(self, t):
        return self.func(t)

    def __repr__(self) -> str:
        return f"TimeDependent({self.func!r})"


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

    @property
    def time_dependent(self) -> bool:
        """Whether the boundary value changes in time (see :class:`TimeDependent`)."""
        return isinstance(self._value_spec, TimeDependent)

    def at_time(self, t: float) -> BoundaryCondition:
        """The condition with its value at time *t* (the condition itself if constant)."""
        if not self.time_dependent:
            return self
        frozen = copy.copy(self)
        frozen._value_spec = self._value_spec(t)
        return frozen

    def _resolve_value(self) -> NDArray:
        """Return a 1-D array of values, one per face point."""
        face = self.face
        n = len(face.flat_indices)
        spec = self._value_spec
        if isinstance(spec, TimeDependent):
            raise ValueError(
                "Time-dependent boundary values are only meaningful in evolve(). "
                "Evaluate them at a fixed time first, e.g. Dirichlet(g(t))."
            )

        if callable(spec) or isinstance(spec, np.ndarray):
            g = spec(face.coords) if callable(spec) else spec
            g = np.asarray(g)
            if g.size == n:
                return g.ravel()
            return np.broadcast_to(g, face.coords[0].shape).ravel()
        elif np.iscomplexobj(spec):
            return np.full(n, complex(spec))
        else:
            return np.full(n, float(spec))

    # Subclasses override these:
    def apply(self, u: NDArray) -> NDArray:
        raise NotImplementedError

    def apply_to_system(self, L: spmatrix, rhs: NDArray) -> tuple[spmatrix, NDArray]:
        raise NotImplementedError

    def _set_rhs(self, rhs: NDArray) -> NDArray:
        g = self._resolve_value()
        rhs = np.array(rhs, dtype=np.result_type(rhs, g), copy=True)
        rhs[self.face.flat_indices] = g
        return rhs

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.face!r}, value={self._value_spec!r})"


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
        I = scipy.sparse.identity(L.shape[0], format="csr")  # noqa: E741
        return _replace_rows(L, self.face.flat_indices, I), self._set_rhs(rhs)


# ---------------------------------------------------------------------------
# Neumann
# ---------------------------------------------------------------------------

class NeumannBC(BoundaryCondition):
    r"""Neumann boundary condition: :math:`\partial u / \partial n = g`.

    The outward normal derivative is ``sign * du/d(axis)`` where *sign* is
    ``+1`` on the ``"high"`` face and ``-1`` on the ``"low"`` face.  On
    curvilinear grids the derivative is taken with respect to the
    coordinate (not the arc length); this makes no difference for
    homogeneous conditions.

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
        else:
            h = coords[-1] - coords[-2]
            src[face.axis_index] = -2
            dst[face.axis_index] = -1
        # Outward normal derivative (u[dst] - u[src]) / h * (+-1) = g
        u[tuple(dst)] = u[tuple(src)] + h * face.normal_sign * g.reshape(
            u[tuple(dst)].shape
        )
        return u

    def apply_to_system(self, L: spmatrix, rhs: NDArray) -> tuple[spmatrix, NDArray]:
        """Replace boundary rows with the normal-derivative operator row.

        Uses the full differentiation matrix from :class:`~numgrids.api.Diff`.
        """
        R = self.face.normal_sign * scipy.sparse.csr_matrix(self._diff.as_matrix())
        return _replace_rows(L, self.face.flat_indices, R), self._set_rhs(rhs)


# ---------------------------------------------------------------------------
# Robin
# ---------------------------------------------------------------------------

class RobinBC(BoundaryCondition):
    r"""Robin boundary condition: :math:`a\,u + b\,\partial u/\partial n = g`.

    Parameters
    ----------
    face : BoundaryFace
        The boundary face.
    a : float, NDArray, or callable
        Coefficient of *u*.
    b : float, NDArray, or callable
        Coefficient of the normal derivative.
    value : float, NDArray, or callable
        The prescribed value *g*.

    The coefficients may vary along the face: like the value, they can be
    arrays matching the face or callables of the face coordinates, e.g. for
    Newton's law of cooling :math:`h\,u + k\,\partial u/\partial n = h\,u_\infty`
    with a conductivity :math:`k` that depends on the position.
    """

    def __init__(self, face: BoundaryFace, a, b, value=0.0) -> None:
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
        D = self.face.normal_sign * scipy.sparse.csr_matrix(self._diff.as_matrix())
        R = self._coefficient(self.a) + self._coefficient(self.b) @ D
        return _replace_rows(L, self.face.flat_indices, R), self._set_rhs(rhs)

    def _coefficient(self, spec) -> spmatrix:
        """Diagonal matrix with the values of a coefficient on the points of the face."""
        n = self.face.grid.size
        if np.isscalar(spec):
            return spec * scipy.sparse.identity(n, format="csr")
        values = BoundaryCondition(self.face, spec)._resolve_value()
        diagonal = np.zeros(n, dtype=values.dtype)
        diagonal[self.face.flat_indices] = values
        return scipy.sparse.diags(diagonal, format="csr")

    def __repr__(self) -> str:
        return f"RobinBC({self.face!r}, a={self.a!r}, b={self.b!r}, value={self._value_spec!r})"


# ---------------------------------------------------------------------------
# Unbound specifications
# ---------------------------------------------------------------------------

class BCSpec:
    """Base class for boundary condition specs not yet bound to a face."""

    def bind(self, face: BoundaryFace) -> BoundaryCondition:
        raise NotImplementedError


class Dirichlet(BCSpec):
    """Dirichlet condition *u = value*, to be bound to a face later.

    Examples
    --------
    >>> from numgrids import Grid, Dirichlet, Neumann, chebyshev
    >>> grid = Grid(x=chebyshev(10, 0, 1))
    >>> bcs = {"x_low": Dirichlet(1.0), "x_high": Neumann(0.0)}
    """

    def __init__(self, value=0.0) -> None:
        self.value = value

    def bind(self, face: BoundaryFace) -> DirichletBC:
        return DirichletBC(face, self.value)

    def __repr__(self) -> str:
        return f"Dirichlet({self.value!r})"


class Neumann(BCSpec):
    """Neumann condition *du/dn = value*, to be bound to a face later."""

    def __init__(self, value=0.0) -> None:
        self.value = value

    def bind(self, face: BoundaryFace) -> NeumannBC:
        return NeumannBC(face, self.value)

    def __repr__(self) -> str:
        return f"Neumann({self.value!r})"


class Robin(BCSpec):
    """Robin condition *a u + b du/dn = value*, to be bound to a face later.

    *a*, *b* and *value* are numbers, arrays matching the face, or callables
    of the face coordinates.
    """

    def __init__(self, a, b, value=0.0) -> None:
        self.a = a
        self.b = b
        self.value = value

    def bind(self, face: BoundaryFace) -> RobinBC:
        return RobinBC(face, self.a, self.b, self.value)

    def __repr__(self) -> str:
        return f"Robin(a={self.a!r}, b={self.b!r}, value={self.value!r})"


_ALL_KEYS = ("all", "*")


def _bind(spec, face: BoundaryFace) -> BoundaryCondition:
    if isinstance(spec, BCSpec):
        return spec.bind(face)
    if isinstance(spec, BoundaryCondition):
        raise TypeError(
            "Use unbound specs (Dirichlet, Neumann, Robin) as dict values, "
            "not bound conditions such as DirichletBC."
        )
    return DirichletBC(face, spec)


def resolve_bcs(grid: Grid, bcs) -> list[BoundaryCondition]:
    """Convert a compact boundary condition specification to a list.

    Parameters
    ----------
    grid : Grid
        The grid the conditions refer to.
    bcs : None, scalar, callable, BCSpec, BoundaryCondition, list or dict
        * ``None`` -- no conditions,
        * a scalar, array or callable -- Dirichlet condition with that
          value on every boundary face,
        * a :class:`Dirichlet`, :class:`Neumann` or :class:`Robin` spec --
          applied to every boundary face,
        * a bound :class:`BoundaryCondition` or a list of them -- used as is,
        * a dict ``{face_key: spec_or_value}``.  Face keys are those of
          :attr:`Grid.faces` (``"0_low"``, ``"x_high"``, ...).  The key
          ``"all"`` applies to every face; specific keys override it.
          Plain values mean Dirichlet conditions.

    Returns
    -------
    list of BoundaryCondition

    Examples
    --------
    >>> from numgrids import Grid, Neumann, chebyshev
    >>> from numgrids.boundary import resolve_bcs
    >>> grid = Grid(x=chebyshev(10, 0, 1), y=chebyshev(10, 0, 1))
    >>> bcs = resolve_bcs(grid, {"all": 0, "y_high": Neumann(1)})
    >>> [type(bc).__name__ for bc in bcs]
    ['DirichletBC', 'DirichletBC', 'DirichletBC', 'DirichletBC', 'NeumannBC']
    """
    if bcs is None:
        return []
    if isinstance(bcs, BoundaryCondition):
        return [bcs]
    if isinstance(bcs, (list, tuple)):
        result = []
        for bc in bcs:
            if not isinstance(bc, BoundaryCondition):
                raise TypeError(
                    f"List entries must be bound boundary conditions, got {type(bc).__name__}. "
                    "Use a dict to map faces to specs."
                )
            result.append(bc)
        return result
    faces = grid.faces
    if isinstance(bcs, dict):
        result = []
        for key in _ALL_KEYS:
            if key in bcs:
                result.extend(_bind(bcs[key], face) for face in faces.values())
        for key, spec in bcs.items():
            if key in _ALL_KEYS:
                continue
            if isinstance(key, BoundaryFace):
                face = key
            else:
                face = faces[key]
            result.append(_bind(spec, face))
        return result
    # A single spec or value for all faces:
    return [_bind(bcs, face) for face in faces.values()]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def apply_bcs(
    bcs, L: spmatrix, rhs: NDArray, grid: Grid | None = None
) -> tuple[spmatrix, NDArray]:
    """Apply a sequence of boundary conditions to a linear system.

    Conditions are applied in list order; for overlapping points (e.g.
    corners) the last condition wins.

    Parameters
    ----------
    bcs : list of BoundaryCondition, or any spec accepted by :func:`resolve_bcs`
        Boundary conditions to apply.  Compact specs (dicts, scalars)
        require *grid*.
    L : spmatrix
        System matrix of shape ``(grid.size, grid.size)``.
    rhs : NDArray
        Right-hand side vector of shape ``(grid.size,)``.
    grid : Grid, optional
        Needed only for compact specifications.

    Returns
    -------
    tuple of (spmatrix, NDArray)
    """
    if not isinstance(bcs, (list, tuple, BoundaryCondition)) and bcs is not None:
        if grid is None:
            raise ValueError("Pass `grid=` to use a compact boundary condition specification.")
        bcs = resolve_bcs(grid, bcs)
    elif isinstance(bcs, BoundaryCondition):
        bcs = [bcs]
    for bc in bcs or []:
        L, rhs = bc.apply_to_system(L, rhs)
    return L, rhs

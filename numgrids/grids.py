"""Tensor-product grids built from one or more axes."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from numgrids.axes import Axis

if TYPE_CHECKING:
    from pathlib import Path

    from numgrids.boundary import BoundaryFace
    from numgrids.operators import Operator


class _FaceDict(dict):
    """Dictionary of boundary faces that also accepts axis-name aliases.

    Faces are stored under their canonical keys ``"{axis_index}_{side}"``
    (e.g. ``"0_low"``).  If an axis has a name, ``"{name}_{side}"``
    (e.g. ``"x_low"``) can be used as well.  Iteration only yields the
    canonical keys, so every face is visited exactly once.
    """

    def __init__(self, faces: dict, aliases: dict[str, str]) -> None:
        super().__init__(faces)
        self._aliases = aliases

    def _canonical(self, key):
        return self._aliases.get(key, key)

    def __getitem__(self, key) -> BoundaryFace:
        try:
            return super().__getitem__(self._canonical(key))
        except KeyError:
            available = list(self) + list(self._aliases)
            raise KeyError(
                f"No boundary face {key!r}. Available faces: {available}"
            ) from None

    def __contains__(self, key) -> bool:
        return super().__contains__(self._canonical(key))

    def get(self, key, default=None):
        return super().get(self._canonical(key), default)


class Grid:
    """An N-dimensional numerical grid built from one or more axes.

    A ``Grid`` is the tensor product of its constituent :class:`~numgrids.axes.Axis`
    objects.  It provides meshed coordinate arrays, boundary masks, and
    convenience methods for differentiation, integration, interpolation,
    vector calculus, refinement and plotting.

    Axes can be passed positionally or as keyword arguments; in the latter
    case the keyword becomes the axis name, which can then be used instead
    of the axis index everywhere (``grid.diff(f, axis="x")``,
    ``grid.faces["x_low"]``, ...).

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, chebyshev, periodic
    >>> grid = Grid(x=chebyshev(20, 0, 1), y=periodic(32, 0, 2 * np.pi))
    >>> grid.shape
    (20, 32)
    >>> X, Y = grid.meshed_coords
    >>> f = X ** 2 * np.sin(Y)
    >>> df_dx = grid.diff(f, axis="x")
    >>> lap = grid.laplacian(f)
    """

    def __init__(self, *axes: Axis, **named_axes: Axis) -> None:
        """
        Constructor

        Parameters
        ----------
        axes: one or more Axis objects
            One or more axes defining the grid. The order of the axes matters!
            The first axis passed will have index 0.
        named_axes: Axis objects as keyword arguments
            Further axes, appended after the positional ones in the given
            order.  The keyword is used as the axis name.
        """
        for name, ax in named_axes.items():
            if not isinstance(ax, Axis):
                raise TypeError(
                    f"Expected an Axis object for axis {name!r}, got {type(ax).__name__}."
                )
        axes = tuple(axes) + tuple(ax.renamed(name) for name, ax in named_axes.items())
        if not axes:
            raise ValueError("A grid needs at least one axis.")
        for ax in axes:
            if not isinstance(ax, Axis):
                raise TypeError(
                    f"Expected Axis objects, got {type(ax).__name__}. "
                    "Create axes with e.g. numgrids.chebyshev(...) or numgrids.create_axis(...)."
                )
        names = [ax.name for ax in axes if ax.name is not None]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate axis names: {sorted(duplicates)}")

        self._axes = axes
        self.ndims = len(axes)
        self._meshed_coords = tuple(np.meshgrid(
            *tuple(a.coords for a in axes), indexing="ij"))

        bdry = np.ones(self.shape, dtype=bool)
        mask = [axis.boundary for axis in self.axes]
        bdry[tuple(mask)] = False
        self._boundary = bdry

        # The functions in the api module can store entities here for re-use:
        self.cache = {
            "diffs": {},
            "integral": None,
            "interpolators": {},
            "faces": None,
            "operators": {},
        }

    # ── construction helpers ─────────────────────────────────────

    def _new_like(self, axes) -> Grid:
        """Create a grid of the same type (and parameters) with new axes."""
        new = copy.copy(self)
        Grid.__init__(new, *axes)
        return new

    # ── basic properties ─────────────────────────────────────────

    @property
    def size(self) -> int:
        """Total number of grid points."""
        return int(np.prod(self.shape))

    def axis_index(self, key: int | str) -> int:
        """Return the (non-negative) index of an axis given by index or name.

        Parameters
        ----------
        key : int or str
            Axis index (negative values count from the end) or axis name.

        Returns
        -------
        int
        """
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            idx = int(key)
            if idx < 0:
                idx += self.ndims
            if not 0 <= idx < self.ndims:
                raise IndexError(
                    f"Axis index {key} out of range for {self.ndims}-dimensional grid."
                )
            return idx
        if isinstance(key, str):
            for i, ax in enumerate(self.axes):
                if ax.name == key:
                    return i
            names = [ax.name for ax in self.axes if ax.name is not None]
            raise KeyError(f"No axis named {key!r}. Named axes: {names}")
        raise TypeError(f"Axis key must be int or str, got {type(key).__name__}.")

    def get_axis(self, idx: int | str = 0) -> Axis:
        """
        Returns the axis with given index or name.

        Parameters
        ----------
        idx: int or str
            The index or name of the requested axis.

        Returns
        -------
        The requested Axis object.
        """
        return self.axes[self.axis_index(idx)]

    def __getitem__(self, inds) -> NDArray:
        if self.ndims == 1:
            return self.axes[0][inds]
        return np.array(
            tuple(self.axes[k][inds[k]] for k in range(self.ndims))
        )

    @property
    def axes(self) -> tuple:
        """Returns the axes objects of the grid."""
        return self._axes

    @property
    def coords(self) -> NDArray | tuple[NDArray, ...]:
        """
        Returns a tuple of lists with the coordinate values along each axis.
        In case of 1D, only a single list is returned.
        """
        if self.ndims == 1:
            return self.axes[0].coords
        return tuple(a.coords for a in self.axes)

    @property
    def meshed_coords(self) -> tuple[NDArray, ...]:
        """Returns a tuple with the meshed coordinate values."""
        return self._meshed_coords

    @property
    def cartesian_coords(self) -> tuple[NDArray, ...]:
        """Meshed Cartesian coordinates of the grid points.

        For a plain (Cartesian) grid this is identical to
        :attr:`meshed_coords`.  Curvilinear grids map their coordinates
        to Cartesian space.
        """
        return self.meshed_coords

    @property
    def volume_element(self) -> float | NDArray:
        """The volume element (Jacobian) of the coordinate system.

        ``1.0`` for Cartesian grids; an array of shape ``grid.shape`` for
        curvilinear grids.
        """
        return 1.0

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns a tuple with the number of grid points along each axis."""
        return tuple(len(axis) for axis in self.axes)

    def __repr__(self) -> str:
        axes = ",\n".join(f"    {ax!r}" for ax in self.axes)
        return f"{type(self).__name__}(shape={self.shape}, ndims={self.ndims}, axes=[\n{axes}\n])"

    def _eq_key(self) -> tuple:
        return (type(self), self.axes)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return self._eq_key() == other._eq_key()

    def __hash__(self) -> int:
        return hash(tuple(map(hash, self.axes)))

    def _antipodal_map(self, axis_index: int) -> Callable[[NDArray], NDArray]:
        """Map a field to its values at the antipodal points of a radial axis.

        Only grids in which a :class:`~numgrids.axes.RadialAxis` passes
        through the origin (polar, cylindrical, spherical) provide it.
        """
        raise ValueError(
            "A RadialAxis passes through the coordinate origin and can only be used as the "
            "radial axis of a PolarGrid, CylindricalGrid or SphericalGrid."
        )

    def _quadrature_weights(self, axis_index: int) -> NDArray:
        """Quadrature weights used for integrating along an axis of this grid."""
        return self.axes[axis_index].quadrature_weights

    def _check_field(self, f, name: str = "f") -> NDArray:
        """Validate that *f* is an array matching the grid shape."""
        f = np.asarray(f)
        if f.shape != self.shape:
            raise ValueError(
                f"Array '{name}' has shape {f.shape}, which does not match "
                f"the grid shape {self.shape}."
            )
        return f

    # ── evaluating functions ────────────────────────────────────

    def evaluate(self, func: Callable[..., Any]) -> NDArray:
        """Evaluate a function on all grid points.

        Parameters
        ----------
        func : callable
            Called as ``func(*meshed_coords)``, i.e. with one coordinate
            array per axis.  May also return a scalar, which is broadcast.

        Returns
        -------
        NDArray
            Array of shape ``grid.shape``.

        Examples
        --------
        >>> from numgrids import Grid, equidistant
        >>> grid = Grid(equidistant(5, 0, 1), equidistant(3, 0, 1))
        >>> grid.evaluate(lambda x, y: x + y).shape
        (5, 3)
        """
        result = np.asarray(func(*self.meshed_coords))
        return np.broadcast_to(result, self.shape).copy()

    # ── calculus ────────────────────────────────────────────────

    def diff(self, f: NDArray, order: int = 1, axis: int | str = 0, acc: int = 4) -> NDArray:
        """Partial derivative of a meshed function along one axis.

        The differentiation operator is cached on the grid.

        Parameters
        ----------
        f : NDArray
            Meshed function values of shape ``grid.shape``.
        order : int
            Derivative order (default 1).
        axis : int or str
            Axis index or name (default 0).
        acc : int
            Accuracy order for finite-difference based methods (default 4).

        Returns
        -------
        NDArray
        """
        from numgrids.api import diff
        return diff(self, f, order=order, axis_index=axis, acc=acc)

    def diff_operator(self, order: int = 1, axis: int | str = 0, acc: int = 4) -> Operator:
        """Partial derivative as an :class:`~numgrids.operators.Operator`.

        Operators can be combined algebraically (``+``, ``-``, ``*``, ``@``,
        ``**``) and support :meth:`~numgrids.operators.Operator.solve` and
        :meth:`~numgrids.operators.Operator.eig`.
        """
        from numgrids.api import Diff
        return Diff(self, order, axis, acc=acc).to_operator()

    def laplacian(self, f: NDArray) -> NDArray:
        r"""Laplacian :math:`\nabla^2 f` of a scalar field."""
        f = self._check_field(f)
        return sum(self.diff(f, 2, i) for i in range(self.ndims))

    def laplacian_operator(self, acc: int = 4) -> Operator:
        r"""Laplacian :math:`\nabla^2` as an :class:`~numgrids.operators.Operator`."""
        key = ("laplacian", acc)
        if key not in self.cache["operators"]:
            from numgrids.api import Diff
            from numgrids.operators import Operator
            M = sum(Diff(self, 2, i, acc=acc).as_matrix() for i in range(self.ndims))
            self.cache["operators"][key] = Operator(self, M)
        return self.cache["operators"][key]

    def gradient_operator(self) -> tuple[Operator, ...]:
        r"""The components of the gradient as operators.

        Returns one :class:`~numgrids.operators.Operator` per axis, so that
        ``[G(f) for G in grid.gradient_operator()]`` equals
        ``grid.gradient(f)``.  Together with :meth:`divergence_operator`
        this builds operators in divergence form, e.g. diffusion with a
        variable coefficient :math:`\nabla\cdot(k \nabla u)`:

        >>> import numpy as np
        >>> from numgrids import Grid, chebyshev
        >>> grid = Grid(x=chebyshev(20, 0, 1))
        >>> k = 1 + grid.coords
        >>> (G,), (D,) = grid.gradient_operator(), grid.divergence_operator()
        >>> L = D @ (k * G)                                   # u -> (k u')'
        >>> bool(np.allclose(L(grid.coords ** 2), 2 + 4 * grid.coords))
        True
        """
        return tuple(self.diff_operator(1, i) for i in range(self.ndims))

    def divergence_operator(self) -> tuple[Operator, ...]:
        r"""The divergence as one operator per vector component.

        :math:`\nabla\cdot\mathbf{v} = \sum_i D_i v_i` for the returned
        operators :math:`D_i`.  On curvilinear grids they act on the
        physical components of the vector field and include the scale
        factors.  See :meth:`gradient_operator` for an example.
        """
        return tuple(self.diff_operator(1, i) for i in range(self.ndims))

    def diffusion_operator(self, k=1.0) -> Operator:
        r"""The operator :math:`u \mapsto \nabla\cdot(k\,\nabla u)` in divergence form.

        Parameters
        ----------
        k : float, NDArray or callable
            The coefficient (e.g. a conductivity or diffusivity): a scalar,
            an array of grid shape, or a callable evaluated on the grid.
            With ``k = 1`` this is the Laplacian (but
            :meth:`laplacian_operator` is the more accurate discretization
            of it on curvilinear grids).

        Examples
        --------
        Steady heat conduction :math:`(k u')' = 0` with
        :math:`k = 1 + x`, :math:`u(0) = 0`, :math:`u(1) = 1`:

        >>> import numpy as np
        >>> from numgrids import Grid, chebyshev
        >>> grid = Grid(x=chebyshev(20, 0, 1))
        >>> u = grid.diffusion_operator(lambda x: 1 + x).solve(0, bcs={"x_low": 0, "x_high": 1})
        >>> bool(np.allclose(u, np.log(1 + grid.coords) / np.log(2)))
        True
        """
        if callable(k):
            k = self.evaluate(k)
        k = np.broadcast_to(np.asarray(k), self.shape)
        return sum(D @ (k * G) for D, G in zip(self.divergence_operator(), self.gradient_operator()))

    def gradient(self, f: NDArray) -> tuple[NDArray, ...]:
        r"""Gradient :math:`\nabla f` of a scalar field (one array per axis)."""
        f = self._check_field(f)
        return tuple(self.diff(f, 1, i) for i in range(self.ndims))

    def divergence(self, *v: NDArray) -> NDArray:
        r"""Divergence :math:`\nabla\cdot\mathbf{v}` of a vector field."""
        if len(v) != self.ndims:
            raise ValueError(f"Expected {self.ndims} vector components, got {len(v)}.")
        return sum(self.diff(self._check_field(vi, f"v[{i}]"), 1, i) for i, vi in enumerate(v))

    def curl(self, *v: NDArray) -> tuple[NDArray, ...] | NDArray:
        r"""Curl :math:`\nabla\times\mathbf{v}` of a vector field.

        Returns a tuple of three arrays on 3D grids and the scalar
        out-of-plane component on 2D grids.
        """
        if self.ndims not in (2, 3):
            raise ValueError("Curl is only defined for 2D and 3D grids.")
        if len(v) != self.ndims:
            raise ValueError(f"Expected {self.ndims} vector components, got {len(v)}.")
        v = [self._check_field(vi, f"v[{i}]") for i, vi in enumerate(v)]
        d = self.diff
        if self.ndims == 2:
            return d(v[1], 1, 0) - d(v[0], 1, 1)
        return (
            d(v[2], 1, 1) - d(v[1], 1, 2),
            d(v[0], 1, 2) - d(v[2], 1, 0),
            d(v[1], 1, 0) - d(v[0], 1, 1),
        )

    def integrate(self, f: NDArray) -> float:
        """Integrate a meshed function over the whole grid domain.

        On curvilinear grids the volume element is included.
        """
        from numgrids.api import integrate
        return integrate(self, f)

    @property
    def quadrature_weights(self) -> NDArray:
        """Quadrature weights of all grid points, as an array of grid shape.

        ``np.sum(grid.quadrature_weights * f)`` equals ``grid.integrate(f)``;
        on curvilinear grids the weights include the volume element.

        Examples
        --------
        >>> from numgrids import PolarGrid, radial, periodic
        >>> disk = PolarGrid(radial(8, 1.0), periodic(16))
        >>> round(float(disk.quadrature_weights.sum()), 10)         # area of the unit disk
        3.1415926536
        """
        if "weights" not in self.cache:
            from numgrids.integration import Integral
            if self.cache["integral"] is None:
                self.cache["integral"] = Integral(self)
            W = self.cache["integral"].weight_array
            W.setflags(write=False)
            self.cache["weights"] = W
        return self.cache["weights"]

    def inner(self, f: NDArray, g: NDArray) -> float | complex:
        r"""The :math:`L^2` inner product :math:`\langle f, g\rangle = \int \bar f\, g\, dV`.

        Examples
        --------
        >>> import numpy as np
        >>> from numgrids import Grid, periodic
        >>> grid = Grid(x=periodic(16))
        >>> x = grid.coords
        >>> round(grid.inner(np.sin(x), np.sin(x)), 10), round(abs(grid.inner(np.sin(x), np.cos(x))), 10)
        (3.1415926536, 0.0)
        """
        f = self._check_field(f, "f")
        g = self._check_field(g, "g")
        return self.integrate(np.conj(f) * g)

    def norm(self, f: NDArray, ord: int | float | str = 2) -> float:
        r"""Norm of a meshed function.

        Parameters
        ----------
        f : NDArray
            Meshed function of shape ``grid.shape``.
        ord : 2, 1, ``"max"`` or ``numpy.inf``
            ``2`` (default): the :math:`L^2` norm :math:`(\int |f|^2 dV)^{1/2}`;
            ``1``: the :math:`L^1` norm :math:`\int |f|\, dV`;
            ``"max"`` or ``numpy.inf``: the largest absolute value.

        Examples
        --------
        >>> import numpy as np
        >>> from numgrids import Grid, chebyshev
        >>> grid = Grid(x=chebyshev(20, 0, 2))
        >>> round(grid.norm(np.ones(grid.shape)), 10), grid.norm(grid.coords, "max")
        (1.4142135624, 2.0)
        """
        f = self._check_field(f)
        if ord == 2:
            return float(np.sqrt(abs(self.integrate(np.abs(f) ** 2))))
        if ord == 1:
            return float(self.integrate(np.abs(f)))
        if ord == "max" or ord == np.inf:
            return float(np.max(np.abs(f)))
        raise ValueError(f"Unknown norm: {ord!r}. Use 2, 1 or 'max'.")

    def mean(self, f: NDArray) -> float | complex:
        r"""The average :math:`\int f\, dV / \int dV` of a meshed function over the domain."""
        return self.integrate(f) / self.integrate(np.ones(self.shape))

    def interpolate(self, f: NDArray, locations, method: str = "cubic"):
        """Interpolate a meshed function at arbitrary locations.

        See :class:`~numgrids.interpol.Interpolator` for the accepted
        *locations* and *method* values.
        """
        from numgrids.api import interpolate
        return interpolate(self, f, locations, method=method)

    # ── plotting ────────────────────────────────────────────────

    def plot(self, f: NDArray | None = None, ax=None, **kwargs):
        """Visualize the grid points, or a meshed function on the grid.

        Without *f*, the grid points are shown (pairwise projections).

        With *f*, the function is plotted: as a line plot for 1D grids, as
        a color plot for 2D grids (in Cartesian space for curvilinear
        grids), and as a 2D slice for higher-dimensional grids.  See
        :func:`numgrids.plots.plot_field` for all options.

        Returns
        -------
        matplotlib.axes.Axes
            Only when *f* is given.
        """
        if f is not None:
            from numgrids.plots import plot_field
            return plot_field(self, f, ax=ax, **kwargs)
        self._plot_points()

    def _plot_points(self) -> None:
        from numgrids.plots import require_matplotlib
        plt = require_matplotlib()

        nplots = self.ndims * (self.ndims - 1) // 2
        fig = plt.figure(figsize=(12, nplots * 8))

        ctr = 0
        for i in range(self.ndims):
            for j in range(i):
                ctr += 1
                ax = fig.add_subplot(nplots, 1, ctr)
                X, Y = self.meshed_coords[i], self.meshed_coords[j]
                ax.plot(X.reshape(-1), Y.reshape(-1), "o", ms=1)

                if self.axes[i].name:
                    ax.set_xlabel(f"${self.axes[i].name}$")
                else:
                    ax.set_xlabel(f"axis-{i}")

                if self.axes[j].name:
                    ax.set_ylabel(f"${self.axes[j].name}$")
                else:
                    ax.set_ylabel(f"axis-{j}")

        plt.show()

    # ── boundary ────────────────────────────────────────────────

    @property
    def faces(self) -> dict:
        """Return a dictionary of all boundary faces.

        Keys follow the pattern ``'{axis_index}_{side}'``, e.g.
        ``'0_low'``, ``'0_high'``.  For named axes, ``'{name}_{side}'``
        (e.g. ``'x_low'``) is accepted as well.  Periodic axes are skipped,
        and radial axes through the origin only have a ``"high"`` face.
        """
        if self.cache["faces"] is None:
            from numgrids.boundary import BoundaryFace
            result = {}
            aliases = {}
            for i, axis in enumerate(self.axes):
                for side in axis.boundary_sides:
                    result[f"{i}_{side}"] = BoundaryFace(self, i, side)
                    if axis.name is not None:
                        aliases[f"{axis.name}_{side}"] = f"{i}_{side}"
            self.cache["faces"] = _FaceDict(result, aliases)
        return self.cache["faces"]

    @property
    def boundary(self) -> NDArray:
        """
        Returns a binary mask of the shape of the grid indicating the
        grid points on the boundary.
        """
        return self._boundary

    @property
    def interior(self) -> NDArray:
        """Boolean mask of the grid points *not* on the boundary."""
        return ~self._boundary

    # ── refinement ──────────────────────────────────────────────

    def refine(self) -> Grid:
        """
        Returns a new grid with twice the number of grid points in each direction.

        The grid type (e.g. :class:`~numgrids.SphericalGrid`) is preserved.
        """
        return self._new_like([axis.resized(len(axis) * 2) for axis in self.axes])

    def coarsen(self) -> Grid:
        """
        Returns a new grid with half the number of grid points in each direction.

        The grid type (e.g. :class:`~numgrids.SphericalGrid`) is preserved.
        """
        return self._new_like([axis.resized(len(axis) // 2) for axis in self.axes])

    def refine_axis(self, axis_index: int | str, factor: float = 2.0) -> Grid:
        """Return a new grid with one axis resized by the given factor.

        Parameters
        ----------
        axis_index : int or str
            Index (or name) of the axis to refine (or coarsen if *factor* < 1).
        factor : float
            Multiplicative factor for the number of points on the
            specified axis.  Values greater than 1 refine; values
            between 0 and 1 coarsen.  The result is rounded to the
            nearest integer (minimum 2).

        Returns
        -------
        Grid
            A new grid (of the same type) with the specified axis resized.

        Raises
        ------
        ValueError
            If *axis_index* is out of range or *factor* is not positive.
        """
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        if isinstance(axis_index, (int, np.integer)) and not (0 <= axis_index < self.ndims):
            raise ValueError(
                f"axis_index {axis_index} out of range for "
                f"{self.ndims}-dimensional grid"
            )
        axis_index = self.axis_index(axis_index)
        new_axes = list(self.axes)
        old_n = len(new_axes[axis_index])
        new_n = max(2, round(old_n * factor))
        new_axes[axis_index] = new_axes[axis_index].resized(new_n)
        return self._new_like(new_axes)

    # ── index helpers ───────────────────────────────────────────

    @property
    def meshed_indices(self) -> list[NDArray]:
        """
        Returns a tuple of length grid.ndims, where each item in the
        tuple is an array of shape grid.shape. Each item stores the
        indices of each grid point for a given axis.

        For example, if you have a 2D grid with 4 points along axis 0
        and 3 points along axis 1, the indices along axis 0 are [0, 1, 2, 3],
        the indices along axis 1 are [0, 1, 2]. meshed_indices returns
        a tuple (I, J), where I and J both have shape (4, 3). I contains
        the "slot-0" indices, J contains the "slot-1" indices.

        This generalizes for grids in higher dimensions, of course.
        """
        return np.meshgrid(*[np.arange(len(axis)) for axis in self.axes], indexing="ij")

    @property
    def index_tuples(self) -> NDArray:
        """
        Returns an array A of shape ``(*grid.shape, grid.ndims)``.

        For example, in case of a 2D grid, A[i, j] contains the index
        tuple (i, j).
        """
        return self._to_tuple_field(*self.meshed_indices)

    @property
    def coord_tuples(self) -> NDArray:
        """
        Returns an array A of shape ``(*grid.shape, grid.ndims)`` with
        all the coordinate tuples.

        For example, in case of a 2D grid, A[i, j] contains the
        tuple (x, y), where x and y are the coordinates of the grid point
        (i, j).
        """
        return self._to_tuple_field(*self.meshed_coords)

    def _to_tuple_field(self, *arrs: NDArray) -> NDArray:
        return np.vstack([arr.reshape(-1) for arr in arrs]).T.reshape(
            *self.shape, -1
        )

    # ── I/O and interoperability ────────────────────────────────

    def save(self, path: str | Path, **data: NDArray) -> None:
        """Save the grid (and optional meshed arrays) to a ``.npz`` file.

        See :func:`numgrids.save_grid`.
        """
        from numgrids.io import save_grid
        save_grid(path, self, **data)

    @staticmethod
    def load(path: str | Path) -> tuple[Grid, dict[str, NDArray]]:
        """Load a grid and its data arrays from a ``.npz`` file.

        See :func:`numgrids.load_grid`.
        """
        from numgrids.io import load_grid
        return load_grid(path)

    def to_xarray(self, **fields: NDArray):
        """Convert meshed arrays on this grid to an :class:`xarray.Dataset`.

        See :func:`numgrids.to_xarray`.
        """
        from numgrids.io import to_xarray
        return to_xarray(self, **fields)

    @staticmethod
    def from_xarray(obj) -> tuple[Grid, dict[str, NDArray]]:
        """Reconstruct a grid (and data) from an xarray object.

        See :func:`numgrids.from_xarray`.
        """
        from numgrids.io import from_xarray
        return from_xarray(obj)


class MultiGrid:
    """
    Represents a multigrid, a hierarchical set of grids of different resolutions
    for the same domain.
    """

    def __init__(self, *axes, min_size: int = 2) -> None:
        """
        Constructor.

        The grids of the various resolutions are stored in the property `levels`.

        Parameters
        ----------
        axes: one or more Axis objects
            The axes for the grid with highest resolution. Starting from the grid
            with the highest resolution (level 0), the constructor will create different levels
            of resolution. Each additional level contains a grid with about half the grid
            points in each direction.  All axis types are supported
            (``EquidistantAxis``, ``ChebyshevAxis``, ``LogAxis``, and
            mixtures thereof).  Axis properties such as ``periodic`` and
            ``name`` are preserved at every level.

        min_size: positive int
            This parameter sets a lower bound for the coarsest grid in the multigrid.
            At the lowest resolution, each axis must have at least min_size grid points.
        """
        grid = Grid(*axes)
        self._levels = [grid]
        while True:
            sizes = np.array(grid.shape)
            # size -> size / 2 if size is even and size -> (size+1)/2 if it is odd:
            sizes = sizes // 2 + sizes % 2
            if min(sizes) < min_size:
                break
            axes = [axis.resized(size) for size, axis in zip(sizes, axes)]
            grid = Grid(*axes)
            self._levels.append(grid)

    @property
    def levels(self) -> list[Grid]:
        """
        Returns a list of grids with different resolution levels.

        levels[0] => finest grid
        levels[-1] => coarsest grid
        """
        return self._levels

    def transfer(self, f: NDArray, level_from: int, level_to: int,
                 method: str = "linear") -> NDArray:
        """
        Coarsen or refine an array on a grid by one level.

        Parameters
        ----------
        f: numpy.ndarray
            An array given on the grid at level = level_from.
        level_from: int
            The level of the grid on which f is given.
        level_to: int
            The level to which f shall be refined or coarsened.
        method: str
            Coarsening and refinement involves interpolation. This
            is the interpolation order.

        Returns
        -------
        The coarsened or refined array.
        """
        from numgrids.interpol import Interpolator
        if f.shape != self.levels[level_from].shape:
            raise ValueError(
                f"Array shape {f.shape} does not match grid shape "
                f"{self.levels[level_from].shape} at level {level_from}."
            )
        if abs(level_from - level_to) != 1:
            raise ValueError("Can only transfer between adjacent levels.")

        grid_fr = self.levels[level_from]
        grid_to = self.levels[level_to]

        interp = Interpolator(grid_fr, f, method=method)
        f_trans = interp(grid_to)

        return f_trans

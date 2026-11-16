"""Axis types for numerical grids.

An :class:`Axis` is the fundamental building block in numgrids. It represents
a single coordinate axis in a (possibly curvilinear) coordinate system.

Each axis type uses an *internal* coordinate system (e.g. ``[0, 1]`` or
Chebyshev nodes on ``[-1, 1]``) and maps it to *external* (user-specified)
coordinates via a linear or nonlinear transformation. The axis type also
determines which differentiation strategy is used.

Available axis types:

- :class:`EquidistantAxis` -- uniformly spaced points, optional periodicity.
- :class:`ChebyshevAxis` -- Chebyshev-node spacing for spectral accuracy.
- :class:`LogAxis` -- logarithmically spaced points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numgrids.diff import GridDiff
    from numgrids.grids import Grid


class Axis:
    """Base class for all axis types.

    An axis represents a single coordinate dimension in a numerical grid.
    It stores the coordinate values and knows how to create the appropriate
    differentiation operator for its discretization strategy.

    Subclasses must implement :meth:`_setup_internal_coords` and
    :meth:`create_diff_operator`.
    """

    def __init__(self, num_points: int, low: float, high: float, periodic: bool, **kwargs) -> None:
        """Base class constructor.

        Parameters
        ----------
        num_points: positive int
            Number of grid points along the axis.
        low: float
            The lower end coordinate value.  Stored as :attr:`low`.
        high: float
            The upper end coordinate value.  Stored as :attr:`high`.
        periodic: bool
            Apply periodic boundary condition or not.
        """
        if isinstance(num_points, bool) or int(num_points) != num_points:
            raise TypeError(f"num_points must be an integer, not {num_points!r}")
        num_points = int(num_points)
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, not {num_points}")
        if num_points < 2:
            raise ValueError(f"An axis needs at least 2 points, got {num_points}")
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError(f"Axis bounds must be finite, got low={low}, high={high}")
        if high <= low:
            raise ValueError(f"high ({high}) must be greater than low ({low})")
        unknown = set(kwargs) - {"name"}
        if unknown:
            raise TypeError(f"Unexpected keyword argument(s): {', '.join(sorted(unknown))}")
        self._num_points = num_points
        self.low = low
        self.high = high
        self.periodic = bool(periodic)
        self._coords_internal = self._setup_internal_coords(low, high)
        self._coords = self._setup_external_coords(low, high)
        self.name: str | None = kwargs.get("name")
        self._quadrature_weights: NDArray | None = None

    def resized(self, num_points: int) -> Axis:
        """Return a new axis of the same type and domain with a different number of points.

        Parameters
        ----------
        num_points : int
            Number of grid points for the new axis.

        Returns
        -------
        Axis
            A new axis instance with the same type, domain, and metadata.
        """
        return self._clone(num_points=num_points)

    def renamed(self, name: str | None) -> Axis:
        """Return a copy of this axis with a different name.

        Parameters
        ----------
        name : str or None
            The new axis name.

        Returns
        -------
        Axis
        """
        return self._clone(name=name)

    _UNSET = object()

    def _clone(self, num_points: int | None = None, name=_UNSET) -> Axis:
        kwargs: dict = {}
        if self.periodic:
            kwargs["periodic"] = True
        name = self.name if name is Axis._UNSET else name
        if name is not None:
            kwargs["name"] = name
        n = len(self) if num_points is None else num_points
        return type(self)(n, self.low, self.high, **kwargs)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Axis):
            return NotImplemented
        return (
            type(self) is type(other)
            and len(self) == len(other)
            and self.low == other.low
            and self.high == other.high
            and self.periodic == other.periodic
            and self.name == other.name
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, len(self), self.low, self.high, self.periodic, self.name))

    @property
    def period(self) -> float:
        """Length of the period ``high - low`` (periodic axes only)."""
        if not self.periodic:
            raise AttributeError("Only periodic axes have a period.")
        return self.high - self.low

    @property
    def quadrature_weights(self) -> NDArray:
        r"""Quadrature weights :math:`w_k` such that :math:`\int f\,dx \approx \sum_k w_k f(x_k)`.

        The weights are computed once and cached.  Subclasses provide
        method-specific weights (Clenshaw--Curtis for Chebyshev axes,
        trapezoidal for periodic axes); the generic fallback inverts the
        axis' own differentiation matrix, so that integration has the
        same order of accuracy as differentiation.
        """
        if self._quadrature_weights is None:
            self._quadrature_weights = self._compute_quadrature_weights()
            self._quadrature_weights.flags.writeable = False
        return self._quadrature_weights

    def _compute_quadrature_weights(self) -> NDArray:
        import scipy.sparse
        import scipy.sparse.linalg

        from numgrids.grids import Grid

        # The antiderivative F with F(x_0) = 0 satisfies D F = f.  Solving
        # on the points 1..N and reading off F(x_N) is a linear functional
        # of f, whose coefficients are the quadrature weights.
        D = self.create_diff_operator(Grid(self), 1, 0).as_matrix()
        D = scipy.sparse.csc_matrix(D)[1:, 1:]
        e_last = np.zeros(D.shape[0])
        e_last[-1] = 1.0
        w = scipy.sparse.linalg.spsolve(D.T.tocsc(), e_last)
        return np.concatenate([[0.0], np.atleast_1d(w)])

    def _setup_internal_coords(self, low: float, high: float) -> NDArray:
        """Create the internal (canonical) coordinate array.

        Parameters
        ----------
        low : float
            Lower bound of the domain.
        high : float
            Upper bound of the domain.

        Returns
        -------
        NDArray
            1-D array of internal coordinate values.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def _setup_external_coords(self, low: float, high: float) -> NDArray:
        """Map internal coordinates to user-facing (external) coordinates.

        The default implementation applies a linear transformation.
        Subclasses may override for nonlinear mappings (e.g. logarithmic).

        Parameters
        ----------
        low : float
            Lower bound of the user domain.
        high : float
            Upper bound of the user domain.

        Returns
        -------
        NDArray
            1-D array of external coordinate values.
        """
        return self._coords_internal * (high - low) + low

    def create_diff_operator(self, grid: Grid, order: int, axis_index: int, acc: int = 4) -> GridDiff:
        """Create the appropriate differentiation operator for this axis type.

        Parameters
        ----------
        grid: Grid
            The numerical grid.
        order: int
            The derivative order.
        axis_index: int
            The axis index.
        acc: int
            The accuracy order for finite-difference based methods.

        Must be implemented by child classes.
        """
        raise NotImplementedError("Must be implemented by child class.")

    @property
    def boundary(self) -> slice:
        """Slice selecting interior points (excluding boundary).

        For periodic axes the full range is returned because there are no
        boundary points. For non-periodic axes the first and last point
        are considered boundary.

        Returns
        -------
        slice
        """
        if self.periodic:
            return slice(None, None)
        else:
            return slice(1, -1)

    def __len__(self) -> int:
        """Returns the number of grid points on the axis."""
        return self._num_points

    def __getitem__(self, idx: int) -> float:
        """Access coordinate value for given index. Is aware of periodic boundary conditions if applied."""
        if self.periodic:
            return self._coords[idx % self._num_points]
        return self._coords[idx]

    @property
    def coords(self) -> NDArray:
        """Returns a 1D array of the coordinate values of the axis, as specified by the user."""
        return self._coords

    @property
    def coords_internal(self) -> NDArray:
        """
        Most Axis types use an internal coordinate system, typically scaled to some
        standard interval. Refer to the child class implementation for details."""
        return self._coords_internal

    def __str__(self) -> str:
        return f"{type(self).__name__}({len(self)} points from {self.coords[0]} to {self.coords[-1]})"

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return (
            f"{type(self).__name__}(num_points={len(self)}, "
            f"low={self.low}, high={self.high}, periodic={self.periodic}{name})"
        )

    def plot(self) -> None:
        """Visualize the axis points using matplotlib."""
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
        ax.set_ylim(-0.01, 0.01)
        ax.set_axis_off()
        ax.plot(self.coords, [0] * len(self), "o")
        ax.plot(self.coords, [0] * len(self), "-k")
        ax.annotate(f"$x_0 = {self.coords[0]}$", xy=(self.coords[0], 0),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    xytext=(0, 30), textcoords='offset points')
        ax.annotate(f"$x_{{{len(self) - 1}}} = {self.coords[-1]}$", xy=(self.coords[-1], 0),
                    arrowprops=dict(facecolor='black'),
                    xytext=(0, 30), textcoords='offset points')
        plt.show()


class EquidistantAxis(Axis):
    """Axis with uniformly spaced grid points.

    Supports both non-periodic and periodic boundary conditions. When periodic,
    the FFT spectral differentiation method is used; otherwise classical
    finite differences are employed.

    For a periodic axis the last point is *not* included (it would coincide
    with the first point after wrapping).
    """

    def __init__(self, num_points: int, low: float = 0, high: float = 1,
                 periodic: bool = False, **kwargs) -> None:
        """Create an equidistant axis.

        Parameters
        ----------
        num_points : int
            Number of grid points.
        low : float, optional
            Lower coordinate bound (default ``0``).
        high : float, optional
            Upper coordinate bound (default ``1``).
        periodic : bool, optional
            Whether to apply periodic boundary conditions (default ``False``).
        **kwargs
            Additional keyword arguments forwarded to :class:`Axis`
            (e.g. ``name``).
        """
        super().__init__(num_points, low, high, periodic, **kwargs)

    def _setup_internal_coords(self, *args) -> NDArray:
        """Internal coordinates on ``[0, 1]``."""
        return np.linspace(0, 1, len(self), endpoint=not self.periodic)

    def create_diff_operator(self, grid: Grid, order: int, axis_index: int, acc: int = 4) -> GridDiff:
        """Create the differentiation operator for this axis.

        Dispatches to :class:`~numgrids.diff.FFTDiff` for periodic axes and
        :class:`~numgrids.diff.FiniteDifferenceDiff` otherwise.
        """
        from numgrids.diff import FFTDiff, FiniteDifferenceDiff
        if self.periodic:
            return FFTDiff(grid, order, axis_index, acc=acc)
        return FiniteDifferenceDiff(grid, order, axis_index, acc=acc)

    @property
    def spacing(self) -> float:
        """The grid spacing."""
        return self._coords[1] - self._coords[0]

    def _compute_quadrature_weights(self) -> NDArray:
        if self.periodic:
            # The trapezoidal rule is spectrally accurate for periodic functions.
            return np.full(len(self), self.spacing)
        return super()._compute_quadrature_weights()

    def plot(self) -> None:
        """Visualize the axis points using matplotlib."""
        if not self.periodic:
            super().plot()
            return
        import matplotlib.patches as mp
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        angles = np.linspace(0, 2 * np.pi, len(self), endpoint=False)
        xx = np.cos(angles)
        yy = np.sin(angles)
        ax.plot(xx, yy, "o")
        ax.add_patch(mp.Circle((0, 0), 1, fill=False))

        style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color="k")

        arrow = mp.FancyArrowPatch((xx[1] * 1.1, yy[1] * 1.1), (xx[3] * 1.1, yy[3] * 1.1),
                                   connectionstyle="arc3,rad=0.2", **kw)

        ax.add_patch(arrow)
        ax.annotate(f"$x_0 = {self.coords[0]} = x_{{{len(self)}}}$", xy=(xx[0], yy[0]),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    xytext=(30, 30), textcoords='offset points')
        ax.annotate(f"$x_{{{len(self) - 1}}} = {self.coords[-1]}$", xy=(xx[-1], yy[-1]),
                    arrowprops=dict(facecolor='black'),
                    xytext=(30, -30), textcoords='offset points')
        plt.show()


class ChebyshevAxis(Axis):
    r"""Axis with grid points at Chebyshev nodes.

    The Chebyshev nodes are defined as

    .. math::
        x_k = \cos\!\left(\frac{k\pi}{N-1}\right), \quad k = 0, \ldots, N-1

    on the canonical interval ``[-1, 1]``, then linearly mapped to
    ``[low, high]``.  Points cluster near the boundaries, which avoids the
    Runge phenomenon and enables spectral accuracy for smooth, non-periodic
    functions.
    """

    def __init__(self, num_points: int, low: float = 0, high: float = 1, **kwargs) -> None:
        """Constructor

        Parameters
        ----------
        num_points: positive int
            Number of grid points
        low: float
            Smallest coordinate value
        high: float
            Greatest coordinate value (included).
        """
        super().__init__(num_points, low, high, periodic=False, **kwargs)

    def _setup_internal_coords(self, *args) -> NDArray:
        # sin form of cos(k pi / N): exactly symmetric and exact at the endpoints
        N = len(self) - 1
        return np.sin(np.pi * (N - 2 * np.arange(N + 1)) / (2 * N))

    def _compute_quadrature_weights(self) -> NDArray:
        """Clenshaw--Curtis weights, mapped to ``[low, high]``."""
        N = len(self) - 1
        theta = np.pi * np.arange(N + 1) / N
        w = np.zeros(N + 1)
        v = np.ones(N - 1)
        inner = theta[1:-1]
        if N % 2 == 0:
            w[0] = w[N] = 1.0 / (N ** 2 - 1)
            for k in range(1, N // 2):
                v -= 2 * np.cos(2 * k * inner) / (4 * k ** 2 - 1)
            v -= np.cos(N * inner) / (N ** 2 - 1)
        else:
            w[0] = w[N] = 1.0 / N ** 2
            for k in range(1, (N - 1) // 2 + 1):
                v -= 2 * np.cos(2 * k * inner) / (4 * k ** 2 - 1)
        w[1:-1] = 2 * v / N
        # The weights are symmetric, so the node ordering does not matter.
        return w * (self.high - self.low) / 2

    def _setup_external_coords(self, low: float, high: float) -> NDArray:
        coords = (self._coords_internal[::-1] + 1) / 2
        coords = coords * (high - low) + low
        coords[0], coords[-1] = low, high
        return coords

    def create_diff_operator(self, grid: Grid, order: int, axis_index: int, acc: int = 4) -> GridDiff:
        """Create Chebyshev spectral differentiation operator.

        Note: acc is accepted for API consistency but not used
        by the Chebyshev spectral method.
        """
        from numgrids.diff import ChebyshevDiff
        return ChebyshevDiff(grid, order, axis_index)


class LogAxis(Axis):
    r"""Axis with logarithmically spaced grid points.

    Internally the coordinate is transformed to a uniform grid on
    ``[ln(low), ln(high)]``.  Differentiation uses finite differences on the
    log-scale and applies the chain rule

    .. math::
        \frac{df}{dx} = \frac{1}{x}\,\frac{df}{d(\ln x)}

    This is useful for problems where fine resolution is needed near the
    lower boundary (e.g. radial coordinates close to zero).

    ``low`` must be strictly positive.
    """

    def __init__(self, num_points: int, low: float, high: float, **kwargs) -> None:
        """Constructor for LogAxis

        Parameters
        ----------
        num_points: positive int
            Number of grid points (endpoint included)
        low: float
            Lowest coordinate value
        high: float
            Highest coordinate value
        """
        if low <= 0:
            raise ValueError("LogAxis requires positive lower boundary.")
        super().__init__(num_points, low, high, periodic=False, **kwargs)

    def _setup_internal_coords(self, low: float, high: float) -> NDArray:
        """The internal coordinates are equidistant."""
        return np.linspace(np.log(low), np.log(high), len(self))

    def _setup_external_coords(self, low: float, high: float) -> NDArray:
        """The external coordinates are as expected by the user."""
        return np.logspace(np.log10(low), np.log10(high), len(self))

    def create_diff_operator(self, grid: Grid, order: int, axis_index: int, acc: int = 6) -> GridDiff:
        """Create logarithmic differentiation operator."""
        from numgrids.diff import LogDiff
        return LogDiff(grid, order, axis_index, acc=acc)

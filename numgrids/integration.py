"""Integration over the whole grid domain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numgrids.grids import Grid


class Integral:
    r"""The integration operator for integrating over a grid.

    .. math::
        \int_V f \, dV

    Integration always runs over the whole grid.  The integral is computed
    as a tensor product of one-dimensional quadrature rules, chosen per
    axis type:

    * Chebyshev axes: Clenshaw--Curtis quadrature (spectral accuracy),
    * periodic equidistant axes: trapezoidal rule (spectral accuracy),
    * other axes: weights obtained by inverting the axis' differentiation
      matrix (same order of accuracy as differentiation).

    On curvilinear grids (:class:`~numgrids.CurvilinearGrid` and its
    subclasses) the integrand is multiplied by the volume element
    :math:`J = h_1 h_2 \cdots h_n` by default, so that e.g. integrating
    ``1`` over a :class:`~numgrids.SphericalGrid` yields the volume of the
    domain.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, Integral, chebyshev
    >>> grid = Grid(chebyshev(20, 0, np.pi))
    >>> I = Integral(grid)
    >>> round(I(np.sin(grid.coords)), 10)
    2.0
    """

    def __init__(self, grid: Grid, volume_element: bool = True) -> None:
        r"""Constructor

        Parameters
        ----------
        grid: Grid
            The grid on which to integrate.
        volume_element: bool
            Whether to include the volume element (Jacobian) of curvilinear
            grids.  If ``False``, the integral is taken in coordinate space,
            i.e. :math:`\int f\, dq_1 \cdots dq_n`.  Has no effect on
            Cartesian grids.  Default is ``True``.
        """
        self.grid = grid
        self.volume_element = volume_element
        self.weights = [axis.quadrature_weights for axis in grid.axes]
        J = getattr(grid, "volume_element", None) if volume_element else None
        self._jacobian = None if J is None or np.isscalar(J) else np.asarray(J)

    def __call__(self, f: NDArray) -> float:
        """
        Apply the integration to a meshed function.

        Parameters
        ----------
        f: numpy.ndarray
            The meshed function to integrate over. The shape must be compatible with
            the shape of the grid.

        Returns
        -------
        The result of the integration (float, or complex for complex input).
        """
        f = np.asarray(f)
        if f.shape != self.grid.shape:
            raise ValueError(
                f"Array of shape {f.shape} does not match grid shape {self.grid.shape}."
            )
        if self._jacobian is not None:
            f = f * self._jacobian
        result = f
        for w in reversed(self.weights):
            result = result @ w
        if np.iscomplexobj(result):
            return complex(result)
        return float(result)

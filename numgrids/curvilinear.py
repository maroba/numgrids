"""Curvilinear coordinate grids with auto-generated vector calculus operators.

An orthogonal curvilinear coordinate system is fully characterised by its
*scale factors* :math:`h_i(\\mathbf{q})`.  Given those, the gradient,
divergence, Laplacian and curl follow from standard differential-geometry
identities.

This module provides :class:`CurvilinearGrid`, a :class:`~numgrids.grids.Grid`
subclass that accepts user-supplied scale-factor callables and automatically
constructs the corresponding operators.

Built-in coordinate systems (:class:`~numgrids.api.SphericalGrid`,
:class:`~numgrids.api.CylindricalGrid`, :class:`~numgrids.api.PolarGrid`)
are thin subclasses of ``CurvilinearGrid``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from numgrids.grids import Grid


ScaleFactorFn = Callable[[tuple[NDArray, ...]], NDArray]
"""Type alias: a scale factor receives meshed coordinates and returns an array."""


class CurvilinearGrid(Grid):
    r"""Orthogonal curvilinear grid with auto-generated vector calculus.

    The user supplies one *scale-factor callable* per axis.  Each callable
    has the signature ``(coords: tuple[NDArray, ...]) -> NDArray``, where
    *coords* is ``self.meshed_coords``.

    From the scale factors :math:`h_i` the class derives:

    * **gradient** —
      :math:`(\nabla f)_i = \frac{1}{h_i}\frac{\partial f}{\partial q_i}`

    * **divergence** —
      :math:`\nabla\!\cdot\!\mathbf{v}
      = \frac{1}{J}\sum_i \frac{\partial}{\partial q_i}
        \!\left(\frac{J}{h_i}\,v_i\right)`,
      :math:`J = \prod h_i`

    * **Laplacian** —
      :math:`\nabla^2 f
      = \frac{1}{J}\sum_i \frac{\partial}{\partial q_i}
        \!\left(\frac{J}{h_i^2}\,\frac{\partial f}{\partial q_i}\right)`

    * **curl** (3D only) —
      :math:`(\nabla\!\times\!\mathbf{v})_i
      = \frac{1}{h_j h_k}\!\left[
        \frac{\partial(h_k\,v_k)}{\partial q_j}
        - \frac{\partial(h_j\,v_j)}{\partial q_k}\right]`

    For **2D** grids a scalar *curl* is provided, equal to the out-of-plane
    component.

    Coordinate singularities (e.g. *r = 0*) are handled gracefully:
    non-finite values are replaced by zero.

    Parameters
    ----------
    *axes : Axis
        One or more :class:`~numgrids.axes.Axis` objects.
    scale_factors : sequence of callables
        One callable per axis.  ``scale_factors[i](meshed_coords)`` must
        return an :class:`~numpy.ndarray` of shape ``grid.shape``.

    Raises
    ------
    ValueError
        If the number of scale factors does not match the number of axes.

    Examples
    --------
    >>> from numgrids import CurvilinearGrid, create_axis, AxisType
    >>> import numpy as np
    >>> r_ax = create_axis(AxisType.CHEBYSHEV, 25, 0.1, 2)
    >>> phi_ax = create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2 * np.pi)
    >>> z_ax = create_axis(AxisType.CHEBYSHEV, 25, -1, 1)
    >>> grid = CurvilinearGrid(
    ...     r_ax, phi_ax, z_ax,
    ...     scale_factors=(
    ...         lambda c: np.ones_like(c[0]),   # h_r = 1
    ...         lambda c: c[0],                  # h_phi = r
    ...         lambda c: np.ones_like(c[0]),   # h_z = 1
    ...     ),
    ... )
    >>> R, Phi, Z = grid.meshed_coords
    >>> lap = grid.laplacian(R ** 2 + Z ** 2)  # should be ~6
    """

    def __init__(
        self,
        *axes,
        scale_factors: Sequence[ScaleFactorFn],
    ) -> None:
        super().__init__(*axes)
        if len(scale_factors) != self.ndims:
            raise ValueError(
                f"Expected {self.ndims} scale factors, got {len(scale_factors)}."
            )
        self._scale_factor_fns = tuple(scale_factors)
        self._diff_ops: dict | None = None
        self._h_arrays: tuple[NDArray, ...] | None = None
        self._jacobian: NDArray | None = None

    # ── lazy initialisation ───────────────────────────────────────

    def _evaluate_scale_factors(self) -> tuple[NDArray, ...]:
        """Evaluate and cache scale-factor arrays on the mesh."""
        if self._h_arrays is None:
            coords = self.meshed_coords
            self._h_arrays = tuple(h(coords) for h in self._scale_factor_fns)
        return self._h_arrays

    def _evaluate_jacobian(self) -> NDArray:
        """Compute and cache the Jacobian J = h_1 * h_2 * ... * h_n."""
        if self._jacobian is None:
            h = self._evaluate_scale_factors()
            self._jacobian = np.ones(self.shape)
            for hi in h:
                self._jacobian = self._jacobian * hi
        return self._jacobian

    def _ensure_diff_ops(self) -> None:
        """Lazily create first-derivative operators for every axis."""
        if self._diff_ops is not None:
            return
        from numgrids.api import Diff
        self._diff_ops = {}
        for i in range(self.ndims):
            self._diff_ops[i] = Diff(self, 1, i)

    # ── vector calculus operators ─────────────────────────────────

    def gradient(self, f: NDArray) -> tuple[NDArray, ...]:
        r"""Gradient of a scalar field.

        .. math::
            (\nabla f)_i = \frac{1}{h_i}\,\frac{\partial f}{\partial q_i}

        Parameters
        ----------
        f : NDArray
            Scalar field of shape ``grid.shape``.

        Returns
        -------
        tuple of NDArray
            Physical gradient components, one per axis.
        """
        self._ensure_diff_ops()
        h = self._evaluate_scale_factors()
        components = []
        for i in range(self.ndims):
            with np.errstate(divide="ignore", invalid="ignore"):
                comp = (1.0 / h[i]) * self._diff_ops[i](f)
            comp = np.where(np.isfinite(comp), comp, 0.0)
            components.append(comp)
        return tuple(components)

    def divergence(self, *v: NDArray) -> NDArray:
        r"""Divergence of a vector field.

        .. math::
            \nabla\!\cdot\!\mathbf{v}
            = \frac{1}{J}\sum_i \frac{\partial}{\partial q_i}
              \!\left(\frac{J}{h_i}\,v_i\right)

        Parameters
        ----------
        *v : NDArray
            Physical components of the vector field (one per axis).

        Returns
        -------
        NDArray
            The scalar divergence field.

        Raises
        ------
        ValueError
            If the number of components does not match the grid dimension.
        """
        if len(v) != self.ndims:
            raise ValueError(
                f"Expected {self.ndims} vector components, got {len(v)}."
            )
        self._ensure_diff_ops()
        h = self._evaluate_scale_factors()
        J = self._evaluate_jacobian()
        result = np.zeros(self.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(self.ndims):
                coeff = J / h[i]
                result = result + self._diff_ops[i](coeff * v[i])
            result = result / J
        return np.where(np.isfinite(result), result, 0.0)

    def laplacian(self, f: NDArray) -> NDArray:
        r"""Laplacian of a scalar field.

        .. math::
            \nabla^2 f
            = \frac{1}{J}\sum_i \frac{\partial}{\partial q_i}
              \!\left(\frac{J}{h_i^2}\,\frac{\partial f}{\partial q_i}\right)

        Parameters
        ----------
        f : NDArray
            Scalar field of shape ``grid.shape``.

        Returns
        -------
        NDArray
            The Laplacian, same shape as *f*.
        """
        self._ensure_diff_ops()
        h = self._evaluate_scale_factors()
        J = self._evaluate_jacobian()
        result = np.zeros(self.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(self.ndims):
                inner = (J / h[i] ** 2) * self._diff_ops[i](f)
                result = result + self._diff_ops[i](inner)
            result = result / J
        return np.where(np.isfinite(result), result, 0.0)

    def curl(self, *v: NDArray) -> tuple[NDArray, ...] | NDArray:
        r"""Curl of a vector field.

        **3D grids** — returns a tuple of three arrays (one per component):

        .. math::
            (\nabla\!\times\!\mathbf{v})_i
            = \frac{1}{h_j h_k}\!\left[
              \frac{\partial(h_k\,v_k)}{\partial q_j}
              - \frac{\partial(h_j\,v_j)}{\partial q_k}\right]

        where :math:`(i, j, k)` is a cyclic permutation of :math:`(0, 1, 2)`.

        **2D grids** — returns a scalar array (the out-of-plane component):

        .. math::
            (\nabla\!\times\!\mathbf{v})_z
            = \frac{1}{h_0 h_1}\!\left[
              \frac{\partial(h_1\,v_1)}{\partial q_0}
              - \frac{\partial(h_0\,v_0)}{\partial q_1}\right]

        Parameters
        ----------
        *v : NDArray
            Physical components of the vector field.

        Returns
        -------
        tuple of NDArray (3D) or NDArray (2D)
            The curl components.

        Raises
        ------
        ValueError
            If the grid dimension is not 2 or 3, or if the number of
            components is wrong.
        """
        if self.ndims == 2:
            return self._curl_2d(*v)
        if self.ndims == 3:
            return self._curl_3d(*v)
        raise ValueError("Curl is only defined for 2D and 3D grids.")

    def _curl_2d(self, v0: NDArray, v1: NDArray) -> NDArray:
        """Scalar curl for 2D curvilinear grids."""
        self._ensure_diff_ops()
        h = self._evaluate_scale_factors()
        d0 = self._diff_ops[0]
        d1 = self._diff_ops[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (1.0 / (h[0] * h[1])) * (
                d0(h[1] * v1) - d1(h[0] * v0)
            )
        return np.where(np.isfinite(result), result, 0.0)

    def _curl_3d(self, v0: NDArray, v1: NDArray, v2: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Vector curl for 3D curvilinear grids."""
        self._ensure_diff_ops()
        h = self._evaluate_scale_factors()
        v = (v0, v1, v2)
        d = [self._diff_ops[i] for i in range(3)]

        components: list[NDArray] = []
        cyclic = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        for i, j, k in cyclic:
            with np.errstate(divide="ignore", invalid="ignore"):
                comp = (1.0 / (h[j] * h[k])) * (
                    d[j](h[k] * v[k]) - d[k](h[j] * v[j])
                )
            comp = np.where(np.isfinite(comp), comp, 0.0)
            components.append(comp)
        return tuple(components)

"""Curvilinear coordinate grids with auto-generated vector calculus operators.

An orthogonal curvilinear coordinate system is fully characterised by its
*scale factors* :math:`h_i(\\mathbf{q})`.  Given those, the gradient,
divergence, Laplacian and curl follow from standard differential-geometry
identities.

This module provides :class:`CurvilinearGrid`, a :class:`~numgrids.grids.Grid`
subclass that accepts user-supplied scale-factor callables and automatically
constructs the corresponding operators, as well as the built-in coordinate
systems

* :class:`PolarGrid` -- :math:`(r, \\varphi)`,
* :class:`CylindricalGrid` -- :math:`(r, \\varphi, z)`,
* :class:`SphericalGrid` -- :math:`(r, \\theta, \\varphi)`,
* :class:`EllipticCylindricalGrid` -- :math:`(\\mu, \\nu, z)`,
* :class:`ProlateSpheroidalGrid` -- :math:`(\\mu, \\nu, \\varphi)`,
* :class:`OblateSpheroidalGrid` -- :math:`(\\mu, \\nu, \\varphi)`.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
from numpy.typing import NDArray

from numgrids.axes import Axis, ColatitudeAxis, RadialAxis, UnboundedAxis
from numgrids.grids import Grid

if TYPE_CHECKING:
    from numgrids.operators import Operator


#: Axis types whose derivatives fold the domain across a coordinate singularity.
_FOLDED_AXES = (RadialAxis, ColatitudeAxis)

ScaleFactorFn = Callable[[tuple[NDArray, ...]], NDArray]
"""Type alias: a scale factor receives meshed coordinates and returns an array."""

CartesianFn = Callable[[tuple[NDArray, ...]], tuple[NDArray, ...]]
"""Type alias: maps meshed curvilinear coordinates to Cartesian coordinates."""


def _finite_or_zero(a: NDArray) -> NDArray:
    return np.where(np.isfinite(a), a, 0.0)


class CurvilinearGrid(Grid):
    r"""Orthogonal curvilinear grid with auto-generated vector calculus.

    The user supplies one *scale-factor callable* per axis.  Each callable
    has the signature ``(coords: tuple[NDArray, ...]) -> NDArray``, where
    *coords* is ``self.meshed_coords``.

    From the scale factors :math:`h_i` the class derives:

    * **gradient**

      .. math::

         (\nabla f)_i = \frac{1}{h_i}\frac{\partial f}{\partial q_i}

    * **divergence** (where :math:`J = \prod h_i`)

      .. math::

         \nabla\!\cdot\!\mathbf{v}
         = \frac{1}{J}\sum_i \frac{\partial}{\partial q_i}
           \!\left(\frac{J}{h_i}\,v_i\right)

    * **Laplacian**

      .. math::

         \nabla^2 f
         = \frac{1}{J}\sum_i \frac{\partial}{\partial q_i}
           \!\left(\frac{J}{h_i^2}\,\frac{\partial f}{\partial q_i}\right)

    * **curl** (3D only)

      .. math::

         (\nabla\!\times\!\mathbf{v})_i
         = \frac{1}{h_j h_k}\!\left[
           \frac{\partial(h_k\,v_k)}{\partial q_j}
           - \frac{\partial(h_j\,v_j)}{\partial q_k}\right]

    For **2D** grids a scalar *curl* is provided, equal to the out-of-plane
    component.

    Integration (:meth:`integrate`) includes the volume element
    :math:`J`.

    Coordinate singularities (e.g. ``r = 0``) are handled gracefully:
    non-finite values are replaced by zero.

    Parameters
    ----------
    \*axes : Axis
        One or more :class:`~numgrids.axes.Axis` objects.
    scale_factors : sequence of callables
        One callable per axis.  ``scale_factors[i](meshed_coords)`` must
        return an :class:`~numpy.ndarray` of shape ``grid.shape``.
    cartesian : callable, optional
        Maps the tuple of meshed curvilinear coordinates to a tuple of
        meshed Cartesian coordinates.  Used for :attr:`cartesian_coords`
        and plotting.

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

    #: Constructor keyword parameters (besides the axes), used for
    #: serialization and equality.  Set by subclasses.
    _params: dict = {}

    #: Default axis names, applied to unnamed axes.  Set by subclasses.
    _default_axis_names: tuple[str, ...] = ()

    #: Exponent m of the radial volume element r^m for grids whose radial
    #: axis passes through the origin (1: disk/cylinder, 2: ball).
    _radial_power: int = 1

    def __init__(
        self,
        *axes: Axis,
        scale_factors: Sequence[ScaleFactorFn],
        cartesian: CartesianFn | None = None,
    ) -> None:
        if self._default_axis_names and len(axes) == len(self._default_axis_names):
            axes = tuple(
                ax.renamed(name) if isinstance(ax, Axis) and ax.name is None else ax
                for ax, name in zip(axes, self._default_axis_names)
            )
        super().__init__(*axes)
        if len(scale_factors) != self.ndims:
            raise ValueError(
                f"Expected {self.ndims} scale factors, got {len(scale_factors)}."
            )
        self._scale_factor_fns = tuple(scale_factors)
        self._cartesian_fn = cartesian
        self._diff_ops: dict | None = None
        self._h_arrays: tuple[NDArray, ...] | None = None
        self._jacobian: NDArray | None = None
        self._lap_coeffs: list | None = None
        self._odd_diff_ops: dict = {}
        self._parities: dict = {}
        # Fail early if an axis through the origin or the poles cannot be supported:
        for i, ax in enumerate(self.axes):
            if isinstance(ax, _FOLDED_AXES):
                self._antipodal_map(i)

    def _antipodal_map(self, axis_index: int):
        if self._antipodal_fn is None:
            return super()._antipodal_map(axis_index)
        return self._antipodal_fn(self, axis_index)

    #: Function ``(grid, axis_index) -> (f -> f at antipodal points)``; set by
    #: subclasses that support a RadialAxis through the origin.
    _antipodal_fn = None

    def _new_like(self, axes) -> CurvilinearGrid:
        new = copy.copy(self)
        CurvilinearGrid.__init__(
            new, *axes, scale_factors=self._scale_factor_fns, cartesian=self._cartesian_fn
        )
        return new

    def _eq_key(self) -> tuple:
        key = (type(self), self.axes, tuple(sorted(self._params.items())))
        if type(self) is CurvilinearGrid:
            key += (self._scale_factor_fns, self._cartesian_fn)
        return key

    def __hash__(self) -> int:
        return super().__hash__()

    def __repr__(self) -> str:
        r = super().__repr__()
        if self._params:
            params = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
            r = r[:-1] + f", {params})"
        return r

    # ── lazy initialisation ───────────────────────────────────────

    def _evaluate_scale_factors(self) -> tuple[NDArray, ...]:
        """Evaluate and cache scale-factor arrays on the mesh."""
        if self._h_arrays is None:
            coords = self.meshed_coords
            self._h_arrays = tuple(
                np.broadcast_to(np.asarray(h(coords), dtype=float), self.shape)
                for h in self._scale_factor_fns
            )
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

    # ── geometry ─────────────────────────────────────────────────

    @property
    def scale_factors(self) -> tuple[NDArray, ...]:
        """The scale factors :math:`h_i` evaluated on the grid."""
        return self._evaluate_scale_factors()

    @property
    def volume_element(self) -> NDArray:
        r"""The volume element :math:`J = h_1 h_2 \cdots h_n` on the grid."""
        return self._evaluate_jacobian()

    @property
    def cartesian_coords(self) -> tuple[NDArray, ...]:
        """Meshed Cartesian coordinates of the grid points.

        Raises
        ------
        NotImplementedError
            If the grid was created without a ``cartesian`` mapping.
        """
        if self._cartesian_fn is None:
            raise NotImplementedError(
                "This grid has no Cartesian mapping. Pass `cartesian=` to CurvilinearGrid."
            )
        return tuple(self._cartesian_fn(self.meshed_coords))

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
        f = self._check_field(f)
        self._ensure_diff_ops()
        h = self._evaluate_scale_factors()
        components = []
        for i in range(self.ndims):
            with np.errstate(divide="ignore", invalid="ignore"):
                comp = (1.0 / h[i]) * self._diff_ops[i](f)
            components.append(_finite_or_zero(comp))
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
        v = [self._check_field(vi, f"v[{i}]") for i, vi in enumerate(v)]
        h = self._evaluate_scale_factors()
        J = self._evaluate_jacobian()
        result = np.zeros(self.shape, dtype=np.result_type(*v, float))
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(self.ndims):
                coeff = J / h[i]
                result = result + self._component_diff(i, i)(coeff * v[i])
            result = result / J
        return _finite_or_zero(result)

    # ── derivatives of vector components across the origin and the poles ──

    def _reflected_coords(self, axis_index: int) -> tuple[NDArray, ...]:
        """Meshed coordinates continued across the origin (r -> -r) or a pole."""
        coords = list(self.meshed_coords)
        if isinstance(self.axes[axis_index], RadialAxis):
            coords[axis_index] = -coords[axis_index]
        else:
            coords[axis_index] = 2 * np.pi - coords[axis_index]
        return tuple(coords)

    def _covariant_parity(self, axis_index: int, component: int) -> int:
        r"""Parity of the covariant component :math:`h_k v_k` across a folded axis.

        A radial axis through the origin and a colatitude axis through the
        poles compute derivatives from the values at the antipodal points,
        assuming that the differentiated quantity is the same function on
        both sides (true for scalar fields).  Components of vector fields
        refer to the local basis, which may flip when the coordinate line
        is continued through the singular point:
        :math:`h_k v_k = \mathbf{v}\cdot\partial\mathbf{x}/\partial q_k`
        is even or odd, depending on whether the tangent vector
        :math:`\partial\mathbf{x}/\partial q_k` at the continued
        coordinates equals plus or minus the one at the antipodal point.
        The tangent vectors are computed from the Cartesian mapping by
        complex-step differentiation.
        """
        key = (axis_index, component)
        if key not in self._parities:
            if self._cartesian_fn is None:
                raise NotImplementedError(
                    "Vector calculus across the origin or the poles needs the Cartesian mapping."
                )
            antipodal = self._antipodal_map(axis_index)

            def tangent(coords):
                coords = list(coords)
                step = 1e-30
                coords[component] = coords[component] + 1j * step
                return [np.imag(np.broadcast_to(np.asarray(x, dtype=complex), self.shape)) / step
                        for x in self._cartesian_fn(tuple(coords))]

            continued = tangent(self._reflected_coords(axis_index))
            at_antipode = [antipodal(x) for x in tangent(self.meshed_coords)]
            overlap = sum(float(np.sum(a * b)) for a, b in zip(continued, at_antipode))
            self._parities[key] = 1 if overlap >= 0 else -1
        return self._parities[key]

    def _scalar_parity(self, axis_index: int, coeff: Callable) -> int:
        """Parity across a folded axis of ``coeff(h)``, a function of the scale factors."""
        antipodal = self._antipodal_map(axis_index)
        coords = self._reflected_coords(axis_index)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_ref = [np.broadcast_to(np.asarray(fn(coords), dtype=float), self.shape)
                     for fn in self._scale_factor_fns]
            continued = _finite_or_zero(coeff(h_ref))
            at_antipode = antipodal(_finite_or_zero(coeff(self._evaluate_scale_factors())))
        return 1 if float(np.sum(continued * at_antipode)) >= 0 else -1

    def _diff_parity(self, axis_index: int, component: int) -> int:
        r"""Parity of the quantity differentiated along *axis_index* in divergence and curl.

        Divergence: :math:`(J/h_i)\, v_i = (J/h_i^2)\,(h_i v_i)` along axis *i*;
        curl: :math:`h_k v_k` along axis :math:`j \ne k`.
        """
        if not isinstance(self.axes[axis_index], _FOLDED_AXES):
            return 1
        key = ("diff", axis_index, component)
        if key not in self._parities:
            parity = self._covariant_parity(axis_index, component)
            if component == axis_index:
                def coeff(h):
                    J = np.ones(self.shape)
                    for hk in h:
                        J = J * hk
                    return J / h[axis_index] ** 2
                parity *= self._scalar_parity(axis_index, coeff)
            self._parities[key] = parity
        return self._parities[key]

    def _component_diff(self, axis_index: int, component: int):
        """First derivative along an axis, for the term of divergence/curl with *component*."""
        self._ensure_diff_ops()
        if self._diff_parity(axis_index, component) == 1:
            return self._diff_ops[axis_index]
        if axis_index not in self._odd_diff_ops:
            from numgrids.diff import ColatitudeDiff, RadialDiff
            cls = RadialDiff if isinstance(self.axes[axis_index], RadialAxis) else ColatitudeDiff
            self._odd_diff_ops[axis_index] = cls(self, 1, axis_index, parity=-1)
        return self._odd_diff_ops[axis_index]

    def gradient_operator(self) -> tuple[Operator, ...]:
        r"""The components of the gradient as operators, :math:`G_i = h_i^{-1}\,\partial_i`."""
        key = "gradient"
        if key not in self.cache["operators"]:
            from numgrids.operators import Operator
            self._ensure_diff_ops()
            h = self._evaluate_scale_factors()
            ops = []
            with np.errstate(divide="ignore", invalid="ignore"):
                for i in range(self.ndims):
                    inv_h = _finite_or_zero(1.0 / h[i]).ravel()
                    ops.append(Operator(self, scipy.sparse.diags(inv_h) @ self._diff_ops[i].as_matrix()))
            self.cache["operators"][key] = tuple(ops)
        return self.cache["operators"][key]

    def divergence_operator(self) -> tuple[Operator, ...]:
        r"""The divergence as one operator per vector component.

        :math:`\nabla\cdot\mathbf{v} = \sum_i D_i v_i` with
        :math:`D_i = J^{-1}\,\partial_i\,(J/h_i)`.
        """
        key = "divergence"
        if key not in self.cache["operators"]:
            from numgrids.operators import Operator
            h = self._evaluate_scale_factors()
            J = self._evaluate_jacobian()
            ops = []
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_J = scipy.sparse.diags(_finite_or_zero(1.0 / J).ravel())
                for i in range(self.ndims):
                    coeff = scipy.sparse.diags(_finite_or_zero(J / h[i]).ravel())
                    D = self._component_diff(i, i).as_matrix()
                    ops.append(Operator(self, inv_J @ D @ coeff))
            self.cache["operators"][key] = tuple(ops)
        return self.cache["operators"][key]

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
        f = self._check_field(f)
        self._ensure_diff_ops()
        J = self._evaluate_jacobian()
        result = np.zeros(self.shape, dtype=np.result_type(f, float))
        with np.errstate(divide="ignore", invalid="ignore"):
            for i, (c, dc) in enumerate(self._laplacian_coefficients()):
                # Product rule d(c df) = c d^2f + dc df: uses the true second
                # derivative, so no spurious modes (e.g. the Nyquist mode on
                # periodic axes) end up in the null space.
                result = result + c * self.diff(f, 2, i) + dc * self.diff(f, 1, i)
            result = result / J
        return _finite_or_zero(result)

    def _laplacian_coefficients(self) -> list[tuple[NDArray, NDArray]]:
        """Coefficients ``c_i = J / h_i^2`` and their derivatives along axis *i*."""
        if getattr(self, "_lap_coeffs", None) is None:
            h = self._evaluate_scale_factors()
            J = self._evaluate_jacobian()
            coeffs = []
            with np.errstate(divide="ignore", invalid="ignore"):
                for i in range(self.ndims):
                    c = _finite_or_zero(J / h[i] ** 2)
                    if isinstance(self.axes[i], _FOLDED_AXES + (UnboundedAxis,)):
                        # The coefficient (e.g. c = r or sin(theta)) is not an even field
                        # across the origin/poles, so the folded derivative does not apply.
                        # Nor can an unbounded axis differentiate a coefficient that grows
                        # towards infinity (e.g. c = r^2).
                        dc = self._coefficient_derivative(i)
                    else:
                        dc = self.diff(c, 1, i)
                    coeffs.append((c, dc))
            self._lap_coeffs = coeffs
        return self._lap_coeffs

    def _coefficient_derivative(self, i: int) -> NDArray:
        """Derivative of ``J / h_i^2`` along axis *i* from the scale-factor functions.

        Uses complex-step differentiation of the user-supplied callables,
        which is exact to round-off for analytic expressions.
        """
        coords = list(self.meshed_coords)
        high = self.axes[i].high
        step = 1e-30 * (max(1.0, abs(high)) if np.isfinite(high) else 1.0)
        coords[i] = coords[i] + 1j * step
        coords = tuple(coords)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = [np.broadcast_to(np.asarray(fn(coords), dtype=complex), self.shape)
                 for fn in self._scale_factor_fns]
            J = np.ones(self.shape, dtype=complex)
            for hk in h:
                J = J * hk
            c = J / h[i] ** 2
            dc = np.imag(c) / step
        return _finite_or_zero(dc)

    def _quadrature_weights(self, axis_index: int) -> NDArray:
        axis = self.axes[axis_index]
        if isinstance(axis, RadialAxis):
            m = self._radial_power
            return axis.radial_quadrature_weights(m) / axis.coords ** m
        return axis.quadrature_weights

    def laplacian_operator(self, acc: int = 4) -> Operator:
        r"""The Laplacian as a sparse :class:`~numgrids.operators.Operator`.

        .. math::
            L = \operatorname{diag}(1/J) \sum_i \left[
                \operatorname{diag}(c_i)\, D_i^{(2)}
                + \operatorname{diag}(\partial_i c_i)\, D_i^{(1)} \right],
            \qquad c_i = J/h_i^2

        Rows at coordinate singularities (where the coefficients are not
        finite) are set to zero, consistent with :meth:`laplacian`.
        """
        key = ("laplacian", acc)
        if key not in self.cache["operators"]:
            from numgrids.api import Diff
            from numgrids.operators import Operator
            J = self._evaluate_jacobian()
            total = None
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_J = _finite_or_zero(1.0 / J).ravel()
                for i, (c, dc) in enumerate(self._laplacian_coefficients()):
                    D1 = Diff(self, 1, i, acc=acc).as_matrix()
                    D2 = Diff(self, 2, i, acc=acc).as_matrix()
                    term = scipy.sparse.diags(c.ravel()) @ D2 + scipy.sparse.diags(dc.ravel()) @ D1
                    total = term if total is None else total + term
            M = scipy.sparse.diags(inv_J) @ total
            self.cache["operators"][key] = Operator(self, M)
        return self.cache["operators"][key]

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
        if self.ndims not in (2, 3):
            raise ValueError("Curl is only defined for 2D and 3D grids.")
        if len(v) != self.ndims:
            raise ValueError(f"Expected {self.ndims} vector components, got {len(v)}.")
        v = [self._check_field(vi, f"v[{i}]") for i, vi in enumerate(v)]
        if self.ndims == 2:
            return self._curl_2d(*v)
        return self._curl_3d(*v)

    def _curl_2d(self, v0: NDArray, v1: NDArray) -> NDArray:
        """Scalar curl for 2D curvilinear grids."""
        h = self._evaluate_scale_factors()
        d0 = self._component_diff(0, 1)
        d1 = self._component_diff(1, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (1.0 / (h[0] * h[1])) * (
                d0(h[1] * v1) - d1(h[0] * v0)
            )
        return _finite_or_zero(result)

    def _curl_3d(self, v0: NDArray, v1: NDArray, v2: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Vector curl for 3D curvilinear grids."""
        h = self._evaluate_scale_factors()
        v = (v0, v1, v2)

        components: list[NDArray] = []
        cyclic = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        for _i, j, k in cyclic:
            with np.errstate(divide="ignore", invalid="ignore"):
                comp = (1.0 / (h[j] * h[k])) * (
                    self._component_diff(j, k)(h[k] * v[k]) - self._component_diff(k, j)(h[j] * v[j])
                )
            components.append(_finite_or_zero(comp))
        return tuple(components)


# ═══════════════════════════════════════════════════════════════════
# Built-in coordinate systems
# ═══════════════════════════════════════════════════════════════════

def _ones(c):
    return np.ones_like(c[0])


def _check_radial_index(grid, axis_index: int) -> None:
    if axis_index != 0 or not isinstance(grid.axes[axis_index], RadialAxis):
        raise ValueError(
            f"Only the first (radial) axis of a {type(grid).__name__} can be a RadialAxis, "
            "and no other axis can be folded across a singularity."
        )


def _half_turn_shift(grid, phi_index: int) -> int:
    """Number of points corresponding to a shift by pi on the azimuthal axis."""
    axis = grid.axes[phi_index]
    if not axis.periodic or len(axis) % 2 or not np.isclose(axis.period, 2 * np.pi):
        raise ValueError(
            "A RadialAxis or ColatitudeAxis requires a periodic azimuthal axis over 2*pi "
            f"with an even number of points; got {axis!r}."
        )
    return len(axis) // 2


def _polar_antipodal(grid, axis_index: int):
    """Antipodal map (r, phi) -> (r, phi + pi) for polar and cylindrical grids."""
    _check_radial_index(grid, axis_index)
    shift = _half_turn_shift(grid, 1)
    return lambda f: np.roll(f, shift, axis=1)


def _spherical_antipodal(grid, axis_index: int):
    """Antipodal maps of spherical grids.

    Radial axis: (-r, theta, phi) is (r, pi - theta, phi + pi).
    Colatitude axis: (2 pi - theta, phi) is (theta, phi + pi); the reversal of
    theta is part of the folded differentiation matrix, so only phi is shifted.
    """
    if axis_index == 1 and isinstance(grid.axes[1], ColatitudeAxis):
        shift = _half_turn_shift(grid, 2)
        return lambda f: np.roll(f, shift, axis=2)
    _check_radial_index(grid, axis_index)
    shift = _half_turn_shift(grid, 2)
    theta = grid.axes[1].coords
    if not np.allclose(theta + theta[::-1], np.pi, atol=1e-12):
        raise ValueError(
            "A RadialAxis through the origin of a SphericalGrid requires a polar axis "
            "that is symmetric about pi/2 (e.g. colatitude(n) or chebyshev(n, eps, pi - eps))."
        )
    return lambda f: np.roll(np.flip(f, axis=1), shift, axis=2)


class PolarGrid(CurvilinearGrid):
    r"""A 2D grid in polar coordinates :math:`(r, \varphi)`.

    .. math::
        x = r\cos\varphi, \quad y = r\sin\varphi; \qquad
        h_r = 1,\ h_\varphi = r

    Provides built-in vector calculus operators:

    - :meth:`laplacian` — scalar Laplacian
    - :meth:`gradient` — gradient of a scalar field
    - :meth:`divergence` — divergence of a 2D vector field
    - :meth:`curl` — curl (returns the scalar *z*-component)

    Coordinate singularities at *r = 0* are handled gracefully.
    Unnamed axes are named ``"r"`` and ``"phi"``.

    Examples
    --------
    >>> from numgrids import *
    >>> import numpy as np
    >>> grid = PolarGrid(
    ...     create_axis(AxisType.CHEBYSHEV, 30, 0.1, 1),
    ...     create_axis(AxisType.EQUIDISTANT_PERIODIC, 40, 0, 2 * np.pi),
    ... )
    >>> R, Phi = grid.meshed_coords
    >>> f = R ** 2 * np.cos(Phi)
    >>> lap_f = grid.laplacian(f)
    """

    _default_axis_names = ("r", "phi")

    def __init__(self, raxis: Axis, phi_axis: Axis) -> None:
        """Constructor

        Parameters
        ----------
        raxis: Axis
            The radial axis.
        phi_axis: Axis
            The azimuthal axis (phi). Should typically be periodic.
        """
        self._antipodal_fn = _polar_antipodal
        super().__init__(
            raxis, phi_axis,
            scale_factors=(
                _ones,                          # h_r = 1
                lambda c: c[0],                 # h_phi = r
            ),
            cartesian=lambda c: (c[0] * np.cos(c[1]), c[0] * np.sin(c[1])),
        )


class CylindricalGrid(CurvilinearGrid):
    r"""A grid in cylindrical coordinates :math:`(r, \varphi, z)`.

    .. math::
        x = r\cos\varphi, \quad y = r\sin\varphi, \quad z = z; \qquad
        h_r = 1,\ h_\varphi = r,\ h_z = 1

    Provides built-in vector calculus operators:

    - :meth:`laplacian` — scalar Laplacian
    - :meth:`gradient` — gradient of a scalar field
    - :meth:`divergence` — divergence of a vector field
    - :meth:`curl` — curl of a vector field

    Coordinate singularities at *r = 0* are handled gracefully.
    Unnamed axes are named ``"r"``, ``"phi"`` and ``"z"``.

    Examples
    --------
    >>> from numgrids import *
    >>> import numpy as np
    >>> grid = CylindricalGrid(
    ...     create_axis(AxisType.CHEBYSHEV, 20, 0.1, 2),
    ...     create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2 * np.pi),
    ...     create_axis(AxisType.CHEBYSHEV, 20, -1, 1),
    ... )
    >>> R, Phi, Z = grid.meshed_coords
    >>> f = R ** 2 + Z ** 2
    >>> lap_f = grid.laplacian(f)  # should be ~6
    """

    _default_axis_names = ("r", "phi", "z")

    def __init__(self, raxis: Axis, phi_axis: Axis, zaxis: Axis) -> None:
        """Constructor

        Parameters
        ----------
        raxis: Axis
            The radial axis.
        phi_axis: Axis
            The azimuthal axis (phi). Should typically be periodic.
        zaxis: Axis
            The axial (z) axis.
        """
        self._antipodal_fn = _polar_antipodal
        super().__init__(
            raxis, phi_axis, zaxis,
            scale_factors=(
                _ones,                          # h_r = 1
                lambda c: c[0],                 # h_phi = r
                _ones,                          # h_z = 1
            ),
            cartesian=lambda c: (c[0] * np.cos(c[1]), c[0] * np.sin(c[1]), c[2]),
        )


class SphericalGrid(CurvilinearGrid):
    r"""A grid in spherical coordinates :math:`(r, \theta, \varphi)`.

    .. math::
        x = r\sin\theta\cos\varphi, \quad
        y = r\sin\theta\sin\varphi, \quad
        z = r\cos\theta; \qquad
        h_r = 1,\ h_\theta = r,\ h_\varphi = r\sin\theta

    Provides built-in vector calculus operators:

    - :meth:`laplacian` — scalar Laplacian
    - :meth:`gradient` — gradient of a scalar field
    - :meth:`divergence` — divergence of a vector field
    - :meth:`curl` — curl of a vector field

    Coordinate singularities at *r = 0* and *theta = 0, pi* are handled
    gracefully: non-finite values are replaced by zero.
    Unnamed axes are named ``"r"``, ``"theta"`` and ``"phi"``.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import SphericalGrid, chebyshev, periodic
    >>> grid = SphericalGrid(chebyshev(20, 0, 1), chebyshev(20, 0, np.pi), periodic(20))
    >>> round(grid.integrate(np.ones(grid.shape)), 8) == round(4 / 3 * np.pi, 8)
    True
    """

    _default_axis_names = ("r", "theta", "phi")
    _radial_power = 2

    def __init__(self, raxis: Axis, theta_axis: Axis, phi_axis: Axis) -> None:
        """Constructor

        Parameters
        ----------
        raxis: Axis
            The radial axis.
        theta_axis: Axis
            The polar axis (theta).
        phi_axis: Axis
            The azimuthal axis (phi). Must be periodic.
        """
        self._antipodal_fn = _spherical_antipodal
        super().__init__(
            raxis, theta_axis, phi_axis,
            scale_factors=(
                _ones,                                  # h_r = 1
                lambda c: c[0],                         # h_theta = r
                lambda c: c[0] * np.sin(c[1]),          # h_phi = r sin(theta)
            ),
            cartesian=lambda c: (
                c[0] * np.sin(c[1]) * np.cos(c[2]),
                c[0] * np.sin(c[1]) * np.sin(c[2]),
                c[0] * np.cos(c[1]),
            ),
        )


class EllipticCylindricalGrid(CurvilinearGrid):
    r"""A grid in elliptic cylindrical coordinates :math:`(\mu, \nu, z)`.

    .. math::
        x = a\cosh\mu\cos\nu, \quad y = a\sinh\mu\sin\nu, \quad z = z; \qquad
        h_\mu = h_\nu = a\sqrt{\sinh^2\mu + \sin^2\nu},\ h_z = 1

    Typically :math:`\mu \ge 0` and :math:`\nu \in [0, 2\pi)` (periodic).
    The foci are located at :math:`x = \pm a`.  Unnamed axes are named
    ``"mu"``, ``"nu"`` and ``"z"``.

    Parameters
    ----------
    mu_axis, nu_axis, zaxis : Axis
        The coordinate axes.
    a : float
        Focal distance (default 1).
    """

    _default_axis_names = ("mu", "nu", "z")

    def __init__(self, mu_axis: Axis, nu_axis: Axis, zaxis: Axis, a: float = 1.0) -> None:
        self.a = a
        self._params = {"a": a}

        def h(c):
            return a * np.sqrt(np.sinh(c[0]) ** 2 + np.sin(c[1]) ** 2)

        super().__init__(
            mu_axis, nu_axis, zaxis,
            scale_factors=(h, h, _ones),
            cartesian=lambda c: (
                a * np.cosh(c[0]) * np.cos(c[1]),
                a * np.sinh(c[0]) * np.sin(c[1]),
                c[2],
            ),
        )


class ProlateSpheroidalGrid(CurvilinearGrid):
    r"""A grid in prolate spheroidal coordinates :math:`(\mu, \nu, \varphi)`.

    .. math::
        x = a\sinh\mu\sin\nu\cos\varphi, \quad
        y = a\sinh\mu\sin\nu\sin\varphi, \quad
        z = a\cosh\mu\cos\nu

    .. math::
        h_\mu = h_\nu = a\sqrt{\sinh^2\mu + \sin^2\nu}, \quad
        h_\varphi = a\sinh\mu\sin\nu

    Typically :math:`\mu \ge 0`, :math:`\nu \in [0, \pi]` and
    :math:`\varphi \in [0, 2\pi)` (periodic).  The foci are located on the
    *z*-axis at :math:`z = \pm a`.  Unnamed axes are named ``"mu"``,
    ``"nu"`` and ``"phi"``.

    Parameters
    ----------
    mu_axis, nu_axis, phi_axis : Axis
        The coordinate axes.
    a : float
        Focal distance (default 1).
    """

    _default_axis_names = ("mu", "nu", "phi")

    def __init__(self, mu_axis: Axis, nu_axis: Axis, phi_axis: Axis, a: float = 1.0) -> None:
        self.a = a
        self._params = {"a": a}

        def h(c):
            return a * np.sqrt(np.sinh(c[0]) ** 2 + np.sin(c[1]) ** 2)

        super().__init__(
            mu_axis, nu_axis, phi_axis,
            scale_factors=(h, h, lambda c: a * np.sinh(c[0]) * np.sin(c[1])),
            cartesian=lambda c: (
                a * np.sinh(c[0]) * np.sin(c[1]) * np.cos(c[2]),
                a * np.sinh(c[0]) * np.sin(c[1]) * np.sin(c[2]),
                a * np.cosh(c[0]) * np.cos(c[1]),
            ),
        )


class OblateSpheroidalGrid(CurvilinearGrid):
    r"""A grid in oblate spheroidal coordinates :math:`(\mu, \nu, \varphi)`.

    .. math::
        x = a\cosh\mu\cos\nu\cos\varphi, \quad
        y = a\cosh\mu\cos\nu\sin\varphi, \quad
        z = a\sinh\mu\sin\nu

    .. math::
        h_\mu = h_\nu = a\sqrt{\sinh^2\mu + \sin^2\nu}, \quad
        h_\varphi = a\cosh\mu\cos\nu

    Typically :math:`\mu \ge 0`, :math:`\nu \in [-\pi/2, \pi/2]` and
    :math:`\varphi \in [0, 2\pi)` (periodic).  The focal ring has radius
    *a* in the *xy*-plane.  Unnamed axes are named ``"mu"``, ``"nu"`` and
    ``"phi"``.

    Parameters
    ----------
    mu_axis, nu_axis, phi_axis : Axis
        The coordinate axes.
    a : float
        Radius of the focal ring (default 1).
    """

    _default_axis_names = ("mu", "nu", "phi")

    def __init__(self, mu_axis: Axis, nu_axis: Axis, phi_axis: Axis, a: float = 1.0) -> None:
        self.a = a
        self._params = {"a": a}

        def h(c):
            return a * np.sqrt(np.sinh(c[0]) ** 2 + np.sin(c[1]) ** 2)

        super().__init__(
            mu_axis, nu_axis, phi_axis,
            scale_factors=(h, h, lambda c: a * np.cosh(c[0]) * np.cos(c[1])),
            cartesian=lambda c: (
                a * np.cosh(c[0]) * np.cos(c[1]) * np.cos(c[2]),
                a * np.cosh(c[0]) * np.cos(c[1]) * np.sin(c[2]),
                a * np.sinh(c[0]) * np.sin(c[1]),
            ),
        )


def _surface_antipodal(grid, axis_index: int):
    """Antipodal map (2 pi - theta, phi) -> (theta, phi + pi) on the sphere surface."""
    if axis_index != 0 or not isinstance(grid.axes[0], ColatitudeAxis):
        raise ValueError(
            "Only the first (polar angle) axis of a SphericalSurfaceGrid can be a ColatitudeAxis."
        )
    shift = _half_turn_shift(grid, 1)
    return lambda f: np.roll(f, shift, axis=1)


class SphericalSurfaceGrid(CurvilinearGrid):
    r"""A 2D grid on the surface of a sphere of radius *R*, coordinates :math:`(\theta, \varphi)`.

    .. math::
        x = R\sin\theta\cos\varphi, \quad
        y = R\sin\theta\sin\varphi, \quad
        z = R\cos\theta; \qquad
        h_\theta = R,\ h_\varphi = R\sin\theta

    The Laplacian is the Laplace--Beltrami operator of the sphere, whose
    eigenfunctions are the spherical harmonics. With a
    :func:`~numgrids.colatitude` axis the whole sphere including both poles is
    covered with spectral accuracy:

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import SphericalSurfaceGrid, colatitude, periodic
    >>> sphere = SphericalSurfaceGrid(colatitude(16), periodic(32))
    >>> vals, _ = (-sphere.laplacian_operator()).eig(k=9)
    >>> np.round(vals, 8) + 0.0     # l (l + 1) with multiplicity 2 l + 1
    array([0., 2., 2., 2., 6., 6., 6., 6., 6.])
    >>> round(sphere.integrate(np.ones(sphere.shape)), 12) == round(4 * np.pi, 12)
    True

    Unnamed axes are named ``"theta"`` and ``"phi"``.

    Parameters
    ----------
    theta_axis, phi_axis : Axis
        Polar angle (e.g. :func:`~numgrids.colatitude`) and azimuth
        (periodic over :math:`2\pi`).
    radius : float
        Radius *R* of the sphere (default 1).
    """

    _default_axis_names = ("theta", "phi")

    def __init__(self, theta_axis: Axis, phi_axis: Axis, radius: float = 1.0) -> None:
        self.radius = radius
        self._params = {"radius": radius}
        self._antipodal_fn = _surface_antipodal
        R = radius
        super().__init__(
            theta_axis, phi_axis,
            scale_factors=(lambda c: R * np.ones_like(c[0]), lambda c: R * np.sin(c[0])),
            cartesian=lambda c: (
                R * np.sin(c[0]) * np.cos(c[1]),
                R * np.sin(c[0]) * np.sin(c[1]),
                R * np.cos(c[0]),
            ),
        )

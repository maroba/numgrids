r"""Functions of operators: fractional powers, square roots, exponentials, inverses.

For an operator :math:`L` with eigenpairs :math:`L \varphi_n = \lambda_n
\varphi_n` and a scalar function :math:`f`, the operator :math:`f(L)` has
the same eigenfunctions and the eigenvalues :math:`f(\lambda_n)`:

.. math::

    f(L)\, u = \sum_n f(\lambda_n)\, c_n\, \varphi_n
    \qquad \text{for} \qquad u = \sum_n c_n\, \varphi_n .

With :math:`f(\lambda) = \lambda^s` this defines the fractional Laplacian
:math:`(-\Delta)^s`, with :math:`f(\lambda) = e^{t \lambda}` the semigroup
:math:`e^{tL}` that solves :math:`u_t = L u`.  The eigenfunctions depend
on the boundary conditions, and so does :math:`f(L)`.

The functions are available as methods of
:class:`~numgrids.operators.Operator`:
:meth:`~numgrids.operators.Operator.function`,
:meth:`~numgrids.operators.Operator.power`,
:meth:`~numgrids.operators.Operator.sqrt`,
:meth:`~numgrids.operators.Operator.exp` and
:meth:`~numgrids.operators.Operator.inverse`.  This module holds the
numerical core.

**Boundary conditions.**  The conditions are homogeneous, as in
:meth:`~numgrids.operators.Operator.eig`.  They express the values on the
boundary points *b* through the values on the other points *i*:
:math:`u_b = S u_i` (:math:`S = 0` for Dirichlet conditions).  On the
functions that satisfy the conditions, the operator acts as the matrix
:math:`L_c = L_{ii} + L_{ib} S` on the values :math:`u_i`, and

.. math::

    f(L) = \begin{pmatrix} f(L_c) & 0 \\ S f(L_c) & 0 \end{pmatrix}

in the ordering :math:`(i, b)`.  The eigenvalues of :math:`L_c` are those
that :meth:`~numgrids.operators.Operator.eig` returns.

**Numerical methods.**  Functions of matrices need dense linear algebra.
Three paths keep the matrices small where the problem allows it:

* **Fourier symbol** -- an operator that is shift invariant along all axes
  (constant coefficients on a periodic grid) is diagonal in Fourier space;
  *f* is applied to its symbol.
* **Fourier blocks** -- if the operator and the boundary conditions are
  shift invariant along some periodic axes (the angles of disks, cylinders,
  spheres and balls), there is one small matrix per Fourier mode.
* **Dense** -- otherwise, one matrix for the whole problem.

In the first two cases the result is a :class:`FourierOperator`, which is
applied with fast Fourier transforms.

The function of each matrix comes from its eigendecomposition,
:math:`f(A) = V f(\Lambda) V^{-1}` (with orthonormal eigenvectors for
Hermitian matrices).  This is accurate as long as the eigenvectors are well
conditioned, which holds for Laplacians, diffusion and Schrödinger
operators also on spectral grids.  For operators that are far from normal
(e.g. dominated by convection), where the condition number of *V* exceeds
:math:`10^5`, Schur-based algorithms take over:
:func:`scipy.linalg.sqrtm`, :func:`scipy.linalg.fractional_matrix_power`,
:func:`scipy.linalg.expm` and :func:`scipy.linalg.inv` for the named
functions, and the Schur-Parlett method :func:`scipy.linalg.funm` for a
general *f*.
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.sparse
from numpy.typing import NDArray

from numgrids.linalg import FourierBlocks, shift_invariant_axes
from numgrids.operators import Operator, _count_zero_rows

if TYPE_CHECKING:
    from numgrids.grids import Grid

__all__ = ["FourierOperator"]

# Largest dense matrix (rows) a function is computed of by default; dense
# eigensolvers need about 25 n^3 operations.
_MAX_SIZE = 4000
# Bound on the memory (bytes) of the Fourier blocks of an operator; the
# computation holds a few arrays of this size.
_MAX_BLOCK_BYTES = 5e8
# Bound on the memory (bytes) of the matrix of a FourierOperator.
_MAX_MATRIX_BYTES = 1e9
# Up to this 1-norm condition number of the eigenvectors, the eigendecomposition
# is used: it then loses at most 5 of 16 digits.
_MAX_EIGENVECTOR_COND = 1e5
# Eigenvalues (and their imaginary parts) below this fraction of the largest
# one are round-off: they are set to zero.
_ROUNDOFF = 1e-12
# A matrix that deviates from its conjugate transpose by less than this fraction is Hermitian.
_HERMITIAN_TOL = 1e-13
# Estimated relative errors above this are reported.
_WARN_ERROR = 1e-8


class FourierOperator(Operator):
    r"""An operator that is block diagonal in Fourier space.

    The operator is shift invariant along one or more periodic axes of the
    grid.  It is stored as one small matrix per Fourier mode of these axes
    (a single number per mode, the *symbol*, if it is shift invariant along
    all axes) and applied with fast Fourier transforms.

    Functions of operators (:meth:`Operator.function
    <numgrids.operators.Operator.function>`, ``power``, ``sqrt``, ``exp``,
    ``inverse``) return this type where the problem has the structure.  It
    behaves like any other :class:`~numgrids.operators.Operator`.  Sums,
    products and functions of such operators and multiples of them keep
    the Fourier form.  Everything else (``matrix``, ``toarray``, ``solve``,
    ``eig``, arithmetic with other operators or with arrays) uses the
    matrix, which is dense and built on first use.

    Parameters
    ----------
    grid : Grid
        The grid on which the operator acts.
    blocks : FourierBlocks
        The Fourier decomposition along the shift-invariant axes.
    hat : NDArray
        The matrices of the Fourier modes, shape
        ``(blocks.n_modes, blocks.n_rest, blocks.n_rest)``.
    real : bool
        Whether the operator maps real functions to real functions.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids import Grid, periodic
    >>> grid = Grid(periodic(64), periodic(64))
    >>> half = (-grid.laplacian_operator()).power(0.5)       # (-Laplacian)^(1/2)
    >>> half
    FourierOperator(grid=Grid(64, 64), modes=4096, block size=1)
    >>> X, Y = grid.meshed_coords
    >>> u = np.sin(3 * X) * np.cos(4 * Y)
    >>> bool(np.allclose(half(u), 5 * u))
    True
    """

    def __init__(self, grid: Grid, blocks: FourierBlocks, hat: NDArray, real: bool = False) -> None:
        hat = np.asarray(hat, dtype=complex)
        if hat.shape != (blocks.n_modes, blocks.n_rest, blocks.n_rest):
            raise ValueError(
                f"Blocks of shape {hat.shape} do not match the Fourier decomposition "
                f"({blocks.n_modes} modes, block size {blocks.n_rest})."
            )
        self.grid = grid
        self._blocks = blocks
        self._hat = hat
        self._real = bool(real)
        self._matrix = None

    @property
    def matrix(self) -> scipy.sparse.csr_matrix:
        """The matrix of the operator (dense in general; built on first use)."""
        if self._matrix is None:
            self._matrix = self._materialize()
        return self._matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (self.grid.size, self.grid.size)

    def __repr__(self) -> str:
        return (
            f"FourierOperator(grid={type(self.grid).__name__}{self.grid.shape}, "
            f"modes={self._blocks.n_modes}, block size={self._blocks.n_rest})"
        )

    def __call__(self, f: NDArray) -> NDArray:
        """Apply the operator to a meshed function of shape ``grid.shape``."""
        f = np.asarray(f)
        if f.shape != self.grid.shape:
            raise ValueError(
                f"Array of shape {f.shape} does not match grid shape {self.grid.shape}."
            )
        coeffs = self._blocks.forward(f.ravel())
        if self._blocks.n_rest == 1:
            coeffs = self._hat[:, 0, 0] * coeffs
        else:
            coeffs = np.matmul(self._hat, coeffs.T[:, :, None])[:, :, 0].T
        u = self._blocks.backward(coeffs)
        if self._real and not np.iscomplexobj(f):
            u = u.real
        return u.reshape(self.grid.shape)

    def _materialize(self) -> scipy.sparse.csr_matrix:
        blocks = self._blocks
        size, n_rest, n_modes = blocks.size, blocks.n_rest, blocks.n_modes
        if 12.0 * size ** 2 > _MAX_MATRIX_BYTES:
            raise ValueError(
                f"The matrix of this operator is dense: with {size} grid points it would need "
                f"{12.0 * size ** 2 / 1e9:.1f} GB. Apply the operator by calling it (this uses fast "
                "Fourier transforms), and combine it only with operators of the same kind."
            )
        # Entries between the points (a, i) and (b, j): C[a, b, j - i].
        hat = np.moveaxis(self._hat, 0, -1).reshape((n_rest, n_rest) + blocks.mode_shape)
        C = np.fft.fftn(hat, axes=tuple(range(2, hat.ndim))) / n_modes
        C = C.reshape(n_rest, n_rest, n_modes)
        if self._real:
            C = C.real
        points = np.indices(blocks.mode_shape).reshape(len(blocks.mode_shape), n_modes)
        M = np.empty((size, size), dtype=C.dtype)
        cols = blocks.index.ravel()
        for i in range(n_modes):
            shift = np.ravel_multi_index([(p - p[i]) % n for p, n in zip(points, blocks.mode_shape)],
                                         blocks.mode_shape)
            M[blocks.index[:, i][:, None], cols[None, :]] = C[:, :, shift].reshape(n_rest, size)
        # Assemble the CSR matrix directly: converting the dense array would double the memory.
        dtype = np.int32 if size * size < 2 ** 31 else np.int64
        indices = np.tile(np.arange(size, dtype=dtype), size)
        indptr = np.arange(0, size * size + 1, size, dtype=dtype)
        return scipy.sparse.csr_matrix((M.ravel(), indices, indptr), shape=(size, size))

    # ── algebra that keeps the Fourier form ─────────────────────

    def _new(self, hat: NDArray, real: bool) -> FourierOperator:
        return FourierOperator(self.grid, self._blocks, hat, real)

    def _partner(self, other) -> FourierOperator | None:
        """*other* if it is block diagonal for the same Fourier modes, else ``None``."""
        if isinstance(other, FourierOperator) and other._blocks.axes == self._blocks.axes:
            self._check_same_grid(other.grid)
            return other
        return None

    def _scalar(self, c) -> tuple[NDArray, bool]:
        return c * np.eye(self._blocks.n_rest), self._real and not np.iscomplexobj(c)

    def __neg__(self) -> FourierOperator:
        return self._new(-self._hat, self._real)

    def __add__(self, other) -> Operator:
        if isinstance(other, numbers.Number):
            shift, real = self._scalar(other)
            return self._new(self._hat + shift, real)
        partner = self._partner(other)
        if partner is not None:
            return self._new(self._hat + partner._hat, self._real and partner._real)
        return super().__add__(other)

    __radd__ = __add__

    def __sub__(self, other) -> Operator:
        if isinstance(other, numbers.Number):
            shift, real = self._scalar(other)
            return self._new(self._hat - shift, real)
        partner = self._partner(other)
        if partner is not None:
            return self._new(self._hat - partner._hat, self._real and partner._real)
        return super().__sub__(other)

    def __rsub__(self, other) -> Operator:
        if isinstance(other, numbers.Number):
            shift, real = self._scalar(other)
            return self._new(shift - self._hat, real)
        return super().__rsub__(other)

    def __mul__(self, other) -> Operator:
        if isinstance(other, numbers.Number):
            return self._new(self._hat * other, self._real and not np.iscomplexobj(other))
        partner = self._partner(other)
        if partner is not None:
            return self._new(np.matmul(self._hat, partner._hat), self._real and partner._real)
        return super().__mul__(other)

    def __rmul__(self, other) -> Operator:
        if isinstance(other, numbers.Number):
            return self._new(other * self._hat, self._real and not np.iscomplexobj(other))
        return super().__rmul__(other)

    def __matmul__(self, other) -> Operator:
        partner = self._partner(other)
        if partner is not None:
            return self._new(np.matmul(self._hat, partner._hat), self._real and partner._real)
        return super().__matmul__(other)

    def __truediv__(self, other) -> Operator:
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return self._new(self._hat / other, self._real and not np.iscomplexobj(other))

    def __pow__(self, n: int) -> FourierOperator:
        if not isinstance(n, numbers.Integral) or n < 0:
            raise ValueError("Operators can only be raised to non-negative integer powers.")
        return self._new(np.linalg.matrix_power(self._hat, int(n)), self._real)


# ---------------------------------------------------------------------------
# The scalar functions
# ---------------------------------------------------------------------------

@dataclass
class _Function:
    """A scalar function, with what is needed to evaluate it on matrices."""

    #: values on an array of eigenvalues
    scalar: Callable[[NDArray], NDArray]
    #: how to call the function in messages
    name: str
    #: raises if the function is not defined on the given eigenvalues
    check: Callable[[NDArray], None] | None = None
    #: Schur-based evaluation on a dense matrix, for matrices far from normal
    matrix: Callable[[NDArray], NDArray] | None = None
    #: whether `matrix` needs a nonsingular matrix
    regular_only: bool = False


def _power_function(s: float) -> _Function:
    if isinstance(s, bool) or not isinstance(s, numbers.Real):
        raise TypeError(f"The exponent must be a real number, got {s!r}.")
    s = float(s)
    if not np.isfinite(s):
        raise ValueError(f"The exponent must be finite, got {s}.")
    integer = s == int(s)
    exponent = int(s) if integer else s

    def scalar(lam: NDArray) -> NDArray:
        # Zero eigenvalues stay zero, also for negative exponents (pseudo-inverse).
        out = np.zeros(lam.shape, dtype=lam.dtype if exponent >= 0 else np.result_type(lam.dtype, float))
        nonzero = lam != 0
        out[nonzero] = lam[nonzero] ** exponent
        if exponent == 0:
            out[:] = 1
        return out

    def check(lam: NDArray) -> None:
        negative = (lam.imag == 0) & (lam.real < 0)
        if not integer and np.any(negative):
            raise ValueError(
                f"The power {s:g} needs eigenvalues off the negative real axis, but the operator has "
                f"{int(np.sum(negative))} negative eigenvalue(s) (down to {lam.real[negative].min():.4g}). "
                "For a negative operator L such as the Laplacian, take the power of -L: "
                f"(-L).power({s:g}, bcs). (If the complex principal branch is meant, use "
                f"L.function(lambda z: np.emath.power(z, {s:g}), bcs).)"
            )

    if s == 0.5:
        matrix = scipy.linalg.sqrtm
    elif s == -1:
        matrix = scipy.linalg.inv
    else:
        def matrix(A: NDArray) -> NDArray:
            return scipy.linalg.fractional_matrix_power(A, s)

    return _Function(scalar, f"The power {s:g}", check, matrix, regular_only=not (integer and s >= 0))


def _exp_function(t: complex) -> _Function:
    if not isinstance(t, numbers.Number):
        raise TypeError(f"t must be a number, got {t!r}.")
    return _Function(lambda lam: np.exp(t * lam), f"exp(t L) with t = {t:g}", None,
                     lambda A: scipy.linalg.expm(t * A))


# ---------------------------------------------------------------------------
# Entry points (called by the methods of Operator)
# ---------------------------------------------------------------------------

def _function(operator: Operator, f: Callable, bcs=None, max_size: int | None = None) -> Operator:
    if not callable(f):
        raise TypeError("f must be a function of the eigenvalues, e.g. np.sqrt or lambda z: z ** 0.5.")
    return _evaluate(operator, _Function(f, "f"), bcs, max_size)


def _power(operator: Operator, s: float, bcs=None, max_size: int | None = None) -> Operator:
    return _evaluate(operator, _power_function(s), bcs, max_size)


def _exp(operator: Operator, t: complex = 1.0, bcs=None, max_size: int | None = None) -> Operator:
    return _evaluate(operator, _exp_function(t), bcs, max_size)


def _evaluate(operator: Operator, func: _Function, bcs, max_size: int | None) -> Operator:
    """The operator ``func(operator)`` on the functions that satisfy the boundary conditions."""
    grid = operator.grid
    max_size = _MAX_SIZE if max_size is None else int(max_size)
    R, boundary = _boundary_rows(grid, bcs)
    n_boundary = int(np.sum(boundary))

    blocks, hat, A = None, None, None
    if isinstance(operator, FourierOperator):
        # The Fourier form stays, if the boundary conditions are shift invariant as well.
        candidate = operator._blocks
        if not n_boundary:
            blocks, hat = candidate, operator._hat
        elif set(candidate.axes) <= set(shift_invariant_axes(R, grid)):
            rows = boundary[candidate.index[:, 0]]
            blocks, hat = candidate, operator._hat.copy()
            hat[:, rows, :] = candidate.blocks(R)[:, rows, :]
        real = operator._real and not np.iscomplexobj(R)
    if blocks is None:
        # The operator on the points without a condition, and the conditions on the others.
        A = scipy.sparse.csr_matrix(scipy.sparse.diags((~boundary).astype(float)) @ operator.matrix)
        n_zero = _count_zero_rows(A) - n_boundary
        A = scipy.sparse.csr_matrix(A + R)
        real = not np.iscomplexobj(A)
        if n_zero:
            warnings.warn(
                f"The operator has {n_zero} empty row(s) not covered by boundary conditions "
                "(e.g. at a coordinate singularity such as r = 0). They act as spurious zero "
                "eigenvalues; start the axis slightly away from the singularity or impose "
                "boundary conditions there.",
                RuntimeWarning,
                stacklevel=4,
            )
        axes = shift_invariant_axes(A, grid)
        if axes:
            blocks = FourierBlocks(grid, axes)

    if blocks is None:
        _check_size(grid.size - n_boundary, 1, max_size)
        L, S = _reduce(A.toarray(), boundary)
        (F,), real, error = _functions([L], func, real)
        result = Operator(grid, _extend(F.real if real else F, S, boundary))
    else:
        rows = boundary[blocks.index[:, 0]]
        _check_size(blocks.n_rest - int(np.sum(rows)), blocks.n_modes, max_size, blocks.nbytes())
        if hat is None:
            hat = blocks.blocks(A)
        reduced = [_reduce(block, rows) for block in hat]
        F, real, error = _functions([L for L, _ in reduced], func, real)
        hat = np.array([_extend(Fm, S, rows) for Fm, (_, S) in zip(F, reduced)], dtype=complex)
        result = FourierOperator(grid, blocks, hat.reshape(blocks.n_modes, blocks.n_rest, blocks.n_rest), real)

    if error > _WARN_ERROR:
        warnings.warn(
            f"{func.name} of this operator may be inaccurate (estimated relative error {error:.0e}): "
            "the operator is far from normal, so its eigenvectors are ill conditioned.",
            RuntimeWarning,
            stacklevel=4,
        )
    return result


def _check_size(n: int, n_blocks: int, max_size: int, nbytes: float = 0.0) -> None:
    """Raise if the dense matrices of the problem are too large."""
    count = "a dense matrix" if n_blocks == 1 else f"{n_blocks} dense matrices"
    hint = (
        "Problems only split into smaller ones along periodic axes on which neither the operator "
        "nor the boundary conditions depend (periodic boxes, the angles of disks, spheres and balls). "
        "Use a coarser grid"
    )
    if n > max_size:
        raise ValueError(
            f"Functions of operators need dense linear algebra: this problem leads to {count} of size "
            f"{n} x {n}, more than max_size = {max_size}. {hint}, or pass a larger max_size if time "
            "and memory permit (the work grows with the third power of the size)."
        )
    if nbytes > _MAX_BLOCK_BYTES:
        raise ValueError(
            f"Functions of operators need dense linear algebra: this problem leads to {count} of size "
            f"{n} x {n}, each set of which takes {nbytes / 1e9:.1f} GB. {hint}."
        )


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _boundary_rows(grid: Grid, bcs) -> tuple[scipy.sparse.csr_matrix, NDArray]:
    """The rows of the boundary conditions as a matrix, and the mask of the points they sit on."""
    from numgrids.boundary import resolve_bcs

    size = grid.size
    R = scipy.sparse.csr_matrix((size, size))
    boundary = np.zeros(size, dtype=bool)
    zeros = np.zeros(size)
    for bc in resolve_bcs(grid, bcs):  # later conditions win on shared points, as in apply_bcs
        R, _ = bc.apply_to_system(R, zeros)
        boundary[bc.face.flat_indices] = True
    return scipy.sparse.csr_matrix(R), boundary


def _reduce(block: NDArray, boundary: NDArray) -> tuple[NDArray, NDArray | None]:
    r"""The operator on the functions that satisfy the boundary conditions.

    The rows of *block* on the boundary points hold the conditions,
    :math:`R_{bb} u_b + R_{bi} u_i = 0`.  They give :math:`u_b = S u_i`,
    and the other rows become :math:`L_c = L_{ii} + L_{ib} S`.  Returns
    :math:`(L_c, S)`, with ``S = None`` if there are no conditions.
    """
    if not boundary.any():
        return block, None
    b, i = np.flatnonzero(boundary), np.flatnonzero(~boundary)
    singular = ValueError(
        "The boundary conditions do not determine the boundary values (singular boundary system)."
    )
    try:
        S = -np.linalg.solve(block[np.ix_(b, b)], block[np.ix_(b, i)])
    except np.linalg.LinAlgError as err:
        raise singular from err
    if not np.all(np.isfinite(S)):
        raise singular
    return block[np.ix_(i, i)] + block[np.ix_(i, b)] @ S, S


def _extend(F: NDArray, S: NDArray | None, boundary: NDArray) -> NDArray:
    """Extend a function of the reduced operator to all points (see the module docstring)."""
    if S is None:
        return F
    b, i = np.flatnonzero(boundary), np.flatnonzero(~boundary)
    full = np.zeros((len(boundary),) * 2, dtype=np.result_type(F, S))
    full[np.ix_(i, i)] = F
    full[np.ix_(b, i)] = S @ F
    return full


# ---------------------------------------------------------------------------
# Functions of dense matrices
# ---------------------------------------------------------------------------

class _Spectrum:
    """The scalar function on the eigenvalues of all blocks of a problem.

    Round-off is removed from the eigenvalues first: negligible imaginary
    parts and eigenvalues that are zero to working precision.  Otherwise a
    zero mode would come out as a huge value of a negative power, and as a
    spurious "negative eigenvalue" of a fractional one.  A real spectrum is
    passed to the function as a real array.
    """

    def __init__(self, func: _Function, eigenvalues: NDArray) -> None:
        self.func = func
        self.scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        lam = self._clean(eigenvalues)
        self.real = not np.any(lam.imag)
        self.eigenvalues = lam
        if func.check is not None:
            func.check(lam)
        self.values = self(lam)
        if not np.all(np.isfinite(self.values)):
            raise ValueError(
                f"{func.name} is not finite on the spectrum of the operator "
                f"(eigenvalues with real parts from {lam.real.min():.4g} to {lam.real.max():.4g})."
            )
        # Does the function commute with complex conjugation on the spectrum?
        # Then the function of a real operator is real.
        size = float(np.max(np.abs(self.values))) if lam.size else 0.0
        mirror = self.values.conj() if self.real else self(lam.conj()).conj()
        self.symmetric = bool(np.all(np.abs(self.values - mirror) <= _ROUNDOFF * size))

    def _clean(self, eigenvalues: NDArray) -> NDArray:
        lam = np.array(eigenvalues, dtype=complex)
        tol = _ROUNDOFF * self.scale
        lam.imag[np.abs(lam.imag) <= tol] = 0.0
        lam[np.abs(lam) <= tol] = 0.0
        return lam

    def __call__(self, eigenvalues: NDArray) -> NDArray:
        lam = self._clean(eigenvalues)
        with np.errstate(all="ignore"):
            values = np.asarray(self.func.scalar(lam.real if self.real else lam))
        if values.shape != lam.shape:
            try:
                values = np.broadcast_to(values, lam.shape)
            except ValueError:
                raise ValueError(
                    f"{self.func.name} must return one value per eigenvalue: it was called with an "
                    f"array of shape {lam.shape} and returned one of shape {values.shape}."
                ) from None
        if values.dtype.kind not in "fc":
            values = values.astype(float)
        return values


def _functions(matrices, func: _Function, real: bool) -> tuple[list[NDArray], bool, float]:
    """The function of each of the square *matrices* (the blocks of one operator).

    Returns the list of results, whether the function of the operator is
    real (for a real operator, given by *real*; the blocks themselves are
    complex if they belong to Fourier modes), and the largest estimated
    relative error.
    """
    if len(matrices) and matrices[0].shape[0] == 1:
        # Numbers instead of matrices: the symbol of a fully shift-invariant operator.
        spectrum = _Spectrum(func, np.array([M[0, 0] for M in matrices]))
        return list(spectrum.values.reshape(-1, 1, 1)), real and spectrum.symmetric, 0.0
    decompositions = [_eigendecomposition(M) for M in matrices]
    spectrum = _Spectrum(func, np.concatenate([d[0] for d in decompositions]))
    real = real and spectrum.symmetric
    results = []
    error = 0.0
    offset = 0
    for M, (w, V, V_inv, cond) in zip(matrices, decompositions):
        values = spectrum.values[offset:offset + len(w)]
        lam = spectrum.eigenvalues[offset:offset + len(w)]
        offset += len(w)
        if cond <= _MAX_EIGENVECTOR_COND:
            F = (V * values) @ V_inv
        else:
            F, estimate = _schur_function(M, spectrum, lam, values, V, V_inv, cond)
            error = max(error, estimate)
        results.append(F)
    return results, real, error


def _eigendecomposition(M: NDArray) -> tuple[NDArray, NDArray, NDArray | None, float]:
    """``(w, V, V_inv, cond)`` with ``M = V diag(w) V_inv`` and the 1-norm condition number of *V*."""
    n = M.shape[0]
    scale = np.max(np.abs(M)) if n else 0.0
    if n <= 1 or scale == 0:
        eye = np.eye(n)
        return np.diag(M).astype(complex), eye, eye, 1.0
    if np.max(np.abs(M - M.conj().T)) <= _HERMITIAN_TOL * scale:
        w, V = scipy.linalg.eigh(M)
        return w.astype(complex), V, V.conj().T, 1.0
    w, V = scipy.linalg.eig(M)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            V_inv = scipy.linalg.inv(V)
    except np.linalg.LinAlgError:  # defective matrix
        return w, V, None, np.inf
    cond = np.linalg.norm(V, 1) * np.linalg.norm(V_inv, 1)
    return w, V, V_inv, float(cond) if np.isfinite(cond) else np.inf


def _schur_function(M, spectrum: _Spectrum, lam, values, V, V_inv, cond) -> tuple[NDArray, float]:
    """The function of a matrix that is far from normal, with its estimated relative error."""
    func = spectrum.func
    eps = np.finfo(float).eps
    if func.matrix is not None and not (func.regular_only and np.any(lam == 0)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                F = func.matrix(M)
        except np.linalg.LinAlgError:
            F = None
        if F is not None and np.all(np.isfinite(F)):
            return F, eps
    # No dedicated algorithm: take the better of the Schur-Parlett method (which
    # suffers from close eigenvalues) and the eigendecomposition.
    F, estimate = scipy.linalg.funm(M, spectrum, disp=False)
    if V_inv is not None and not estimate <= eps * cond:
        F, estimate = (V * values) @ V_inv, eps * cond
    return F, float(estimate)

r"""Time integrators for stiff semilinear problems :math:`u_t = L u + N(t, u)`.

Many nonlinear PDEs consist of a stiff linear part *L* (diffusion,
dispersion: the highest derivatives) and a nonlinear part *N* that is not
stiff.  Explicit methods need tiny time steps for the linear part, and
implicit methods have to solve nonlinear systems.  The schemes of this
module treat the two parts differently and need neither:

* **Exponential time differencing** (``"ETDRK4"``, ``"ETDRK2"``,
  ``"ETD1"``) integrates the linear part exactly, with the matrix
  exponential :math:`e^{hL}` and the related functions
  :math:`\varphi_k(hL)`, and the nonlinear part with an explicit
  Runge-Kutta scheme.
* The **implicit-explicit** scheme ``"IMEX2"`` treats the linear part with
  an implicit Runge-Kutta scheme, which only needs solutions of linear
  systems, and the nonlinear part explicitly.

They are used through :func:`~numgrids.evolution.evolve` with the argument
``linear=``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import scipy.fft
import scipy.linalg
import scipy.sparse
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numgrids.evolution import EvolutionResult
    from numgrids.grids import Grid

#: The methods of :func:`~numgrids.evolution.evolve` for semilinear problems.
ETD_METHODS = ("ETDRK4", "ETDRK2", "ETD1")
IMEX_METHODS = ("IMEX2",)
METHODS = ETD_METHODS + IMEX_METHODS

# Up to this many unknowns, the ETD methods compute dense matrix functions.
_ETD_DENSE_LIMIT = 2500
# Systems of several fields are solved with dense LU up to this size, if at least this dense.
_SOLVE_DENSE_LIMIT = 4000
_SOLVE_DENSE_DENSITY = 0.01
# The coefficients of the schemes are kept for this many different step sizes.
_MAX_STEPPERS = 4
# Matrix phi-functions: Taylor polynomial of this degree for matrices of norm <= _PHI_THETA.
_PHI_THETA = 0.5
_PHI_DEGREE = 14
# Scalar phi-functions: Taylor polynomial of this degree for |z| < _PHI_SCALAR_RADIUS.
_PHI_SCALAR_RADIUS = 2.0
_PHI_SCALAR_DEGREE = 26

# Exponential Runge-Kutta schemes: the nodes c and the coefficients a[i][j], b[j],
# written as {k: weight} for the sum of weight * phi_k(c_i h L) (for b: phi_k(h L)).
_SCHEMES = {
    # Exponential Euler method
    "ETD1": ((0.0,), ((),), ({1: 1.0},)),
    # Cox & Matthews, J. Comput. Phys. 176 (2002): ETD2RK
    "ETDRK2": ((0.0, 1.0), ((), ({1: 1.0},)), ({1: 1.0, 2: -1.0}, {2: 1.0})),
    # Krogstad, J. Comput. Phys. 203 (2005): the weights of Cox & Matthews' ETD4RK
    # with internal stages that are accurate to one order more for stiff problems
    "ETDRK4": (
        (0.0, 0.5, 0.5, 1.0),
        ((), ({1: 0.5},), ({1: 0.5, 2: -1.0}, {2: 1.0}), ({1: 1.0, 2: -2.0}, {}, {2: 2.0})),
        ({1: 1.0, 2: -3.0, 3: 4.0}, {2: 2.0, 3: -4.0}, {2: 2.0, 3: -4.0}, {2: -1.0, 3: 4.0}),
    ),
}


# ── phi-functions ───────────────────────────────────────────────


def _phi_scalar(z, kmax: int) -> NDArray:
    r"""The functions :math:`\varphi_0, \ldots, \varphi_{k_{max}}` of numbers, elementwise.

    :math:`\varphi_0(z) = e^z` and :math:`\varphi_{k+1}(z) = (\varphi_k(z) - 1/k!)/z`,
    i.e. :math:`\varphi_1 = (e^z - 1)/z`, :math:`\varphi_2 = (e^z - 1 - z)/z^2`, ...
    The recurrence loses accuracy by cancellation for small :math:`|z|`,
    so the Taylor series :math:`\varphi_k(z) = \sum_j z^j/(j+k)!` is used
    for :math:`|z| < 2`.

    Returns an array of shape ``(kmax + 1,) + z.shape``, real for real *z*.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids.etd import _phi_scalar
    >>> phi = _phi_scalar([0.0, 1e-3, -3.0], 2)
    >>> np.round(phi[2], 6)                       # (exp(z) - 1 - z) / z^2
    array([0.5     , 0.500167, 0.227754])
    """
    z = np.asarray(z)
    z = z.astype(np.result_type(z, float), copy=False)
    out = np.empty((kmax + 1,) + z.shape, dtype=z.dtype)
    small = np.abs(z) < _PHI_SCALAR_RADIUS
    large = np.where(small, 1.0, z)
    out[0] = np.exp(z)
    p = np.exp(large)
    for k in range(1, kmax + 1):
        p = (p - 1.0 / math.factorial(k - 1)) / large
        out[k] = p
    zs = z[small]
    if zs.size:
        for k in range(1, kmax + 1):
            s = np.full(zs.shape, 1.0 / math.factorial(_PHI_SCALAR_DEGREE + k), dtype=z.dtype)
            for j in range(_PHI_SCALAR_DEGREE - 1, -1, -1):
                s = s * zs + 1.0 / math.factorial(j + k)
            out[k][small] = s
    return out


def _phi_double(phis: list[NDArray]) -> list[NDArray]:
    r"""The matrices :math:`\varphi_k(2Z)` from the matrices :math:`\varphi_k(Z)`.

    :math:`\varphi_k(2Z) = 2^{-k} \big[\varphi_0(Z) \varphi_k(Z) + \sum_{j=1}^k \varphi_j(Z)/(k-j)!\big]`.
    """
    E = phis[0]
    out = [E @ E]
    for k in range(1, len(phis)):
        acc = E @ phis[k]
        for j in range(1, k + 1):
            acc = acc + phis[j] / math.factorial(k - j)
        out.append(acc / 2.0 ** k)
    return out


def _phi_dense(Z: NDArray, kmax: int) -> NDArray:
    r"""The functions :math:`\varphi_0, \ldots, \varphi_{k_{max}}` of square matrices.

    Scaling and modified squaring (Skaflestad & Wright, Appl. Numer. Math.
    59, 2009): the matrix is divided by a power of two such that its norm is
    at most 1/2, where the Taylor series converges quickly, and the result
    is doubled back with :func:`_phi_double`.  No linear systems are
    solved, so singular matrices (zero eigenvalues) are no problem.

    *Z* may be a stack of matrices of shape ``(..., n, n)``; each one gets
    its own scaling.  Returns an array of shape ``(kmax + 1,) + Z.shape``.

    Examples
    --------
    >>> import numpy as np
    >>> from numgrids.etd import _phi_dense
    >>> Z = np.array([[0.0, 1.0], [0.0, 0.0]])          # nilpotent: phi_1 = 1 + Z/2
    >>> _phi_dense(Z, 1)[1]
    array([[1. , 0.5],
           [0. , 1. ]])
    """
    Z = np.asarray(Z)
    n = Z.shape[-1]
    stack = Z.reshape((-1, n, n))
    stack = stack.astype(np.result_type(stack, float), copy=False)
    norms = np.abs(stack).sum(axis=-2).max(axis=-1, initial=0.0)
    if not np.all(np.isfinite(norms)):
        raise ValueError("The linear part has entries that are not finite.")
    squarings = np.zeros(len(stack), dtype=int)
    big = norms > _PHI_THETA
    squarings[big] = np.ceil(np.log2(norms[big] / _PHI_THETA)).astype(int)
    out = np.empty((kmax + 1,) + stack.shape, dtype=stack.dtype)
    eye = np.eye(n)
    for s in np.unique(squarings):
        sel = squarings == s
        X = stack[sel] / 2.0 ** s
        P = np.broadcast_to(eye / math.factorial(_PHI_DEGREE + kmax), X.shape)
        for j in range(_PHI_DEGREE - 1, -1, -1):
            P = X @ P + eye / math.factorial(j + kmax)
        phis = [P]
        for k in range(kmax, 0, -1):          # phi_{k-1}(X) = X phi_k(X) + 1/(k-1)!
            phis.insert(0, X @ phis[0] + eye / math.factorial(k - 1))
        for _ in range(s):
            phis = _phi_double(phis)
        out[:, sel] = phis
    return out.reshape((kmax + 1,) + Z.shape)


# ── the linear part in a form suited for the time stepping ──────


class _FourierProblem:
    r"""A semilinear problem whose linear part is diagonal in Fourier space.

    On a grid that is periodic along all axes, an operator with constant
    coefficients is a circulant matrix: it multiplies the Fourier mode
    :math:`e^{i k x}` by a number, its symbol :math:`\lambda(k)`.  The state
    of the time stepping are the Fourier coefficients of the fields.  All
    functions of the linear part are then arrays of numbers (or, if the
    linear part couples the fields, one small matrix per mode), and the
    nonlinear part is evaluated on the grid, between two fast Fourier
    transforms.
    """

    def __init__(self, grid: Grid, symbols: NDArray, real: bool, fields: list[NDArray], evaluate,
                 source_fn, constant_source: bool) -> None:
        self.shape = grid.shape
        self.size = grid.size
        self.nfields = nfields = len(fields)
        self.real = real
        self._axes = tuple(range(len(self.shape)))
        if real:  # real fields: half of the modes suffice
            symbols = symbols[..., : self.shape[-1] // 2 + 1]
        off_diagonal = [symbols[i, j] for i in range(nfields) for j in range(nfields) if i != j]
        self.coupled = any(np.any(s) for s in off_diagonal)
        if self.coupled:
            # one nfields x nfields matrix per mode; state of shape modes + (nfields,)
            self.symbol = np.moveaxis(symbols, (0, 1), (-2, -1))
        else:
            self.symbol = np.stack([symbols[i, i] for i in range(nfields)])
            if not np.any(self.symbol.imag):
                self.symbol = self.symbol.real
        self._evaluate = evaluate
        self.z0 = self._forward(fields)
        self._source_fn = source_fn
        self._source = None
        if source_fn is not None and constant_source:
            self._source = self._forward_flat(source_fn(0.0))
        self.has_forcing = source_fn is not None

    def _forward(self, parts) -> NDArray:
        if self.real:
            coeffs = [scipy.fft.rfftn(np.asarray(p, dtype=float), axes=self._axes) for p in parts]
        else:
            coeffs = [scipy.fft.fftn(p, axes=self._axes) for p in parts]
        return np.stack(coeffs, axis=-1 if self.coupled else 0)

    def _forward_flat(self, y: NDArray) -> NDArray:
        return self._forward(y.reshape((self.nfields,) + self.shape))

    def _backward(self, z: NDArray) -> list[NDArray]:
        coeffs = np.moveaxis(z, -1, 0) if self.coupled else z
        if self.real:
            return [scipy.fft.irfftn(c, s=self.shape, axes=self._axes) for c in coeffs]
        return [scipy.fft.ifftn(c, axes=self._axes) for c in coeffs]

    def nonlinear(self, t: float, z: NDArray) -> NDArray:
        return self._forward(self._evaluate(t, self._backward(z)))

    def forcing(self, t: float) -> NDArray:
        if self._source is not None:
            return self._source
        return self._forward_flat(self._source_fn(t))

    def output(self, z: NDArray, t: float) -> NDArray:
        return np.concatenate([p.ravel() for p in self._backward(z)])

    def clean(self, z: NDArray) -> NDArray:
        r"""Remove the part of the state that does not belong to real fields (in place).

        The half spectrum of a real field still contains some coefficients
        twice, as complex conjugates: those with the wave number 0 (or the
        highest one) along the last axis, for :math:`+k` and :math:`-k`
        along the other axes.  The inverse transform only sees the
        Hermitian part of these pairs.  The other part is fed by round-off
        and never reaches the nonlinear terms, so nothing limits it: a
        linear part with growing modes would amplify it without bound.
        """
        if not self.real:
            return z
        coeffs = np.moveaxis(z, -1, 0) if self.coupled else z           # fields first
        n = self.shape[-1]
        lead = tuple(range(1, coeffs.ndim - 1))                         # the other axes of the grid
        for index in {0, n // 2} if n % 2 == 0 else {0}:
            plane = coeffs[..., index]
            mirrored = np.conj(np.roll(np.flip(plane, lead), 1, lead)) if lead else np.conj(plane)
            coeffs[..., index] = 0.5 * (plane + mirrored)
        return z

    def phi_set(self, h: float, nodes, kmax: int) -> dict:
        """``{c: [phi_0(c h L), ..., phi_kmax(c h L)]}`` for the nodes *c*."""
        if self.coupled:
            return {c: _phi_dense(c * h * self.symbol, kmax) for c in nodes}
        return {c: _phi_scalar(c * h * self.symbol, kmax) for c in nodes}

    def apply(self, coef: NDArray, z: NDArray) -> NDArray:
        if self.coupled:
            return np.matmul(coef, z[..., None])[..., 0]
        return coef * z

    def solver(self, alpha: float) -> Callable[[NDArray], NDArray]:
        """A function solving ``(1 - alpha L) x = b``."""
        if self.coupled:
            inverse = np.linalg.inv(np.eye(self.nfields) - alpha * self.symbol)
        else:
            inverse = 1.0 / (1.0 - alpha * self.symbol)
        return lambda b: self.apply(inverse, b)


class _MatrixProblem:
    r"""A semilinear problem with a general linear part, given as a matrix.

    The state is the flat array of all values on the grid, as in the other
    paths of :func:`~numgrids.evolution.evolve`: the boundary conditions
    :math:`u_b = S u_i + c(t)` are folded into the matrix *A* of the linear
    part, and the rows of the boundary points are zero, so that these
    entries of the state do not change.  What the boundary values and the
    source contribute to the right-hand side does not depend on the
    solution; it is the *forcing* :math:`q(t)`.
    """

    def __init__(self, grid: Grid, A, constraints, zero_rows: NDArray, y0: NDArray, evaluate,
                 inhomogeneity, constant: bool) -> None:
        self.grid = grid
        self.nfields = len(constraints)
        self.A = scipy.sparse.csr_matrix(A)
        self._constraints = constraints
        self._zero_rows = zero_rows
        self._evaluate = evaluate
        self.z0 = y0
        self._inhomogeneity = inhomogeneity
        self._q = inhomogeneity(0.0) if inhomogeneity is not None and constant else None
        self.has_forcing = inhomogeneity is not None and (self._q is None or bool(np.any(self._q)))
        self._dense: NDArray | None = None
        self._eig = None

    def nonlinear(self, t: float, y: NDArray) -> NDArray:
        size, shape = self.grid.size, self.grid.shape
        parts = []
        for f, con in enumerate(self._constraints):
            part = y[f * size:(f + 1) * size]
            if con is not None:
                part = con.project(part, t)
            parts.append(part.reshape(shape))
        dy = np.concatenate([o.ravel() for o in self._evaluate(t, parts)])
        dy[self._zero_rows] = 0
        return dy

    def forcing(self, t: float) -> NDArray:
        return self._q if self._q is not None else self._inhomogeneity(t)

    def output(self, z: NDArray, t: float) -> NDArray:
        return z

    @staticmethod
    def clean(z: NDArray) -> NDArray:
        return z

    def _prepare_dense(self) -> None:
        n = self.A.shape[0]
        if n > _ETD_DENSE_LIMIT:
            raise ValueError(
                f"The ETD methods need dense matrix functions of the linear part here, which is limited to "
                f"{_ETD_DENSE_LIMIT} unknowns (this problem has {n}). Only on grids that are periodic along "
                f'all axes, a linear part with constant coefficients needs no matrices. Use method="IMEX2", '
                f"which solves linear systems instead."
            )
        self._dense = self.A.toarray()
        scale = np.max(np.abs(self._dense), initial=0.0)
        if scale == 0:
            return
        # A normal matrix has an orthonormal eigenbasis, in which functions of it are diagonal.
        AH = self._dense.conj().T
        if np.max(np.abs(self._dense - AH)) <= 1e-13 * scale:
            w, V = scipy.linalg.eigh(self._dense)
            self._eig = (w, V)
        elif np.max(np.abs(self._dense + AH)) <= 1e-13 * scale:
            w, V = scipy.linalg.eigh(1j * self._dense)      # i A is Hermitian
            self._eig = (-1j * w, V)

    def phi_set(self, h: float, nodes, kmax: int) -> dict:
        """``{c: [phi_0(c h A), ..., phi_kmax(c h A)]}`` for the nodes *c*."""
        if self._dense is None:
            self._prepare_dense()
        if self._eig is not None:
            lam, V = self._eig
            real = not np.iscomplexobj(self._dense)
            out = {}
            for c in nodes:
                mats = [(V * p) @ V.conj().T for p in _phi_scalar(c * h * lam, kmax)]
                out[c] = [m.real for m in mats] if real else mats
            return out
        nodes = sorted(nodes)
        if nodes == [0.5, 1.0]:  # the last doubling gives the functions of the full step
            half = list(_phi_dense(0.5 * h * self._dense, kmax))
            return {0.5: half, 1.0: _phi_double(half)}
        return {c: list(_phi_dense(c * h * self._dense, kmax)) for c in nodes}

    @staticmethod
    def apply(coef: NDArray, z: NDArray) -> NDArray:
        return coef @ z

    def solver(self, alpha: float) -> Callable[[NDArray], NDArray]:
        """A function solving ``(1 - alpha A) x = b``."""
        from numgrids.linalg import _dense_solver, _sparse_solver
        from numgrids.operators import Operator

        n = self.A.shape[0]
        K = scipy.sparse.identity(n, format="csr") - alpha * self.A
        if self.nfields == 1:
            # The solver of operators decouples Fourier modes where it can, and
            # chooses between dense and sparse factorizations otherwise.
            solve = Operator(self.grid, K).factorize()
            return lambda b: solve(b.reshape(self.grid.shape)).ravel()
        if n <= _SOLVE_DENSE_LIMIT and K.nnz >= _SOLVE_DENSE_DENSITY * n * n:
            return _dense_solver(K.toarray())
        return _sparse_solver(K)


# ── time stepping ───────────────────────────────────────────────


def _etd_stepper(problem, scheme, h: float) -> Callable[[float, NDArray], NDArray]:
    r"""One step of an exponential Runge-Kutta scheme, as a function ``step(t, z)``.

    With the explicit part :math:`F = N + q`, the stages are
    :math:`U_i = e^{c_i h L} u_n + h \sum_{j<i} a_{ij}(hL) F(t_n + c_j h, U_j)` and
    :math:`u_{n+1} = e^{hL} u_n + h \sum_j b_j(hL) F(t_n + c_j h, U_j)`.
    """
    c, a, b = scheme
    kmax = max(k for spec in b + tuple(s for row in a for s in row) for k in spec)
    phis = problem.phi_set(h, sorted({ci for ci in c if ci} | {1.0}), kmax)

    def combine(spec, ci):
        return h * sum(w * phis[ci][k] for k, w in spec.items())

    propagators = [None if ci == 0 else phis[ci][0] for ci in c]
    stages = [[(j, combine(spec, c[i])) for j, spec in enumerate(row) if spec] for i, row in enumerate(a)]
    # Stages with the same weight are added up before the weight is applied.
    groups: dict[tuple, list[int]] = {}
    for j, spec in enumerate(b):
        groups.setdefault(tuple(sorted(spec.items())), []).append(j)
    final = [(js, combine(dict(key), 1.0)) for key, js in groups.items()]
    E = phis[1.0][0]
    apply = problem.apply
    nonlinear = problem.nonlinear
    forcing = problem.forcing if problem.has_forcing else None

    def F(t, z):
        return nonlinear(t, z) if forcing is None else nonlinear(t, z) + forcing(t)

    def step(t: float, z: NDArray) -> NDArray:
        Fs = [F(t, z)]
        for i in range(1, len(c)):
            U = apply(propagators[i], z)
            for j, coef in stages[i]:
                U = U + apply(coef, Fs[j])
            Fs.append(F(t + c[i] * h, U))
        out = apply(E, z)
        for js, coef in final:
            out = out + apply(coef, Fs[js[0]] if len(js) == 1 else sum(Fs[j] for j in js))
        return out

    return step


def _imex_stepper(problem, h: float) -> Callable[[float, NDArray], NDArray]:
    r"""One step of the implicit-explicit Runge-Kutta scheme ARS(2,2,2).

    Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997): the linear part
    and the forcing are integrated with an L-stable, two-stage, diagonally
    implicit scheme with the diagonal entry :math:`\gamma = 1 - 1/\sqrt{2}`,
    the nonlinear part with an explicit scheme on the same nodes.  Both
    stages solve a system with the matrix :math:`1 - \gamma h L`.
    """
    gamma = 1.0 - 1.0 / math.sqrt(2.0)
    delta = 1.0 - 1.0 / (2.0 * gamma)
    solve = problem.solver(gamma * h)
    nonlinear = problem.nonlinear
    forcing = problem.forcing if problem.has_forcing else None

    def step(t: float, z: NDArray) -> NDArray:
        N1 = nonlinear(t, z)
        b = z + gamma * h * N1
        if forcing is not None:
            b = b + gamma * h * forcing(t + gamma * h)
        U = solve(b)
        K = (U - z) / (gamma * h) - N1                # L U + q: the implicit part at the first stage
        N2 = nonlinear(t + gamma * h, U)
        b = z + h * ((1.0 - gamma) * K + delta * N1 + (1.0 - delta) * N2)
        if forcing is not None:
            b = b + gamma * h * forcing(t + h)
        return solve(b)

    return step


# ── driver ──────────────────────────────────────────────────────


def _is_symbol(block) -> bool:
    """Whether a block of the linear part is a Fourier symbol: a function of the wave numbers."""
    from numgrids.operators import Operator

    return callable(block) and not isinstance(block, Operator) and not hasattr(block, "as_matrix")


def _wavenumbers(grid: Grid) -> list[NDArray]:
    """The meshed wave numbers of the discrete Fourier transform on a periodic grid."""
    k = [2 * np.pi * scipy.fft.fftfreq(len(axis), d=axis.period / len(axis)) for axis in grid.axes]
    return np.meshgrid(*k, indexing="ij")


def _matrix_symbol(grid: Grid, block) -> NDArray | None:
    """Fourier symbol of a sparse matrix on a periodic grid; ``None`` if it is not shift invariant."""
    block = scipy.sparse.csr_matrix(block)
    if block.nnz == 0:
        return np.zeros(grid.shape)
    # The eigenvalues of a circulant matrix are the Fourier transform of its first row.
    symbol = scipy.fft.ifftn(block[0].toarray().reshape(grid.shape)) * grid.size
    # A shift-invariant matrix acts on any field like its symbol on the Fourier coefficients.
    # (One product with a random field is much cheaper than comparing the matrix with its shifts.)
    v = np.random.default_rng(0).standard_normal(grid.shape)
    by_symbol = scipy.fft.ifftn(symbol * scipy.fft.fftn(v)).ravel()
    if np.max(np.abs(block @ v.ravel() - by_symbol)) > 1e-10 * np.max(np.abs(block.data)):
        return None
    # Round-off must not turn dispersion into (anti-)damping or vice versa.
    tiny = 1e-13 * np.max(np.abs(symbol))
    if np.max(np.abs(symbol.imag)) <= tiny:
        symbol = symbol.real
    elif np.max(np.abs(symbol.real)) <= tiny:
        symbol = 1j * symbol.imag
    return symbol


def _fourier_symbols(grid: Grid, linear, nfields: int, system: bool, real: bool) -> tuple[NDArray, bool] | None:
    r"""Fourier symbols of the blocks of the linear part on a grid that is periodic along all axes.

    Returns the symbols, shape ``(nfields, nfields) + grid.shape``, and
    whether they belong to a real operator, :math:`\lambda(-k) =
    \overline{\lambda(k)}`; or ``None`` if a block is an operator that is
    not shift invariant (variable coefficients).

    A block is a function of the meshed wave numbers, or anything that has
    a matrix (operator, array, number).  The highest wave number of an axis
    with an even number of points stands for :math:`+k` and :math:`-k` at
    once; if the fields are real (*real*), a symbol given as a function is
    replaced by the mean of both there, as in the matrices of the
    differentiation operators (which is why odd derivatives vanish for
    this mode).
    """
    from numgrids.nonlinear import _assemble_jacobian

    if system:
        if (len(linear) != nfields
                or any(not isinstance(row, (list, tuple)) or len(row) != nfields for row in linear)):
            raise ValueError(f"The linear part of a system of {nfields} fields must be a "
                             f"{nfields} x {nfields} nested list of blocks.")
        blocks = linear
    else:
        blocks = [[linear]]
    given = any(_is_symbol(block) for row in blocks for block in row)
    K = _wavenumbers(grid) if given else None
    symbols = np.zeros((nfields, nfields) + grid.shape, dtype=complex)
    real_operator = True
    for i in range(nfields):
        for j in range(nfields):
            block = blocks[i][j]
            if block is None:
                continue
            if _is_symbol(block):
                try:
                    symbol = np.broadcast_to(np.asarray(block(*K)), grid.shape)
                except ValueError:
                    raise ValueError(
                        f"The symbol of the linear part must return an array of shape {grid.shape} "
                        f"for the {grid.ndims} meshed wave numbers."
                    ) from None
            else:
                matrix = _assemble_jacobian(grid, block, 1)
                symbol = _matrix_symbol(grid, matrix)
                if symbol is None:
                    if given:
                        raise ValueError(
                            "Blocks given as Fourier symbols can only be combined with operators "
                            "that have constant coefficients."
                        )
                    return None
                real_operator = real_operator and not np.iscomplexobj(matrix)
            symbols[i, j] = symbol
    if not np.all(np.isfinite(symbols)):
        raise ValueError("The symbol of the linear part has values that are not finite.")
    if given:
        # Wave numbers that are their own negatives: the highest one of an even number of points.
        own_negative = np.zeros(grid.shape, dtype=bool)
        for axis, n in enumerate(grid.shape):
            if n % 2 == 0:
                index = [slice(None)] * grid.ndims
                index[axis] = n // 2
                own_negative[tuple(index)] = True
        axes = tuple(range(2, symbols.ndim))
        mirrored = np.roll(np.flip(symbols, axes), 1, axes).conj()          # conj(lambda(-k))
        scale = np.max(np.abs(symbols), initial=0.0)
        if np.any(np.abs(mirrored - symbols)[..., ~own_negative] > 1e-12 * scale):
            real_operator = False
        if real and real_operator:
            symbols[..., own_negative] = 0.5 * (symbols + mirrored)[..., own_negative]
    return symbols, real_operator


def _physical_rhs(func, grid: Grid, nfields: int, system: bool):
    """The nonlinear part as a function ``(t, fields) -> list of arrays of grid shape``."""
    shape = grid.shape

    def evaluate(t, parts):
        if func is None:
            return [np.zeros(shape) for _ in range(nfields)]
        out = func(t, *parts)
        if system:
            if np.ndim(out) == 0 or len(out) != nfields:
                found = "a scalar" if np.ndim(out) == 0 else f"{len(out)} arrays"
                raise ValueError(f"rhs returned {found}, expected {nfields} arrays.")
        else:
            out = [out]
        try:
            return [np.broadcast_to(np.asarray(o), shape) for o in out]
        except ValueError:
            shapes = ", ".join(str(np.shape(o)) for o in out)
            raise ValueError(f"rhs returned an array of wrong shape ({shapes}), expected {shape}.") from None

    return evaluate


def _evolve_semilinear(grid: Grid, rhs, linear, fields: list[NDArray], constraints, zero_rows: NDArray,
                       moving_bcs: bool, source_fn, constant_source: bool, t_span, t_eval, method: str, dt,
                       system: bool) -> EvolutionResult:
    """Integrate :math:`u_t = L u + N(t, u)` with a fixed time step.

    Called by :func:`~numgrids.evolution.evolve`, which prepares the
    initial *fields*, the boundary *constraints* (with the points that are
    not evolved in *zero_rows*) and the source.
    """
    from numgrids.evolution import _check_linear_part, _linear_part, _output_times, _result
    from numgrids.operators import Operator

    if dt is None:
        raise ValueError(f'method="{method}" integrates with a fixed time step: pass it as dt=...')
    dt = float(dt)
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"The time step dt must be a positive number, got {dt}.")
    if rhs is not None and (isinstance(rhs, Operator) or not callable(rhs)):
        raise TypeError(
            f"With linear=, rhs is the nonlinear part: a function rhs(t, u) or None, not {type(rhs).__name__}."
        )
    t0 = float(t_span[0])
    t_out = _output_times(t_span, t_eval)
    nfields = len(fields)
    evaluate = _physical_rhs(rhs, grid, nfields, system)

    # One evaluation of the nonlinear part and the source tells whether the problem is real.
    probe = [f.ravel() for f in fields] + [np.ravel(o) for o in evaluate(t0, fields)]
    if source_fn is not None:
        probe.append(source_fn(t0))
    dtype = np.result_type(float, *probe)
    constrained = any(con is not None for con in constraints) or zero_rows.any()

    blocks = [linear]
    if system and isinstance(linear, (list, tuple)):
        blocks = [b for row in linear if isinstance(row, (list, tuple)) for b in row]
    given = any(_is_symbol(b) for b in blocks)
    if _is_symbol(linear):
        if system:
            raise TypeError(
                "A single Fourier symbol only works for a single field; "
                "for systems pass a nested list of blocks, e.g. [[None, 1], [symbol, None]]."
            )
    else:
        _check_linear_part(linear, system, "linear part")
    fourier = not constrained and all(axis.periodic for axis in grid.axes)
    if given and not fourier:
        raise ValueError(
            "A linear part given by its Fourier symbol (a function of the wave numbers) needs a grid "
            "that is periodic along all axes, and no boundary conditions. Pass an Operator instead."
        )
    symbols = None
    if fourier:
        found = _fourier_symbols(grid, linear, nfields, system, not np.issubdtype(dtype, np.complexfloating))
        if found is not None:
            symbols, real_operator = found
            if not real_operator:
                dtype = np.result_type(dtype, complex)
    if symbols is None:
        M, A, inhomogeneity = _linear_part(grid, linear, nfields, system, constraints, zero_rows, source_fn,
                                           "linear part")
        dtype = np.result_type(dtype, A.dtype)
    real = not np.issubdtype(dtype, np.complexfloating)
    if symbols is not None:
        problem = _FourierProblem(grid, symbols, real, fields, evaluate, source_fn, constant_source)
    else:
        y0 = np.concatenate([f.ravel() for f in fields]).astype(dtype)
        problem = _MatrixProblem(grid, A, constraints, zero_rows, y0, evaluate, inhomogeneity,
                                 constant_source and not moving_bcs)

    steppers: dict[float, Callable] = {}
    Y = np.empty((len(t_out), nfields * grid.size), dtype=dtype)
    y_start = np.concatenate([f.ravel() for f in fields])
    z = problem.z0
    now = t0
    nsteps = 0
    done = 0
    message = None
    with np.errstate(over="ignore", invalid="ignore"):
        for k, tk in enumerate(t_out):
            interval = tk - now
            if interval != 0:
                # Equal steps, not longer than dt, that end exactly at the output time.
                n = max(1, math.ceil(abs(interval) / dt * (1 - 1e-12)))
                h = interval / n
                key = float(f"{h:.13e}")
                if key not in steppers:
                    if len(steppers) >= _MAX_STEPPERS:      # their matrices can be large
                        del steppers[next(iter(steppers))]
                    steppers[key] = (_imex_stepper(problem, h) if method in IMEX_METHODS
                                     else _etd_stepper(problem, _SCHEMES[method], h))
                step = steppers[key]
                for i in range(n):
                    z = problem.clean(step(now + i * h, z))
                    if not np.all(np.isfinite(z)):
                        message = (f"The solution became infinite or undefined near t = {now + (i + 1) * h:g}. "
                                   f"Try a smaller time step dt.")
                        break
                nsteps += n
                now = tk
            if message is not None:
                break
            # Before the first step, the output is the initial state itself (no round trip of transforms).
            Y[k] = problem.output(z, tk) if nsteps else y_start
            done = k + 1
    success = message is None
    if success:
        message = f"The integration reached the end of the time interval with {nsteps} steps of {method}."
    return _result(grid, t_out[:done], Y[:done], constraints, nfields, system, success, message, None)

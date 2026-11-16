"""Small numerical utilities used throughout numgrids."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse
from numpy.typing import NDArray
from scipy.sparse import kron, spmatrix


def multi_kron(*matrices: spmatrix) -> spmatrix:
    r"""Compute the Kronecker product of an arbitrary number of sparse matrices.

    The product is evaluated left-to-right:
    ``multi_kron(A, B, C)`` returns ``kron(kron(A, B), C)``.

    This is used internally to lift 1-D operators into multi-dimensional
    grids via the identity

    .. math::
        D_{\text{axis}\ k} = I_0 \otimes \cdots \otimes D_k \otimes \cdots \otimes I_{n-1}

    Parameters
    ----------
    *matrices : spmatrix
        Two or more scipy sparse matrices.

    Returns
    -------
    spmatrix
        The Kronecker product of all input matrices.
    """
    A = matrices[0]
    for i in range(1, len(matrices)):
        A = kron(A, matrices[i])
    return A


def embed_1d_operator(D: spmatrix, shape: Sequence[int], axis_index: int) -> spmatrix:
    """Lift a 1-D operator acting along *axis_index* to a full grid of *shape*.

    Parameters
    ----------
    D : spmatrix
        Square matrix of size ``shape[axis_index]``.
    shape : sequence of int
        Shape of the grid.
    axis_index : int
        The axis the operator acts on.

    Returns
    -------
    spmatrix
        Sparse matrix of shape ``(prod(shape), prod(shape))`` in CSC format.
    """
    if len(shape) == 1:
        return scipy.sparse.csc_matrix(D)
    factors = [
        D if i == axis_index else scipy.sparse.identity(n, format="csc")
        for i, n in enumerate(shape)
    ]
    return scipy.sparse.csc_matrix(multi_kron(*factors))


NORMS = ("max", "l2", "mean")


def vector_norm(values: NDArray, norm: str = "max") -> float:
    """Scalar measure of an array of (error) values.

    Parameters
    ----------
    values : NDArray
        Array of values (any shape).
    norm : str
        ``"max"`` (L-infinity), ``"l2"`` (root mean square) or ``"mean"``
        (mean absolute value).

    Returns
    -------
    float
    """
    values = np.abs(np.asarray(values))
    if norm == "max":
        return float(np.max(values))
    if norm == "l2":
        return float(np.sqrt(np.mean(values ** 2)))
    if norm == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unknown norm {norm!r}. Use one of {NORMS}.")

"""Sparse-matrix utilities for numgrids."""

from __future__ import annotations

from scipy.sparse import kron, spmatrix


def multi_kron(*matrices: spmatrix) -> spmatrix:
    """Compute the Kronecker product of an arbitrary number of sparse matrices.

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

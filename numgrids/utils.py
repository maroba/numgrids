from __future__ import annotations

from scipy.sparse import kron, spmatrix


def multi_kron(*matrices: spmatrix) -> spmatrix:
    """Compute the Kronecker product of multiple sparse matrices."""
    A = matrices[0]
    for i in range(1, len(matrices)):
        A = kron(A, matrices[i])
    return A

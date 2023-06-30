from scipy.sparse import kron


def multi_kron(*matrices):
    A = matrices[0]
    for i in range(1, len(matrices)):
        A = kron(A, matrices[i])
    return A

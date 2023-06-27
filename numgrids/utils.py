from scipy.sparse import kron


def as_array_like(x):
    if hasattr(x, "__len__"):
        return x
    return (x,)


def multi_kron(*matrices):
    A = matrices[0]
    for i in range(1, len(matrices)):
        A = kron(A, matrices[i])
    return A

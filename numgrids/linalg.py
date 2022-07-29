import numpy as np


def norm(vec):
    return np.sqrt(np.dot(vec, vec))


def angle(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2))

import numpy as np

from numgrids.linalg import norm, angle


class Parallelepiped:

    def __init__(self, base_vec_1, base_vec_2, offset):
        assert len(base_vec_1) == len(base_vec_2) == len(offset)
        assert len(base_vec_1) >= 2
        self.ndims = 2
        self.ndims_ambient = len(base_vec_1)
        self.base_vecs = v1, v2 = np.array(base_vec_1), np.array(base_vec_2)
        self.origin = np.array(offset)
        self.area = norm(v1) * norm(v2) * np.sin(angle(v1, v2))

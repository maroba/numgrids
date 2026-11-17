import numpy as np
import scipy
from numgrids import SphericalGrid, colatitude, periodic, radial

b = SphericalGrid(radial(12, 1.0), colatitude(14), periodic(16))
X, Y, Z = b.cartesian_coords
u = b.laplacian_operator().solve(lambda r, t, p: 6 * r * np.cos(t), bcs=lambda c: (c[0] * np.cos(c[1])) ** 3)
vals, _ = (-b.laplacian_operator()).eig(k=3, bcs=0)
print("ok", scipy.__version__, float(np.abs(u - Z ** 3).max()))

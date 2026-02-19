<h1 align="center">numgrids</h1>
<p align="center"> Working with numerical grids made easy.</p>

Setting up numerical grids, differentiation matrices, and coordinate
transformations by hand is tedious and error-prone.
*numgrids* gives you a high-level, NumPy-friendly API that handles all of
this — so you can focus on the physics or mathematics of your problem instead
of bookkeeping grid indices and scale factors.

**Main Features**

- Quickly define numerical grids for any rectangular or curvilinear coordinate system
- Multiple axis types: **equidistant**, **Chebyshev**, **logarithmic**, and **periodic**
- Built-in **spherical**, **cylindrical**, and **polar** coordinate grids
- **Custom curvilinear coordinates** — supply scale factors and get gradient, divergence, curl, and Laplacian automatically
- **Vector calculus operators**: gradient, divergence, curl, and Laplacian on curvilinear grids
- High-precision **spectral methods** (FFT + Chebyshev) selected automatically where possible
- Differentiation, integration, and interpolation
- **Boundary conditions**: Dirichlet, Neumann, and Robin — at the array level or inside sparse linear systems
- **Adaptive mesh refinement** with built-in Richardson-extrapolation error estimation
- **Save / load** grids and data to `.npz` files
- Multigrid hierarchies with inter-level transfer operators
- Fully compatible with *NumPy* and *SciPy*

## Installation

```shell
pip install --upgrade numgrids
```

## Quick Start

```python
from numgrids import *
import numpy as np

# Define axes
axis_r = create_axis(AxisType.CHEBYSHEV, 20, 0, 1)
axis_phi = create_axis(AxisType.EQUIDISTANT, 50, 0, 2*np.pi, periodic=True)

# Create grid and sample a function
grid = Grid(axis_r, axis_phi)
R, Phi = grid.meshed_coords
f = R**2 * np.sin(Phi)**2

# Differentiate
d_dr = Diff(grid, 1, 0)
df_dr = d_dr(f)

# Integrate
I = Integral(grid)
I(f * R)
```

Full documentation including API reference and example notebooks is available at
**[maroba.github.io/numgrids](https://maroba.github.io/numgrids/)**.

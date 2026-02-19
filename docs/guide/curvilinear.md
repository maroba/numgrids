# Curvilinear grids

Many physical problems are naturally expressed in non-Cartesian coordinates --
spherical coordinates for planetary atmospheres, cylindrical coordinates for
pipes, polar coordinates for 2D disk-shaped domains. numgrids provides
`CurvilinearGrid` and three built-in coordinate systems that give you
gradient, divergence, Laplacian, and curl out of the box.

```python
from numgrids import *
import numpy as np
```

## How it works

An orthogonal curvilinear coordinate system is fully characterized by its
*scale factors* $h_i(\mathbf{q})$. Given these, the standard
differential-geometry identities define all vector calculus operators:

**Gradient:**

$$
(\nabla f)_i = \frac{1}{h_i}\frac{\partial f}{\partial q_i}
$$

**Divergence:**

$$
\nabla \cdot \mathbf{v}
= \frac{1}{J}\sum_i \frac{\partial}{\partial q_i}
  \left(\frac{J}{h_i}\,v_i\right),
\quad J = \prod_i h_i
$$

**Laplacian:**

$$
\nabla^2 f
= \frac{1}{J}\sum_i \frac{\partial}{\partial q_i}
  \left(\frac{J}{h_i^2}\,\frac{\partial f}{\partial q_i}\right)
$$

**Curl (3D):**

$$
(\nabla \times \mathbf{v})_i
= \frac{1}{h_j h_k}\left[
  \frac{\partial(h_k\,v_k)}{\partial q_j}
  - \frac{\partial(h_j\,v_j)}{\partial q_k}\right]
$$

where $(i, j, k)$ is a cyclic permutation of $(0, 1, 2)$.

`CurvilinearGrid` takes a set of axes and scale-factor callables, then builds
these operators automatically using the grid's differentiation machinery.

## Built-in coordinate systems

### SphericalGrid $(r, \theta, \phi)$

Scale factors: $h_r = 1$, $h_\theta = r$, $h_\phi = r\sin\theta$.

```python
grid = SphericalGrid(
    create_axis(AxisType.CHEBYSHEV, 25, 0.5, 2.0),              # r
    create_axis(AxisType.CHEBYSHEV, 20, 0.1, np.pi - 0.1),      # theta
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2 * np.pi),  # phi
)

R, Theta, Phi = grid.meshed_coords
f = R**2 * np.sin(Theta)

lap_f = grid.laplacian(f)
grad_f = grid.gradient(f)           # (grad_r, grad_theta, grad_phi)
```

The three axes must be passed in the order $(r, \theta, \phi)$. The azimuthal
axis (phi) should be periodic.

### CylindricalGrid $(r, \phi, z)$

Scale factors: $h_r = 1$, $h_\phi = r$, $h_z = 1$.

```python
grid = CylindricalGrid(
    create_axis(AxisType.CHEBYSHEV, 20, 0.1, 2.0),               # r
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2 * np.pi),  # phi
    create_axis(AxisType.CHEBYSHEV, 20, -1.0, 1.0),              # z
)

R, Phi, Z = grid.meshed_coords
f = R**2 + Z**2
lap_f = grid.laplacian(f)  # should be close to 6 everywhere
```

### PolarGrid $(r, \phi)$

Scale factors: $h_r = 1$, $h_\phi = r$.

```python
grid = PolarGrid(
    create_axis(AxisType.CHEBYSHEV, 30, 0.1, 1.0),              # r
    create_axis(AxisType.EQUIDISTANT_PERIODIC, 40, 0, 2 * np.pi),  # phi
)

R, Phi = grid.meshed_coords
f = R**2 * np.cos(Phi)
lap_f = grid.laplacian(f)
```

## Vector calculus operators

All curvilinear grids provide four operators:

### `gradient(f)`

Takes a scalar field and returns a tuple of arrays, one per coordinate:

```python
grad_r, grad_theta, grad_phi = grid.gradient(f)
```

### `divergence(v_r, v_theta, v_phi)`

Takes the physical components of a vector field (one array per coordinate)
and returns a scalar field:

```python
div_v = grid.divergence(v_r, v_theta, v_phi)
```

### `laplacian(f)`

Takes a scalar field and returns the scalar Laplacian:

```python
lap_f = grid.laplacian(f)
```

### `curl(*v)`

For **3D grids**, takes three vector components and returns a tuple of three
arrays:

```python
curl_r, curl_theta, curl_phi = grid.curl(v_r, v_theta, v_phi)
```

For **2D grids** (e.g. `PolarGrid`), takes two vector components and returns
a single scalar array -- the out-of-plane component:

```python
curl_z = grid.curl(v_r, v_phi)   # scalar array
```

## Singularity handling

Curvilinear coordinates often have singularities where scale factors vanish --
for example $r = 0$ in spherical or cylindrical coordinates, or
$\theta = 0, \pi$ in spherical coordinates. At these points, expressions like
$1/h_i$ produce infinities.

numgrids handles this gracefully: all non-finite values (infinity, NaN) are
replaced by zero in the operator output. This means you can safely include
points near (or at) coordinate singularities, though accuracy at those
specific points will be limited.

```{tip}
For best results, avoid placing grid boundaries exactly at singularities.
Instead, start the radial axis at a small positive value like `0.01`
instead of `0`, and the polar axis at `0.01` instead of `0`.
```

## Custom curvilinear coordinates

You can define your own orthogonal coordinate system by passing scale-factor
callables directly to `CurvilinearGrid`. Each callable receives the tuple of
meshed coordinate arrays and returns an array of the same shape.

For example, *parabolic cylindrical coordinates* $(\sigma, \tau, z)$ with
scale factors $h_\sigma = h_\tau = \sqrt{\sigma^2 + \tau^2}$ and $h_z = 1$:

```python
sigma_ax = create_axis(AxisType.CHEBYSHEV, 25, 0.1, 3.0)
tau_ax   = create_axis(AxisType.CHEBYSHEV, 25, 0.1, 3.0)
z_ax     = create_axis(AxisType.CHEBYSHEV, 20, -1.0, 1.0)

grid = CurvilinearGrid(
    sigma_ax, tau_ax, z_ax,
    scale_factors=(
        lambda c: np.sqrt(c[0]**2 + c[1]**2),   # h_sigma
        lambda c: np.sqrt(c[0]**2 + c[1]**2),   # h_tau
        lambda c: np.ones_like(c[0]),             # h_z
    ),
)

Sigma, Tau, Z = grid.meshed_coords
f = Sigma**2 - Tau**2
lap_f = grid.laplacian(f)
grad_f = grid.gradient(f)
```

```{warning}
The number of scale-factor callables must match the number of axes. A
`ValueError` is raised otherwise.
```

```{note}
`CurvilinearGrid` is a subclass of `Grid`, so all standard grid
operations (refinement, boundary faces, indexing, I/O) work exactly the
same way. The built-in coordinate grids (`SphericalGrid`,
`CylindricalGrid`, `PolarGrid`) are thin subclasses of
`CurvilinearGrid` that set the scale factors for you.
```

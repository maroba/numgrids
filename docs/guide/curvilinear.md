# Curvilinear grids

Many physical problems are naturally expressed in non-Cartesian coordinates --
spherical coordinates for planetary atmospheres, cylindrical coordinates for
pipes, polar coordinates for 2D disk-shaped domains. numgrids provides
`CurvilinearGrid` and six built-in coordinate systems that give you
gradient, divergence, Laplacian, curl and volume integrals out of the box.

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

### EllipticCylindricalGrid $(\mu, \nu, z)$

$x = a\cosh\mu\cos\nu$, $y = a\sinh\mu\sin\nu$, $z = z$. Scale factors:
$h_\mu = h_\nu = a\sqrt{\sinh^2\mu + \sin^2\nu}$, $h_z = 1$. Useful for
elliptic domains, e.g. flows around elliptic cylinders.

```python
elliptic = EllipticCylindricalGrid(
    chebyshev(20, 0.0, 1.0),   # mu
    periodic(40),              # nu
    chebyshev(10, -1, 1),      # z
    a=2.0,                     # focal distance
)
```

### ProlateSpheroidalGrid and OblateSpheroidalGrid $(\mu, \nu, \phi)$

Spheroidal coordinates for elongated (prolate, $\nu\in[0,\pi]$) or
flattened (oblate, $\nu\in[-\pi/2,\pi/2]$) bodies, with focal distance
`a`:

```python
prolate = ProlateSpheroidalGrid(chebyshev(20, 0, 1), chebyshev(20, 0, np.pi), periodic(16), a=1.0)
oblate = OblateSpheroidalGrid(chebyshev(20, 0, 1), chebyshev(20, -np.pi/2, np.pi/2), periodic(16), a=1.0)
```

### Axis names

The built-in grids name their axes after the coordinates (unless you named
them yourself): `"r"`, `"theta"`, `"phi"`, `"z"`, `"mu"`, `"nu"`. So you
can write `grid.diff(f, axis="r")` or `grid.faces["r_high"]`.

## Geometry

Every curvilinear grid exposes its geometry:

```python
grid.scale_factors      # tuple of arrays h_i
grid.volume_element     # J = h_1 h_2 ... h_n
grid.cartesian_coords   # meshed Cartesian coordinates (X, Y[, Z])
```

`cartesian_coords` is used by `grid.plot(f)` to draw 2D curvilinear grids in
Cartesian space.

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

### Sparse operators and integration

For solving PDEs, the Laplacian is also available as a sparse
[Operator](operators.md):

```python
annulus = PolarGrid(chebyshev(30, 0.5, 1.0), periodic(40))
L = annulus.laplacian_operator()
u = L.solve(1.0, bcs=0)                        # Poisson problem, u = 0 on both rims
vals, modes = (-L).eig(k=6, bcs=0)             # Laplace eigenmodes
```

`grid.integrate(f)` includes the volume element, i.e. it computes
$\int f\, dV$; see [Integration](integration.md).

## Full disks, cylinders and balls

To include the center $r = 0$, use a **radial axis** as the first axis of a
`PolarGrid`, `CylindricalGrid` or `SphericalGrid`:

```python
disk = PolarGrid(radial(24, 1.0), periodic(32))
cylinder = CylindricalGrid(radial(16, 1.0), periodic(32), chebyshev(16, 0, 2))
ball = SphericalGrid(radial(16, 1.0), chebyshev(16, 1e-3, np.pi - 1e-3), periodic(32))
```

The radial points are the positive half of Chebyshev nodes on the diameter
$[-R, R]$; no point lies on the origin. Radial derivatives are computed on
the whole diameter, using that the point $(-r, \varphi)$ is the same as
$(r, \varphi + \pi)$ (and $(r, \pi - \theta, \varphi + \pi)$ on a ball). As
a result

* there is **no singularity and no boundary condition** at the center -- the
  only boundary face is the rim `"r_high"`,
* derivatives, Laplacians, integrals and interpolation (also exactly at the
  center) are **spectrally accurate**,
* operators have **no spurious eigenvalues**, and their spectrum is real and
  negative, so **time stepping is stable**.

```python
disk = PolarGrid(radial(24, 1.0), periodic(32))
vals, modes = (-disk.laplacian_operator()).eig(k=6, bcs=0)
np.sqrt(vals)       # 2.4048, 3.8317, 3.8317, 5.1356, 5.1356, 5.5201 (Bessel zeros)

X, Y = disk.cartesian_coords
u = disk.laplacian_operator().solve(4.0, bcs=1.0)    # u = x^2 + y^2
disk.integrate(np.ones(disk.shape))                  # pi
```

Requirements: the azimuthal axis must be periodic over $2\pi$ with an even
number of points, and on a `SphericalGrid` the polar axis must be symmetric
about $\pi/2$ (e.g. `chebyshev(n, eps, pi - eps)`). A `ValueError` explains
what is missing otherwise.

## Singularity handling

Curvilinear coordinates often have singularities where scale factors vanish --
for example $r = 0$ in polar, spherical or cylindrical coordinates, or
$\theta = 0, \pi$ in spherical coordinates. At these points, expressions like
$1/h_i$ produce infinities.

* **The center $r = 0$**: use a radial axis, see above.
* **The poles $\theta = 0, \pi$** of spherical grids: start the polar axis
  slightly away from the poles, e.g. `chebyshev(n, 1e-3, np.pi - 1e-3)`. The
  equation is then enforced at the points next to the poles; this is accurate
  for eigenvalue problems, and integrals miss only the tiny polar caps
  (relative error of order $\epsilon^2$).

Grid points placed *exactly* on a singularity yield non-finite coefficients,
which are replaced by zero; the resulting empty operator rows are reported by
`Operator.solve` (error) and `Operator.eig` (warning).

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
    # optional: mapping to Cartesian coordinates (for plotting)
    cartesian=lambda c: (c[0] * c[1], (c[1]**2 - c[0]**2) / 2, c[2]),
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
same way. The built-in coordinate grids are thin subclasses of
`CurvilinearGrid` that set the scale factors for you. Refinement preserves
the grid type. Custom `CurvilinearGrid` instances cannot be saved with
`save_grid` (the scale-factor functions are not serializable); the
built-in grids can.
```

"""Create docs/assets/hero.png, the title image of the README.

Usage (from the repository root):  python docs/assets/scripts/make_hero.py docs/assets/hero.png
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from numgrids import Grid, PolarGrid, SphericalSurfaceGrid, chebyshev, colatitude, periodic, radial  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/hero.png"
fig = plt.figure(figsize=(15, 5))

# 1) A vibration mode of a circular drum (full disk, through the center)
disk = PolarGrid(radial(40, 1.0), periodic(64))
vals, modes = (-disk.laplacian_operator()).eig(k=12, bcs=0)
ax = fig.add_subplot(1, 3, 1)
mode = modes[10] / np.abs(modes[10]).max()
disk.plot(mode, ax=ax, cmap="RdBu_r", colorbar=False, vmin=-1, vmax=1)
rim = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(rim), np.sin(rim), color="#7b8794", lw=1.5)
ax.set_axis_off()

# 2) An eigenfunction of the Laplacian on the full sphere (through the poles)
sphere = SphericalSurfaceGrid(colatitude(40), periodic(80))
svals, smodes = (-sphere.laplacian_operator()).eig(k=36)
# The l = 5 eigenvalue is 11-fold degenerate; pick the sectoral mode Y_5^5 from its eigenspace.
space = smodes[np.isclose(svals, 30.0)].reshape(-1, sphere.size)
TH, PH = sphere.meshed_coords
target = (np.sin(TH) ** 5 * np.cos(5 * PH)).ravel()
f = (np.linalg.lstsq(space.T, target, rcond=None)[0] @ space).reshape(sphere.shape)
# Spectral interpolation onto a fine mesh (including the poles) for a smooth picture
theta = np.linspace(0, np.pi, 121)
phi = np.linspace(0, 2 * np.pi, 241)
T, P = np.meshgrid(theta, phi, indexing="ij")
pts = np.column_stack([T.ravel(), np.mod(P.ravel(), 2 * np.pi)])
vals_s = sphere.interpolate(f, pts, method="spectral").reshape(T.shape)
vals_s /= np.abs(vals_s).max()
ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.plot_surface(np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T),
                facecolors=plt.cm.RdBu_r((vals_s + 1) / 2), rstride=1, cstride=1,
                antialiased=False, shade=False, linewidth=0)
ax.set_box_aspect([1, 1, 1], zoom=1.45)
ax.view_init(elev=20, azim=35)
ax.set_axis_off()
ax.patch.set_alpha(0)

# 3) Quantum harmonic oscillator: energy levels and eigenstates
grid = Grid(chebyshev(80, -10, 10))
x = grid.coords
H = -0.5 * grid.diff_operator(2) + 0.5 * x ** 2
energies, states = H.eig(k=6, bcs=0)
xf = np.linspace(-4.6, 4.6, 400)
ax = fig.add_subplot(1, 3, 3)
ax.fill_between(xf, 0.5 * xf ** 2, 7, color="#9aa5b1", alpha=0.18, lw=0)
ax.plot(xf, 0.5 * xf ** 2, color="#7b8794", lw=1.5)
colors = plt.cm.viridis(np.linspace(0.3, 0.95, len(energies)))
for E, psi, c in zip(energies, states, colors):
    psi = psi / np.sqrt(grid.integrate(psi ** 2))
    psi_f = grid.interpolate(psi, xf, method="spectral")
    sign = np.sign(psi_f[np.argmax(np.abs(psi_f) > 1e-3)])
    ax.plot(xf, E + 0.9 * sign * psi_f, color=c, lw=2)
    ax.axhline(E, color=c, lw=0.6, alpha=0.5)
ax.set_xlim(-4.6, 4.6)
ax.set_ylim(-0.2, 6.6)
ax.set_axis_off()

fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.05)
fig.savefig(out, dpi=110, transparent=True)
print("saved", out, "| drum sqrt(lambda) of mode:", np.sqrt(vals[10]), "| sphere eigenvalue:", round(svals[30], 8))

import numpy as np


class Axis:

    def __init__(self, num_points, low, high, periodic):
        """Base class constructor.

        Parameters
        ----------
        num_points: positive int
            Number of grid points along the axis.
        low: float
            The lower end coordinate value.
        high: float
            The upper end coordinate value.
        periodic: bool
            Apply periodic boundary condition or not.
        """
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, not {num_points}")
        assert high > low
        self._num_points = num_points
        self.periodic = bool(periodic)
        self._coords_internal = self.setup_internal_coords(low, high)
        self._coords = self.setup_external_coords(low, high)

    def setup_internal_coords(self, low, high):
        raise NotImplementedError("Must be implemented by child class.")

    def setup_external_coords(self, low, high):
        return self._coords_internal * (high - low) + low

    def __len__(self):
        """Returns the number of grid points on the axis."""
        return self._num_points

    def __getitem__(self, idx):
        """Access coordinate value for given index."""
        if self.periodic:
            return self._coords[idx % self._num_points]
        return self._coords[idx]

    @property
    def coords(self):
        return self._coords

    @property
    def coords_internal(self):
        return self._coords_internal

    def __str__(self):
        return "{}({} points from {} to {}.".format(str(type(self)), len(self), self.coords[0], self.coords[-1])

    def __repr__(self):
        """Enable nice representation in Jupyter"""
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
        ax.set_ylim(-0.01, 0.01)
        ax.set_axis_off()
        ax.plot(self.coords, [0] * len(self), "o")
        ax.plot(self.coords, [0] * len(self), "-k")
        ax.annotate("$x_0 = {}$".format(self.coords[0]), xy=(self.coords[0], 0),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    xytext=(0, 30), textcoords='offset points')
        ax.annotate("$x_{{{}}} = {}$".format(len(self) - 1, self.coords[-1]), xy=(self.coords[-1], 0),
                    arrowprops=dict(facecolor='black'),
                    xytext=(0, 30), textcoords='offset points')

        return ""


class EquidistantAxis(Axis):
    """Represents an axis with grid points spaced equidistantly.

        Can be specified as non-periodic or periodic.
    """

    def __init__(self, num_points, low=0, high=1, periodic=False):
        super(EquidistantAxis, self).__init__(num_points, low, high, periodic)

    def setup_internal_coords(self, *args):
        return np.linspace(0, 1, len(self), endpoint=not self.periodic)

    @property
    def spacing(self):
        return self._coords[1] - self._coords[0]

    def __repr__(self):
        """Visualize the grid."""
        if not self.periodic:
            return super().__repr__()
        from matplotlib import pyplot as plt
        import matplotlib.patches as mp
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        angles = np.linspace(0, 2 * np.pi, len(self), endpoint=False)
        xx = np.cos(angles)
        yy = np.sin(angles)
        ax.plot(xx, yy, "o")
        ax.add_patch(mp.Circle((0, 0), 1, fill=False))

        style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color="k")

        arrow = mp.FancyArrowPatch((xx[1] * 1.1, yy[1] * 1.1), (xx[3] * 1.1, yy[3] * 1.1),
                                   connectionstyle="arc3,rad=0.2", **kw)

        ax.add_patch(arrow)
        ax.annotate("$x_0 = {} = x_{{{}}}$".format(self.coords[0], len(self)), xy=(xx[0], yy[0]),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    xytext=(30, 30), textcoords='offset points')
        ax.annotate("$x_{{{}}} = {}$".format(len(self) - 1, self.coords[-1]), xy=(xx[-1], yy[-1]),
                    arrowprops=dict(facecolor='black'),
                    xytext=(30, -30), textcoords='offset points')
        return ""


class ChebyshevAxis(Axis):
    """Represents an axis with grid points localized at the Chebyshev points.

        Allows using spectral methods even for non-periodic functions.
    """

    def __init__(self, num_points, low=0, high=1):
        super(ChebyshevAxis, self).__init__(num_points, low, high, periodic=False)

    def setup_internal_coords(self, *args):
        n = len(self)
        return np.cos(np.arange(n) * np.pi / (n - 1))

    def setup_external_coords(self, low, high):
        coords = (self._coords_internal[::-1] + 1) / 2
        return coords * (high - low) + low


class LogAxis(Axis):
    """Represents an axis with grid points spaced logarithmically."""

    def __init__(self, num_points, low, high):
        if low <= 0:
            raise ValueError("LogAxis requires positive lower boundary.")
        super(LogAxis, self).__init__(num_points, low, high, periodic=False)

    def setup_internal_coords(self, low, high):
        return np.linspace(np.log(low), np.log(high), len(self))

    def setup_external_coords(self, low, high):
        return np.logspace(np.log10(low), np.log10(high), len(self))

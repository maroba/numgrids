import numpy as np


class Grid:

    def __init__(self, *axes):
        self.axes = axes
        self.ndims = len(axes)
        self.meshed_coords = np.meshgrid(
            *tuple(a.coords for a in axes), indexing="ij")

        bdry = np.ones(self.shape, dtype=bool)
        mask = [axis.boundary for axis in self.axes]
        bdry[tuple(mask)] = False
        self._boundary = bdry

    def get_axis(self, idx=0):
        return self.axes[idx]

    def __getitem__(self, inds):
        if self.ndims == 1:
            return self.axes[0][inds]
        return np.array(
            tuple(self.axes[k][inds[k]] for k in range(self.ndims))
        )

    @property
    def coords(self):
        if self.ndims == 1:
            return self.axes[0].coords
        return tuple(a.coords for a in self.axes)

    @property
    def shape(self):
        return tuple(len(axis) for axis in self.axes)

    def __repr__(self):
        import matplotlib.pyplot as plt

        nplots = self.ndims * (self.ndims - 1) // 2
        fig = plt.figure(figsize=(12, nplots*8))

        ctr = 0
        for i in range(self.ndims):
            for j in range(i):
                ctr += 1
                ax = fig.add_subplot(nplots, 1, ctr)
                X, Y = self.meshed_coords[i], self.meshed_coords[j]
                ax.plot(X.reshape(-1), Y.reshape(-1), "o", ms=1)

                if self.axes[i].name:
                    ax.set_xlabel("${}$".format(self.axes[i].name))
                else:
                    ax.set_xlabel(f"axis-{i}")

                if self.axes[j].name:
                    ax.set_ylabel("${}$".format(self.axes[j].name))
                else:
                    ax.set_ylabel(f"axis-{j}")

        return ""

    @property
    def boundary(self):
        """
        Returns a binary mask of the shape of the grid indicating the
        grid points on the boundary.
        """
        return self._boundary

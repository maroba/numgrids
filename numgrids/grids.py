import numpy as np


class Grid:

    def __init__(self, *axes):
        self.axes = axes
        self.ndims = len(axes)
        self.meshed_coords = np.meshgrid(
            *tuple(a.coords for a in axes), indexing="ij")

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
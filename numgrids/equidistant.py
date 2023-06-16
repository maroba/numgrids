import numpy as np


class EquidistantAxis:

    def __init__(self, num_points, low=0, high=1, periodic=False):
        assert high > low
        self._num_points = num_points
        self._coords_internal = np.linspace(0, 1, num_points)
        self._coords = self._coords_internal * (high - low) + low
        self.periodic = bool(periodic)

    def __len__(self):
        return self._num_points

    def __getitem__(self, idx):
        if self.periodic:
            return self._coords[idx % self._num_points]
        return self._coords[idx]

    @property
    def coords(self):
        return self._coords


class Grid:

    def __init__(self, *axes):
        self.axes = axes
        self.ndims = len(axes)

    def get_axis(self, idx=0):
        return self.axes[idx]

    def __getitem__(self, inds):
        if self.ndims == 1:
            return self.axes[0][inds]
        return np.array(
            tuple(self.axes[k][inds[k]] for k in range(self.ndims))
        )

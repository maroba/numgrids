import numpy as np


class EquidistantAxis:

    def __init__(self, num_points, low=0, high=1, periodic=False):
        assert high > low
        self._num_points = num_points
        self._coords_internal = np.linspace(0, 1, num_points)
        self._coords = self._coords_internal * (high - low) + low
        self.periodic = bool(periodic)

    def __len__(self):
        """Returns the number of grid points on the axis."""
        return self._num_points

    def __getitem__(self, idx):
        if self.periodic:
            return self._coords[idx % self._num_points]
        return self._coords[idx]

    def get_coordinate(self, idx):
        return self[idx]

    @property
    def coords(self):
        return self._coords

    @property
    def spacing(self):
        return self._coords[1] - self._coords[0]

class ChebyshevAxis:

    def __init__(self):
        raise NotImplementedError


class LogAxis:

    def __init__(self):
        raise NotImplementedError

import numpy as np


class Axis:

    def __init__(self, num_points, low, high, periodic):
        assert high > low
        self._num_points = num_points
        self.periodic = bool(periodic)
        self._coords_internal = self.setup_internal_coords()
        self._coords = self.setup_external_coords(low, high)

    def setup_internal_coords(self):
        raise NotImplementedError("Must be implemented by child class.")

    def setup_external_coords(self, low, high):
        return self._coords_internal * (high - low) + low

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
    def coords_internal(self):
        return self._coords_internal


class EquidistantAxis(Axis):

    def __init__(self, num_points, low=0, high=1, periodic=False):
        super(EquidistantAxis, self).__init__(num_points, low, high, periodic)

    def setup_internal_coords(self):
        return np.linspace(0, 1, len(self), endpoint=not self.periodic)

    @property
    def spacing(self):
        return self._coords[1] - self._coords[0]


class ChebyshevAxis(Axis):

    def __init__(self, num_points, low=0, high=1):
        super(ChebyshevAxis, self).__init__(num_points, low, high, periodic=False)

    def setup_internal_coords(self):
        n = len(self)
        return np.cos(np.arange(n) * np.pi / (n-1))

    def setup_external_coords(self, low, high):
        coords = (self._coords_internal[::-1] + 1) / 2
        return coords * (high - low) + low


class LogAxis:

    def __init__(self):
        raise NotImplementedError

import numpy as np


class Grid:
    """
    Represents a numerical grid.
    """

    def __init__(self, *axes):
        """
        Constructor

        Parameters
        ----------
        axes: one or more Axis objects
            One or more axes defining the grid. The order of the axes matters!
            The first axis passed will have index 0.
        """
        self._axes = axes
        self.ndims = len(axes)
        self._meshed_coords = np.meshgrid(
            *tuple(a.coords for a in axes), indexing="ij")

        bdry = np.ones(self.shape, dtype=bool)
        mask = [axis.boundary for axis in self.axes]
        bdry[tuple(mask)] = False
        self._boundary = bdry

        # The functions in the api module can store entities here for re-use:
        self.cache = {
            "diffs": {},
            "integrals": None,
            "interpolators": {}
        }

    @property
    def size(self):
        return np.prod(self.shape)

    def get_axis(self, idx=0):
        """
        Returns the axis with given index.

        Parameters
        ----------
        idx: int
            The index of the requested axis.

        Returns
        -------
        The requested Axis object.
        """
        return self.axes[idx]

    def __getitem__(self, inds):
        if self.ndims == 1:
            return self.axes[0][inds]
        return np.array(
            tuple(self.axes[k][inds[k]] for k in range(self.ndims))
        )

    @property
    def axes(self):
        """
        Returns a list with the axes objects of the grid.
        """
        return self._axes

    @property
    def coords(self):
        """
        Returns a tuple of lists with the coordinate values along each axis.
        In case of 1D, only a single list is returned.
        """
        if self.ndims == 1:
            return self.axes[0].coords
        return tuple(a.coords for a in self.axes)

    @property
    def meshed_coords(self):
        """
        Returns the a tuple with the meshed coordinate values.
        """
        return self._meshed_coords

    @property
    def shape(self):
        """
        Returns a tuple with the number of grid points along each axis.
        """
        return tuple(len(axis) for axis in self.axes)

    def __repr__(self):
        import matplotlib.pyplot as plt

        nplots = self.ndims * (self.ndims - 1) // 2
        fig = plt.figure(figsize=(12, nplots * 8))

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

    def refine(self):
        """
        Returns a new grid with twice the number of grid points in each direction.
        """
        new_axes = []
        for axis in self.axes:
            cls = type(axis)
            x = axis.coords
            new_axes.append(
                cls(len(axis) * 2, x[0], x[-1])
            )
        return Grid(*new_axes)

    def coarsen(self):
        """
        Returns a new grid with half the number of grid points in each direction.
        """
        new_axes = []
        for axis in self.axes:
            cls = type(axis)
            x = axis.coords
            new_axes.append(
                cls(len(axis) // 2, x[0], x[-1])
            )
        return Grid(*new_axes)

    @property
    def meshed_indices(self):
        """
        Returns a tuple of length grid.ndims, where each item in the
        tuple is an array of shape grid.shape. Each item stores the
        indices of each grid point for a given axis.

        For example, if you have a 2D grid with 4 points along axis 0
        and 3 points along axis 1, the indices along axis 0 are [0, 1, 2, 3],
        the indices along axis 1 are [0, 1, 2]. meshed_indices returns
        a tuple (I, J), where I and J both have shape (4, 3). I contains
        the "slot-0" indices, J contains the "slot-1" indices.

        This generalizes for grids in higher dimensions, of course.
        """
        return np.meshgrid(*[np.arange(len(axis)) for axis in self.axes], indexing="ij")

    @property
    def index_tuples(self):
        """
        Returns an array A of shape (*grid.shape, grid.ndims).

        For example, in case of a 2D grid, A[i, j] contains the index
        tuple (i, j).
        """
        return self._to_tuple_field(*self.meshed_indices)

    @property
    def coord_tuples(self):
        """
        Returns an array A of shape (*grid.shape, grid.ndims) with
        all the coordinate tuples.

        For example, in case of a 2D grid, A[i, j] contains the
        tuple (x, y), where x and y are the coordinates of the grid point
        (i, j).
        """
        return self._to_tuple_field(*self.meshed_coords)

    def _to_tuple_field(self, *arrs):
        return np.vstack([arr.reshape(-1) for arr in arrs]).T.reshape(
            *self.shape, -1
        )


class MultiGrid:
    """
    Represents a multigrid, a hierarchical set of grids of different resolutions
    for the same domain.
    """

    def __init__(self, *axes, min_size=2):
        """
        Constructor.

        The grids of the various resolutions are stored in the property `levels`.

        Parameters
        ----------
        axes: one or more Axis objects
            The axes for the grid with highest resolution. Starting from the grid
            with the highest resolution (level 0), the constructor will create different levels
            of resolution. Each additional level contains a grid with about half the grid
            points in each direction.

        min_size: positive int
            This parameter sets a lower bound for the coarsest grid in the multigrid.
            At the lowest resolution, each axis must have at least min_size grid points.
        """
        grid = Grid(*axes)
        self._levels = [grid]
        while True:
            sizes = np.array(grid.shape)
            # size -> size / 2 if size is even and size -> (size+1)/2 if it is odd:
            sizes = sizes // 2 + sizes % 2
            if min(sizes) < min_size:
                break
            axes = [type(axis)(size, axis.coords[0], axis.coords[-1])
                    for size, axis in zip(sizes, axes)]
            grid = Grid(*axes)
            self._levels.append(grid)

    @property
    def levels(self):
        """
        Returns a list of grids with different resolution levels.

        levels[0] => finest grid
        levels[-1] => coarsest grid
        """
        return self._levels

    def transfer(self, f, level_from, level_to, method="linear"):
        """
        Coarsen or refine an array on a grid by one level.

        Parameters
        ----------
        f: numpy.ndarray
            An array given on the grid at level = level_from.
        level_from: int
            The level of the grid on which f is given.
        level_to: int
            The level to which f shall be refined or coarsened.
        method: str
            Coarsening and refinement involves interpolation. This
            is the interpolation order.

        Returns
        -------
        The coarsened or refined array.
        """
        from numgrids import Interpolator
        assert f.shape == self.levels[level_from].shape
        assert abs(level_from - level_to) == 1

        grid_fr = self.levels[level_from]
        grid_to = self.levels[level_to]

        interp = Interpolator(grid_fr, f, method=method)
        f_trans = interp(grid_to)

        return f_trans

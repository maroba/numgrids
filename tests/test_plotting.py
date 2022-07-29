import unittest

from matplotlib import pyplot as plt

from numgrids.plotting import plot
from numgrids.shapes import Parallelepiped


class TestPlotting(unittest.TestCase):
    def test_plot_parallelepiped_2d(self):
        rect = Parallelepiped((1, 1), (1, -2), (1, 1))
        plot(rect)
        plt.show()


if __name__ == '__main__':
    unittest.main()

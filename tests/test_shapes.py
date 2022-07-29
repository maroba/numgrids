import unittest

from numgrids.shapes import Parallelepiped


class TestRectangle(unittest.TestCase):

    def test_rectangle_size_ambient_2(self):
        rect = Parallelepiped((1, 1), (1, -1), (1, 1))
        self.assertAlmostEqual(2, rect.area)

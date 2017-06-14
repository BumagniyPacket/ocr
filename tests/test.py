import unittest

import numpy as np
import os

from config import MAX_SIZE, WORK_PATH
from ocr import io


class TestIO(unittest.TestCase):
    def test_open_image(self):
        image = io.img_open(os.path.join(WORK_PATH, 'tests', 'test_image', 'image.jpg'))
        self.assertEqual(len(image.shape), 2)

    def test_open_image_color_map(self):
        image = io.img_open(os.path.join(WORK_PATH, 'tests', 'test_image', 'image.jpg'))
        self.assertTrue(0 <= image[5][5] <= 1)

    def test_downscale_big_image(self):
        image = np.ones((2050, 2050))
        downscaled = io.downscale(image)
        self.assertLessEqual(max(downscaled.shape), MAX_SIZE)

    def test_downscale_default_image(self):
        image = np.ones((100, 100))
        downscaled = io.downscale(image)
        self.assertEqual(image.shape, downscaled.shape)


if __name__ == '__main__':
    unittest.main()

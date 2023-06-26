import cv2
import numpy as np
import unittest

from mushrooms import MushroomGarden
from os.path import join

class TestMushrooms(unittest.TestCase):
    def test_get_sample_pixels(self):
        sample = cv2.imread(join('test_images', 'colour_block.png'))
        unique_pixels = np.array([
            [0, 0, 255],
            [20, 0, 255],
            [0, 255, 5],
            [217, 217, 217],
            [255, 255, 255],
            [0, 0, 0],
            [0, 245, 255],
            [144, 0, 142],
            [0, 72, 134]
        ])
        mg = MushroomGarden()

        pixels = mg.get_sample_pixels(sample)
        self.assertTrue(np.equal(unique_pixels, pixels).all)

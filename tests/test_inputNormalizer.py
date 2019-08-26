from unittest import TestCase
import imageio
import numpy as np

from sources.input_normalizer import InputNormalizer


class TestInputNormalizer(TestCase):

    def test_normalize_state(self):
        normalizer = InputNormalizer((84, 84))

        initial_state = imageio.imread('resources/output.png')
        normalized_state = normalizer.normalize_state(initial_state)

        self.assertEquals(normalized_state.shape, (1, 84 * 84))
        self.assertTrue((normalized_state <= 1).all())
        self.assertTrue(np.array_equal(normalized_state, np.load("resources/output_normalized.npy", allow_pickle=True)))

    def test_normalize_input(self):
        normalizer = InputNormalizer((84, 84))

        normalized_input = normalizer.normalize_input(
            frame=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), current_action=0.5)

        self.assertTrue(
            np.array_equal(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0.5]]), normalized_input))

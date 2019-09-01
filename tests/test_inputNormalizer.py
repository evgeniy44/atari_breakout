from unittest import TestCase
import imageio
import numpy as np

from sources.input_normalizer import InputNormalizer


class TestInputNormalizer(TestCase):

    def test_normalize_state(self):
        normalizer = InputNormalizer((84, 84), total_actions=4)

        initial_state = imageio.imread('../resources/output.png')
        normalized_state = normalizer.normalize_state(initial_state)

        self.assertEquals(normalized_state.shape, (84, 84))
        self.assertTrue((normalized_state <= 1).all())

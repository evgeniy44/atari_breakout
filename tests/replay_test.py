from unittest import TestCase

from sources.replay import ReplayMemory

import numpy as np


class ReplayTest(TestCase):

    def test_observe(self):
        replay = ReplayMemory(observation_size=2, max_size=100)

        state = np.array([[0.11, 0.12], [0.13, 0.14]], dtype=np.float32)

        replay.observe(state=state, action=0.5, reward=1, done=False)

        self.assertTrue(np.array_equal(replay.samples['state'][0], state))
        self.assertEquals(replay.samples['reward'][0, 0], 1)
        self.assertEquals(replay.samples['terminal'][0, 0], False)
        self.assertEquals(replay.samples['action'][0, 0], 0.5)
        self.assertEquals(replay.num_observed, 1)

    def test_sample(self):
        np.random.seed(1)
        replay = ReplayMemory(observation_size=2, max_size=100)

        replay.observe(state=np.array([[0.11, 0.12], [0.13, 0.14]]), action=0.5, reward=1, done=False)
        replay.observe(state=np.array([[0.21, 0.22], [0.23, 0.24]]), action=0.6, reward=0, done=False)
        replay.observe(state=np.array([[0.31, 0.32], [0.33, 0.34]]), action=0.7, reward=0, done=False)
        replay.observe(state=np.array([[0.41, 0.42], [0.43, 0.44]]), action=0.8, reward=1, done=True)
        replay.observe(state=np.array([[0.51, 0.52], [0.53, 0.54]]), action=0.9, reward=0, done=True)

        (state, action, reward, s_next, is_terminal) = replay.sample_minibatch(minibatch_size=2, frame_size=2)

        self.assertTrue(np.array_equal(state[0, :, :, 0], np.array([[0.11, 0.12], [0.13, 0.14]], dtype=np.float32)))
        self.assertTrue(np.array_equal(state[0, :, :, 1], np.array([[0.21, 0.22], [0.23, 0.24]], dtype=np.float32)))
        self.assertTrue(np.array_equal(s_next[0, :, :, 0], np.array([[0.21, 0.22], [0.23, 0.24]], dtype=np.float32)))
        self.assertTrue(np.array_equal(s_next[0, :, :, 1], np.array([[0.31, 0.32], [0.33, 0.34]], dtype=np.float32)))
        self.assertEquals(round(float(action[0]), 3), float(0.6))
        self.assertEquals(reward[0, 0], 0.0)
        self.assertEquals(is_terminal[0], False)

        self.assertTrue(np.array_equal(state[1, :, :, 0], np.array([[0.31, 0.32], [0.33, 0.34]], dtype=np.float32)))
        self.assertTrue(np.array_equal(state[1, :, :, 1], np.array([[0.41, 0.42], [0.43, 0.44]], dtype=np.float32)))
        self.assertTrue(np.array_equal(s_next[1, :, :, 0], np.array([[0.41, 0.42], [0.43, 0.44]], dtype=np.float32)))
        self.assertTrue(np.array_equal(s_next[1, :, :, 1], np.array([[0.51, 0.52], [0.53, 0.54]], dtype=np.float32)))
        self.assertEquals(round(float(action[1]), 3), float(0.8))
        self.assertEquals(reward[1, 0], 1.0)
        self.assertEquals(is_terminal[1], True)
